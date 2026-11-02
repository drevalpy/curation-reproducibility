"""Backed-mode reading of assembled MuData files into small snapshots.

The assembled files run to several hundred MB each, almost all of it omics
matrices. The report only needs which samples each modality covers, which is
metadata, plus measurement counts from the response matrices. Every file is
therefore opened backed and reduced to a :class:`DatasetSnapshot` before the next
one is opened, so at most one file handle - and never a full omics matrix - is
held at a time.

Replicate counts come from the response's replicate-fits table
(``uns["replicate_fits"]``, see :mod:`data_curation.response.replicate_fits`).
Backed mode still reads ``uns`` into memory, but the table is long rather than
wide and only four of its columns are used.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import mudata as md
import numpy as np
import pandas as pd
from drevalpy.curation import Status
from upath import UPath

from data_curation.core import layout
from data_curation.core.config import processing_config
from data_curation.core.errors import CurationError
from data_curation.response import replicate_fits

md.set_options(pull_on_update=False)

ASSEMBLED_DIR = layout.ASSEMBLED
RESPONSE_KEY = "response"
TISSUE_COLUMN = "tissue"


@dataclass(frozen=True)
class ReplicateCounts:
    """Replicate structure of one response modality; see :func:`replicate_counts`.

    Attributes:
        samples: Samples with at least one replicated pair.
        n_pairs: (sample, drug) pairs with two or more fitted series.
        n_biological_pairs: Replicated pairs with fitted series in two or more biological replicates.
        n_technical_only_pairs: Replicated pairs whose fitted series all belong to one biological replicate.
        median_biological: Median number of fitted biological replicates per replicated pair.
        median_technical: Median number of fitted series per biological replicate of a replicated pair.
    """

    samples: pd.Index = field(default_factory=lambda: pd.Index([]))
    n_pairs: int = 0
    n_biological_pairs: int = 0
    n_technical_only_pairs: int = 0
    median_biological: float = float("nan")
    median_technical: float = float("nan")


@dataclass(frozen=True)
class DatasetSnapshot:
    """Everything the report needs from one assembled dataset.

    Attributes:
        name: Dataset name, the file stem.
        response_samples: Samples with at least one finite response value.
        n_drugs: Drugs in the response modality.
        n_measured: Finite values in the response matrix.
        tissues: Tissue per response sample (``"Unknown"`` when missing).
        response_drugs: Drugs with at least one finite response value.
        omics_samples: Per omics modality, the samples it covers.
        omics_features: Per omics modality, its number of features.
        replicates: Replicate structure of the response's replicate-fits table.
    """

    name: str
    response_samples: pd.Index
    n_drugs: int
    n_measured: int
    tissues: pd.Series
    response_drugs: pd.Index = field(default_factory=lambda: pd.Index([]))
    omics_samples: dict[str, pd.Index] = field(default_factory=dict)
    omics_features: dict[str, int] = field(default_factory=dict)
    replicates: ReplicateCounts = field(default_factory=ReplicateCounts)


def replicate_counts(table: pd.DataFrame | None) -> ReplicateCounts:
    """Describe the replicate structure of a replicate-fits table.

    Only fitted rows count. A (sample, drug) pair is replicated when two or more of its
    series were fitted; a pair with a single fitted series carries only the pooled fit and
    is no replicate. Its fitted series are then grouped by ``biological_replicate``.

    Args:
        table: The ``uns["replicate_fits"]`` table, or ``None`` when absent.

    Returns:
        The replicated pairs, split by whether they have biological replicates, and the
        median replicate counts at both levels.
    """
    if table is None or table.empty:
        return ReplicateCounts()
    pair = [layout.SAMPLE_INDEX_NAME, replicate_fits.DRUG_COLUMN]
    fitted = table.loc[
        (table["status"].astype(str) == Status.FITTED).to_numpy(),
        [*pair, "biological_replicate"],
    ]
    fitted = fitted.assign(**{column: fitted[column].astype(str) for column in pair})
    n_fitted = fitted.groupby(pair)[pair[0]].transform("size")
    replicated = fitted.loc[(n_fitted >= 2).to_numpy()]
    if replicated.empty:
        return ReplicateCounts()
    per_biological = replicated.groupby([*pair, "biological_replicate"]).size()
    n_biological = per_biological.groupby(level=pair).size()
    return ReplicateCounts(
        samples=pd.Index(replicated[pair[0]].unique()),
        n_pairs=len(n_biological),
        n_biological_pairs=int((n_biological >= 2).sum()),
        n_technical_only_pairs=int((n_biological == 1).sum()),
        median_biological=float(n_biological.median()),
        median_technical=float(per_biological.median()),
    )


def discover_datasets(assembled_dir: UPath = ASSEMBLED_DIR, exclude: tuple[str, ...] | None = None) -> list[UPath]:
    """List the assembled ``.h5mu`` files, leaving out the toy dataset.

    Args:
        assembled_dir: Directory holding the assembled files.
        exclude: Dataset names to skip. Defaults to the configured toy dataset.

    Returns:
        Paths sorted by dataset name.
    """
    if exclude is None:
        exclude = (str(processing_config("toy")["name"]),)
    return sorted((p for p in assembled_dir.glob("*.h5mu") if p.stem not in exclude), key=lambda p: p.stem)


@contextmanager
def open_backed(path: UPath) -> Iterator[md.MuData]:
    """Open a MuData file read-only in backed mode and close it on exit."""
    mdata = md.read_h5mu(str(path), backed="r")
    try:
        yield mdata
    finally:
        mdata.file.close()


def finite_counts(matrix, chunk_rows: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Count finite values per row and per column of a (possibly backed) dense matrix.

    Reads ``chunk_rows`` rows at a time, so memory stays flat regardless of the
    matrix size, and both axes come out of the same single pass.

    Args:
        matrix: Anything sliceable by row ranges into a 2-D array, e.g. an
            ``h5py.Dataset`` or a NumPy array.
        chunk_rows: Rows read per block.

    Returns:
        Per-row and per-column counts of finite values.
    """
    n_rows, n_cols = matrix.shape
    rows = np.zeros(n_rows, dtype=np.int64)
    cols = np.zeros(n_cols, dtype=np.int64)
    for start in range(0, n_rows, chunk_rows):
        finite = np.isfinite(np.asarray(matrix[start : start + chunk_rows]))
        rows[start : start + finite.shape[0]] = finite.sum(axis=1)
        cols += finite.sum(axis=0)
    return rows, cols


def count_finite(matrix, chunk_rows: int = 256) -> np.ndarray:
    """Count finite values per row; see :func:`finite_counts`."""
    return finite_counts(matrix, chunk_rows)[0]


def _measured(mdata: md.MuData, key: str) -> tuple[pd.Index, pd.Index, int]:
    """Return the rows and columns with any finite value in one modality, and the total count."""
    modality = mdata.mod[key]
    rows, cols = finite_counts(modality.X)
    return pd.Index(modality.obs_names[rows > 0]), pd.Index(modality.var_names[cols > 0]), int(rows.sum())


def read_snapshot(path: UPath) -> DatasetSnapshot:
    """Reduce one assembled file to a :class:`DatasetSnapshot`.

    Args:
        path: Assembled ``.h5mu`` file.

    Returns:
        The snapshot; the file is closed again before returning.

    Raises:
        CurationError: If the file has no response modality.
    """
    with open_backed(path) as mdata:
        if RESPONSE_KEY not in mdata.mod:
            msg = f"{path} has no '{RESPONSE_KEY}' modality"
            raise CurationError(msg)
        samples, drugs, n_measured = _measured(mdata, RESPONSE_KEY)
        if TISSUE_COLUMN in mdata.obs.columns:
            raw = mdata.obs[TISSUE_COLUMN].reindex(samples).astype("object")
            tissues = raw.where(raw.notna() & (raw.astype(str).str.strip() != ""), "Unknown")
        else:
            tissues = pd.Series("Unknown", index=samples, dtype="object")

        omics = {k: m for k, m in mdata.mod.items() if not k.startswith(RESPONSE_KEY)}
        replicates = replicate_counts(mdata.mod[RESPONSE_KEY].uns.get(replicate_fits.KEY))

        return DatasetSnapshot(
            name=path.stem,
            response_samples=samples,
            n_drugs=int(mdata.mod[RESPONSE_KEY].n_vars),
            n_measured=n_measured,
            tissues=tissues.astype(str),
            response_drugs=drugs,
            omics_samples={k: pd.Index(m.obs_names) for k, m in omics.items()},
            omics_features={k: int(m.n_vars) for k, m in omics.items()},
            replicates=replicates,
        )


def read_snapshots(paths: list[UPath]) -> list[DatasetSnapshot]:
    """Read one snapshot per file, opening the files one after another."""
    return [read_snapshot(path) for path in paths]
