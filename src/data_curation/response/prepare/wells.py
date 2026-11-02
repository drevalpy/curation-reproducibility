"""What a preparer returns, and how it becomes the well table ``drevalpy.curation.curate()`` reads.

A **series** is one dose-response series of one (sample, drug) on one plate: the wells a
dataset's :class:`SeriesSpec` ``key`` columns group together. Each series becomes one
replicate. An **experiment** is one independently seeded culture: the series of one experiment
are technical replicates of each other, different experiments of a pair biological ones.

The preparers keep the dataset's raw column names and describe them with a spec; :func:`to_wells`
renames them into curate's input columns (:data:`WELL_COLUMNS`). Everything else about series -
labels, repeat-block splitting, averaging, ``min_doses``, the replicate numbering - is curate's
(``drevalpy.curation._series``), so this repository and a user's own screen are curated alike.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd

#: The columns ``curate()`` reads; every other column of a well table is per-series metadata.
WELL_COLUMNS = ("sample", "drug", "concentration", "intensity", "experiment", "replicate", "position")

_KEY_SEPARATOR = ":"


@dataclass(frozen=True)
class SeriesSpec:
    """How one dataset's wells group into series.

    Attributes:
        sample: Column holding the native sample identifier.
        drug: Column holding the native drug identifier.
        key: Columns that, with sample and drug, identify one series (e.g. a plate). Empty
            when every (sample, drug) is a single series.
        experiment: Columns that, with sample and drug, identify one experiment (one independently
            seeded culture). Series of the same experiment are technical replicates, different
            experiments biological ones.
        metadata: Experiment-level columns to carry into the replicate-fits table (lowercased).
        position: Column ordering the wells of a series, used to split repeat blocks. Without
            it repeated doses are always averaged.
    """

    sample: str
    drug: str
    key: tuple[str, ...] = ()
    experiment: tuple[str, ...] = ()
    metadata: tuple[str, ...] = ()
    position: str | None = None

    @property
    def columns(self) -> list[str]:
        """Every column :func:`to_wells` reads from the wells frame, in first-mention order."""
        named = [self.sample, self.drug, "concentration", "intensity", *self.key, *self.experiment, *self.metadata]
        if self.position is not None:
            named.append(self.position)
        return list(dict.fromkeys(named))


class Wells(NamedTuple):
    """A preparer's output: well-level measurements and how they group into series."""

    frame: pd.DataFrame
    """``concentration``, ``intensity`` and every column :attr:`spec` names."""

    spec: SeriesSpec


def to_wells(wells: Wells) -> pd.DataFrame:
    """Rename a preparer's wells into curate's input columns.

    The ``key`` columns are joined with ``:`` into ``replicate`` and the ``experiment`` columns
    into ``experiment``; a missing part renders as an empty string and an integral float loses
    its ``.0``. Specs without key or experiment columns leave that column out, which curate
    reads as one series per pair or one experiment per pair. The metadata columns keep their
    raw names.

    Args:
        wells: A preparer's output.

    Returns:
        The well table: the :data:`WELL_COLUMNS` the spec defines, then the metadata columns.
    """
    frame, spec = wells
    out = pd.DataFrame(
        {
            "sample": frame[spec.sample].to_numpy(),
            "drug": frame[spec.drug].to_numpy(),
            "concentration": frame["concentration"].to_numpy(),
            "intensity": frame["intensity"].to_numpy(),
        }
    )
    if spec.experiment:
        out["experiment"] = _joined(frame, spec.experiment)
    if spec.key:
        out["replicate"] = _joined(frame, spec.key)
    if spec.position is not None:
        out["position"] = frame[spec.position].to_numpy()
    for column in spec.metadata:
        out[column] = frame[column].to_numpy()
    for column in out.columns:
        out[column] = _uniform(out[column])
    return out


def metadata_columns(wells: pd.DataFrame) -> list[str]:
    """The metadata columns of a well table: every column curate does not read itself."""
    return [column for column in wells.columns if column not in WELL_COLUMNS]


def _joined(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Categorical:
    """Join *columns* row-wise into one categorical string key."""
    parts = [_as_category(frame[column]) for column in columns]
    codes = pd.MultiIndex.from_arrays([part.codes for part in parts])
    ids, combinations = codes.factorize()
    labels = [
        _KEY_SEPARATOR.join(
            str(part.categories[code]) if code >= 0 else "" for part, code in zip(parts, combo, strict=True)
        )
        for combo in combinations
    ]
    unique_labels, inverse = np.unique(np.asarray(labels, dtype=object).astype(str), return_inverse=True)
    return pd.Categorical.from_codes(inverse[ids], categories=pd.Index(unique_labels, dtype=object))


def _as_category(values: pd.Series) -> pd.Categorical:
    """A column as categorical strings (integral floats lose their ``.0``), missing values kept missing."""
    codes, uniques = pd.factorize(values)
    if len(uniques) == 0:
        return pd.Categorical.from_codes(codes, categories=pd.Index([], dtype=object))
    rendered = np.array([_render(value) for value in uniques], dtype=object)
    labels, inverse = np.unique(rendered.astype(str), return_inverse=True)
    mapped = np.where(codes >= 0, inverse[np.maximum(codes, 0)], -1)
    return pd.Categorical.from_codes(mapped, categories=pd.Index(labels, dtype=object))


def _uniform(values: pd.Series) -> pd.Series:
    """*values*, or, if it mixes Python types (e.g. ``int`` and ``str`` IDs), its values as strings.

    Parquet stores one type per column. A mixed column renders like a key part (integral floats
    lose their ``.0``), which is how curate reads identifiers anyway; missing values stay missing.
    """
    if values.dtype != object or len({type(v) for v in values.dropna().unique()}) <= 1:
        return values
    codes, uniques = pd.factorize(values)
    rendered = np.array([_render(value) for value in uniques], dtype=object)
    out = np.full(len(values), None, dtype=object)
    out[codes >= 0] = rendered[codes[codes >= 0]]
    return pd.Series(out, index=values.index, dtype=object)


def _render(value: object) -> str:
    """Render one key value as a string, dropping a float's ``.0``."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()
