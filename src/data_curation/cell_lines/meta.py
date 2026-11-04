"""Build the global sample metadata tables consumed by MuData assembly.

drevalpy reads sample annotations from ``mdata.obs``, indexed by a unique sample
ID, and requires only the ``tissue`` column. ``cellosaurus_id`` and
``cell_line_name`` are optional display columns that only datasets whose samples
are cell lines carry. ``mdata.obs`` spans the union of every modality, which is
wider than any single one - GDSC2 has 1675 cell lines against gene expression's
1584 - so the table is materialised up front and joined at assembly time on its
``sample_id`` column. Datasets with Cell Model Passports metadata share one
table; the others get their own.

Cell Model Passports is the primary source for cell lines. Of its 98 columns the
pipeline used 3; the ones added here are the ones measured at full coverage over
the cell lines these datasets actually screen. ``patient_id`` is the load-bearing
one: 1931 distinct patients across 2266 models means some cell lines share a
donor, which is a leakage channel the splitters could not previously see.

Three sources are supported:

* Cell Model Passports ``model_list.csv``, which carries the annotations and the
  Cellosaurus accession in ``RRID``. The sample ID is that accession, and the
  table also carries ``cellosaurus_id`` and ``cell_line_name``.
* Datasets with no CMP entry at all (patient specimens, PDX models), where tissue
  is a dataset-level constant supplied as ``datasets.<dataset>.tissue`` in
  ``config/sources.yaml`` and the table holds nothing but ``sample_id`` and
  ``tissue``: they have no Cellosaurus accession or cell line name, so none is
  written.
* Datasets in the ``standard`` input format (``sample_metadata: standard``), which
  ship their own ``samples.parquet`` (see :mod:`data_curation.core.standard`) with
  ``sample_id``, ``tissue`` and any further metadata columns.

Cellosaurus contributes only what CMP lacks - the contamination and
misidentification flags - and never overrides it.
"""

from collections.abc import Iterable

import anndata as ad
import pandas as pd
from upath import UPath

from data_curation.cell_lines.cellosaurus import QUALITY_COLUMNS, quality_flags
from data_curation.core import layout, standard
from data_curation.core.config import dataset_config
from data_curation.core.errors import CurationError

#: The join key of every metadata table; it becomes the ``mdata.obs`` index.
SAMPLE_ID_COLUMN = "sample_id"

#: The one column drevalpy requires, under exactly this name.
REQUIRED_COLUMNS = ["tissue"]

#: Optional display columns, written only for samples that are cell lines. A table
#: carries both or neither.
CELL_LINE_COLUMNS = ["cellosaurus_id", "cell_line_name"]

#: Additional Cell Model Passports columns, mapped to their output names. Each
#: was measured at >=99.8% coverage over the 1,248 screened cell lines.
CMP_EXTRA_COLUMNS = {
    "cancer_type": "cancer_type",
    "cancer_type_ncit_id": "cancer_type_ncit_id",
    "cancer_type_detail": "cancer_type_detail",
    "tissue_status": "tissue_status",
    "sample_site": "sample_site",
    "model_type": "model_type",
    "growth_properties": "growth_properties",
    "gender": "gender",
    "ethnicity": "ethnicity",
    "age_at_sampling": "age_at_sampling",
    "patient_id": "patient_id",
    "CCLE_ID": "ccle_id",
    "BROAD_ID": "broad_id",
    "COSMIC_ID": "cosmic_id",
    "msi_status": "msi_status",
    "mutational_burden": "mutational_burden",
}

#: Columns of the Cell Model Passports table, in output order.
CMP_META_COLUMNS = [
    SAMPLE_ID_COLUMN,
    *CELL_LINE_COLUMNS,
    *REQUIRED_COLUMNS,
    *CMP_EXTRA_COLUMNS.values(),
    *QUALITY_COLUMNS,
]


def sample_meta_path(dataset: str) -> UPath:
    """Return the metadata table that holds a dataset's samples.

    A dataset with a constant ``tissue`` or with ``sample_metadata: standard`` in
    ``config/sources.yaml`` has no Cell Model Passports entry and its own table;
    every other dataset reads the table shared by all of them.

    Args:
        dataset: Dataset name, e.g. ``"GDSC2"``.

    Returns:
        ``data/06_samples/metadata/<dataset>.parquet`` for a constant-tissue or
        standard dataset, otherwise ``data/06_samples/metadata/cmp.parquet``.
    """
    config = dataset_config(dataset)
    if config.get("tissue") or config.get("sample_metadata") == standard.STANDARD:
        return layout.sample_meta(dataset)
    return layout.cmp_sample_meta()


def read_sample_meta(path: UPath) -> pd.DataFrame:
    """Read a metadata table, indexed by its ``sample_id`` column.

    Args:
        path: A table written by :func:`build_cmp_meta` or :func:`build_constant_tissue_meta`.

    Returns:
        The table indexed by sample ID. It holds ``tissue`` and, only when the
        samples are cell lines, ``cellosaurus_id`` and ``cell_line_name``.

    Raises:
        CurationError: If the table lacks ``sample_id`` or a required column, or
            carries only one of the two cell-line columns.
    """
    frame = pd.read_parquet(str(path))
    missing = [c for c in [SAMPLE_ID_COLUMN, *REQUIRED_COLUMNS] if c not in frame.columns]
    if missing:
        msg = f"{path} lacks the column(s) {missing}; re-run the process_sample_meta stages"
        raise CurationError(msg)
    present = [c for c in CELL_LINE_COLUMNS if c in frame.columns]
    if present and len(present) != len(CELL_LINE_COLUMNS):
        msg = f"{path} carries {present} but needs both of {CELL_LINE_COLUMNS} or neither"
        raise CurationError(msg)
    return frame.drop_duplicates(subset=SAMPLE_ID_COLUMN, keep="first").set_index(SAMPLE_ID_COLUMN)


def _finalize_cmp(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows without an accession, deduplicate, attach flags, order columns."""
    frame = frame.dropna(subset=["cellosaurus_id"])
    frame = frame.drop_duplicates(subset="cellosaurus_id", keep="first").reset_index(drop=True)
    frame[SAMPLE_ID_COLUMN] = frame["cellosaurus_id"]

    for column in CMP_META_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    flags = quality_flags(pd.Index(frame["cellosaurus_id"]))
    for column in QUALITY_COLUMNS:
        frame[column] = flags[column].to_numpy()

    frame = frame[CMP_META_COLUMNS].copy()

    # Pin an explicit dtype contract so the Parquet round-trip is lossless:
    # "problematic" is a true boolean; every other column is a nullable string so
    # numeric CMP fields (cosmic_id, age_at_sampling, mutational_burden) do not
    # round-trip as floats and gain decimal points in mdata.obs.
    frame["problematic"] = frame["problematic"].astype("boolean").fillna(False).astype(bool)
    for column in CMP_META_COLUMNS:
        if column not in ("problematic", SAMPLE_ID_COLUMN, "cellosaurus_id"):
            frame[column] = frame[column].fillna("").astype(str)
    return frame


def build_cmp_meta(model_list_path: UPath) -> pd.DataFrame:
    """Build the metadata table from a Cell Model Passports model list.

    Only rows whose ``RRID`` is a Cellosaurus accession are usable, since every
    other identifier in the pipeline is keyed by ``CVCL`` accession.

    Args:
        model_list_path: Location of a CMP ``model_list.csv``.

    Returns:
        DataFrame with one row per accession, carrying :data:`CMP_META_COLUMNS`:
        the ``sample_id`` (the accession), the cell-line columns, ``tissue``, the
        :data:`CMP_EXTRA_COLUMNS` annotations and the Cellosaurus quality flags.
    """
    models = pd.read_csv(str(model_list_path), low_memory=False)
    accession = models["RRID"].astype("string")
    # A single RRID must hold one accession; see _sidm_mapping._valid_cvcl.
    is_cvcl = accession.str.startswith("CVCL").fillna(value=False) & ~accession.str.contains(";", na=False)

    frame = pd.DataFrame(
        {
            "cellosaurus_id": accession[is_cvcl],
            "cell_line_name": models.loc[is_cvcl, "model_name"],
            "tissue": models.loc[is_cvcl, "tissue"],
        }
    )
    for source, target in CMP_EXTRA_COLUMNS.items():
        if source in models.columns:
            frame[target] = models.loc[is_cvcl, source]

    return _finalize_cmp(frame)


def build_constant_tissue_meta(h5ad_paths: Iterable[UPath], tissue: str) -> pd.DataFrame:
    """Build the metadata table for a dataset whose tissue is a known constant.

    The sample index is the union of the ``obs`` indices of the given processed
    modalities, so the table covers exactly the sample IDs that MuData assembly
    will see. The samples are not cell lines, so the table has no Cellosaurus
    accession or cell line name.

    Args:
        h5ad_paths: Processed per-modality ``.h5ad`` files for one dataset.
        tissue: Tissue label to assign to every sample of the dataset.

    Returns:
        DataFrame with the columns ``sample_id`` and ``tissue``, one row per sample.
    """
    sample_ids: dict[str, None] = {}
    for path in h5ad_paths:
        for sample_id in ad.read_h5ad(str(path), backed="r").obs_names:
            sample_ids.setdefault(str(sample_id))

    return pd.DataFrame(
        {
            SAMPLE_ID_COLUMN: pd.Series(list(sample_ids), dtype="string"),
            "tissue": pd.Series([tissue] * len(sample_ids), dtype="string"),
        }
    )
