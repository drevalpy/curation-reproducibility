"""Materialise the global sample metadata tables.

Two kinds of table, each keyed by a ``sample_id`` column and carrying ``tissue``
among others. MuData assembly joins the one matching its dataset onto the union
sample index to populate ``mdata.obs``, which is where drevalpy expects sample
annotations to live (see ``meta.sample_meta_path``).

* ``main_cmp`` writes ``data/06_samples/metadata/cmp.parquet`` from the Cell
  Model Passports model list. Its samples are cell lines, so it also carries
  ``cellosaurus_id`` and ``cell_line_name``. It does not depend on the dataset, so
  it runs once for every dataset with CMP metadata (including PRISM).
* ``main`` writes ``data/06_samples/metadata/<dataset>.parquet`` for a dataset
  with an explicit ``datasets.<dataset>.tissue`` constant in
  ``config/sources.yaml``, the route for datasets without CMP metadata. Its
  samples are not cell lines, so the table holds only ``sample_id`` and ``tissue``.
"""

from upath import UPath

from data_curation.cell_lines.download_model_list import cmp_model_list_path
from data_curation.cell_lines.meta import build_cmp_meta, build_constant_tissue_meta
from data_curation.core import layout
from data_curation.core.config import dataset_config
from data_curation.core.errors import CurationError

# Modalities scanned to recover the sample index of a dataset that has no CMP
# metadata. These are exactly the modalities such datasets provide, and both are
# declared as stage dependencies.
CONSTANT_TISSUE_SOURCES = ("response", "transcriptomics")


def _constant_tissue_sources(dataset: str) -> list[UPath]:
    """Return the processed per-modality files that define the sample index."""
    paths = [layout.modality(key, dataset) for key in CONSTANT_TISSUE_SOURCES]
    existing = [path for path in paths if path.exists()]
    if not existing:
        msg = f"No processed modalities found for dataset {dataset}: looked for {[str(p) for p in paths]}"
        raise FileNotFoundError(msg)
    return existing


def main_cmp() -> None:
    """Write the sample metadata table shared by the Cell Model Passports datasets."""
    output_path = layout.cmp_sample_meta()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_cmp_meta(cmp_model_list_path()).to_parquet(str(output_path), index=False)


def main(dataset: str) -> None:
    """Write the sample metadata table of a dataset with a constant tissue.

    Raises:
        CurationError: If the dataset has no ``tissue`` constant; its metadata is the
            shared table written by :func:`main_cmp`.
    """
    tissue = dataset_config(dataset).get("tissue")
    if not tissue:
        msg = f"{dataset} has no constant tissue in config/sources.yaml; its metadata is the shared CMP table"
        raise CurationError(msg)

    output_path = layout.sample_meta(dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta = build_constant_tissue_meta(_constant_tissue_sources(dataset), str(tissue))
    meta.to_parquet(str(output_path), index=False)
