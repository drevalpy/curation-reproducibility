"""The on-disk layout of the pipeline's data directory - the single source of truth.

Every stage reads and writes through these helpers rather than spelling paths out,
so the directory tree can be understood (and changed) in one place::

    data/
      01_raw/          downloads, one subdirectory per source kind; the Cell Model
                       Passports files shared by several datasets sit in cmp/<kind>/
      02_omics/        processed omics matrices, one per modality
      03_drugs/        per-dataset drug mappings and the global drug ontology
      04_response/     wells -> fitted curves (+ replicate fits) -> harmonized response
      06_samples/      sample metadata: one table shared by the Cell Model
                       Passports datasets, one per remaining dataset
      07_assembled/    final MuData files
      08_reports/      rendered summary report
      cache/           untracked scratch, e.g. one Nextflow launch directory per fitted dataset

The numbers follow the order in which the stages run, so a directory only reads
from lower-numbered ones. ``05`` is unused: it held the per-replicate tables, which now
travel inside the curve files as ``uns["replicate_fits"]``.

This module is deliberately *not* listed as a dependency of any stage in
``dvc.yaml``. Every path here is also written out in ``dvc.yaml``, so a layout
change not mirrored there fails loudly - a stage writes somewhere DVC does not
expect - rather than silently. Listing it would make every stage, including the
~9 h ``fit_curves``, depend on one file. ``tests/cli/test_dvc_contract.py``
checks that the two stay in step.
"""

from upath import UPath

DATA = UPath("data")

RAW = DATA / "01_raw"
OMICS = DATA / "02_omics"
DRUGS = DATA / "03_drugs"
RESPONSE = DATA / "04_response"
SAMPLES = DATA / "06_samples"
ASSEMBLED = DATA / "07_assembled"
REPORTS = DATA / "08_reports"

#: Every numbered root, in pipeline order.
ROOTS = (RAW, OMICS, DRUGS, RESPONSE, SAMPLES, ASSEMBLED, REPORTS)

#: Untracked, git-ignored scratch that no stage declares as a dep or out.
CACHE = DATA / "cache"

# Shared raw inputs.
CELLOSAURUS = RAW / "mapping" / "cellosaurus.parquet"
CMP_RAW = RAW / "cmp"
RAW_DRUG_ONTOLOGY = RAW / "drug_ontology"

# Global outputs.
DRUG_MAPPING_DIR = DRUGS / "mapping"
DRUG_ONTOLOGY = DRUGS / "ontology.parquet"
SAMPLE_META_DIR = SAMPLES / "metadata"
REPORT_HTML = REPORTS / "data_summary.html"

#: Name of the ``obs`` index of every sample-keyed AnnData and of the assembled MuData
#: (``drevalpy`` reads the sample ID from it). It must be the same on every modality: mudata
#: refuses to join a named global index onto modalities with a different or no index name.
SAMPLE_INDEX_NAME = "sample_id"

#: Downloads of Cell Model Passports, each landing in ``CMP_RAW / <kind>``; the keys
#: of the ``cell_model_passports`` block in ``config/sources.yaml``.
CMP_KINDS = ("model_list", "rnaseq", "mutations", "cnv", "proteomics")

#: Assembled modality key -> directory holding that modality's per-dataset .h5ad.
MODALITY_DIRS = {
    "response": RESPONSE / "harmonized",
    "transcriptomics": OMICS / "transcriptomics",
    "mutations": OMICS / "mutations",
    "cnv": OMICS / "cnv",
    "proteomics": OMICS / "proteomics",
}


def raw_dir(kind: str, dataset: str) -> UPath:
    """Download directory for one dataset, e.g. ``raw_dir("response", "GDSC2")``."""
    return RAW / kind / dataset


def cmp_raw(kind: str) -> UPath:
    """Download directory of one Cell Model Passports file, e.g. ``cmp_raw("cnv")``.

    Raises:
        ValueError: If *kind* is not one of :data:`CMP_KINDS`.
    """
    if kind not in CMP_KINDS:
        msg = f"Unknown Cell Model Passports kind '{kind}'; expected one of {', '.join(CMP_KINDS)}"
        raise ValueError(msg)
    return CMP_RAW / kind


def modality(key: str, dataset: str) -> UPath:
    """Per-dataset ``.h5ad`` of a processed modality, keyed as in :data:`MODALITY_DIRS`."""
    return MODALITY_DIRS[key] / f"{dataset}.h5ad"


def omics(kind: str, dataset: str) -> UPath:
    """Processed omics matrix, e.g. ``omics("cnv", "GDSC1")``."""
    return OMICS / kind / f"{dataset}.h5ad"


def drug_mapping(dataset: str) -> UPath:
    """Per-dataset drug identifier mapping."""
    return DRUG_MAPPING_DIR / f"{dataset}.parquet"


def wells(dataset: str) -> UPath:
    """Well-level table fed to curve fitting, in ``drevalpy.curation.curate()``'s input columns."""
    return RESPONSE / "wells" / f"{dataset}.parquet"


def curves(dataset: str) -> UPath:
    """Curves fitted on the pooled replicates."""
    return RESPONSE / "curves" / f"{dataset}.h5ad"


def nextflow_launch_dir(dataset: str) -> UPath:
    """Directory ``fit_curves`` runs Nextflow from: its history, work directory and results.

    One per dataset, so that ``-resume`` picks up that dataset's previous run.
    """
    return CACHE / "nextflow" / dataset


def harmonized(dataset: str) -> UPath:
    """Harmonized response matrix (the assembled ``response`` modality)."""
    return modality("response", dataset)


def sample_meta(dataset: str) -> UPath:
    """Sample metadata table of a dataset outside Cell Model Passports."""
    return SAMPLE_META_DIR / f"{dataset}.parquet"


def cmp_sample_meta() -> UPath:
    """Sample metadata table shared by every Cell Model Passports dataset (all cell lines)."""
    return SAMPLE_META_DIR / "cmp.parquet"


def assembled(dataset: str) -> UPath:
    """Final assembled MuData file."""
    return ASSEMBLED / f"{dataset}.h5mu"
