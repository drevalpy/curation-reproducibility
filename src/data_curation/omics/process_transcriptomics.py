"""Process raw transcriptomics data into a standardized AnnData file.

Handles dataset-specific processing:
- CCLE/GDSC1/GDSC2/CTRPv1/CTRPv2: the shared Cell Model Passports RNA-seq download
  (``data/01_raw/cmp/rnaseq``), log2(TPM+1) as ``X``
  with the RSEM expected counts kept as a layer, SIDM ID mapping to Cellosaurus
  accessions via the model list.
- BeatAML2: log2-normalized RNA-seq keyed by patient subject.
- PDX_Bruna: log2 microarray intensities keyed by PDTC sample.

The value semantics differ per source, so each processor records its unit in
``adata.uns["expression_unit"]`` and, where layers exist, in
``adata.uns["layer_units"]``.
"""

from collections.abc import Callable

import anndata as ad
from upath import UPath

from data_curation.cell_lines.download_model_list import cmp_model_list_path
from data_curation.core import layout
from data_curation.core.config import is_cmp_dataset
from data_curation.core.errors import CurationError
from data_curation.core.paths import find_first
from data_curation.omics import COUNT_LAYER
from data_curation.omics.cmp_matrix import CmpMatrix, read_cmp_matrix, to_cell_line_anndata
from data_curation.omics.transcriptomics.beataml import process_beataml_transcriptomics
from data_curation.omics.transcriptomics.bruna import process_bruna_transcriptomics
from data_curation.omics.transcriptomics.prism import process_prism_transcriptomics

# model_id / model_name / data_source / (gene_symbol, ensembl_gene_id, gene_id)
# precede the per-gene data rows.
RNASEQ_HEADER_ROWS = 4

# Gene annotation column used as var_names, matching the symbol-keyed CNV and
# mutation views.
GENE_SYMBOL_COLUMN = "gene_symbol"


def _assert_matrices_aligned(tpm: CmpMatrix, counts: CmpMatrix) -> None:
    """Check that the TPM and count matrices describe the same genes and samples.

    Raises:
        CurationError: If gene order or sample columns differ between the two files.
    """
    if tpm.sample_ids != counts.sample_ids:
        msg = "TPM and expected-count matrices do not share the same sample columns"
        raise CurationError(msg)
    if not tpm.annotations.equals(counts.annotations):
        msg = "TPM and expected-count matrices do not share the same gene order"
        raise CurationError(msg)


def _process_cmp_rnaseq(raw_dir: UPath, model_list_path: UPath) -> ad.AnnData:
    """Process Cell Model Passports RNA-seq into AnnData.

    ``X`` holds the distributed log2(TPM+1) values and the RSEM expected counts
    are kept alongside as a layer. Genes with any missing value in either matrix
    are dropped: missingness is a Broad-vs-Sanger batch artifact that is absent
    or complete per source block, so imputing it would encode the source itself.
    Shared processor for the CMP datasets (CTRPv1, CTRPv2, GDSC1, GDSC2).
    """
    tpm = read_cmp_matrix(find_first(raw_dir, ("*rsem_tpm*", "*tpm*")), RNASEQ_HEADER_ROWS)
    counts = read_cmp_matrix(find_first(raw_dir, ("*expected_count*", "*read_count*")), RNASEQ_HEADER_ROWS)
    _assert_matrices_aligned(tpm, counts)

    complete = tpm.values.notna().all(axis=1) & counts.values.notna().all(axis=1)

    adata = to_cell_line_anndata(
        values=tpm.values.loc[complete],
        gene_names=tpm.annotations.loc[complete, GENE_SYMBOL_COLUMN],
        model_list_path=model_list_path,
        layers={COUNT_LAYER: counts.values.loc[complete]},
    )
    adata.uns["layer_units"] = {COUNT_LAYER: "rsem_expected_count"}
    return adata


#: Processors of the datasets with their own transcriptomics download. The Cell
#: Model Passports datasets (``omics_source: cell_model_passports`` in
#: ``config/sources.yaml``) all use ``_process_cmp_rnaseq``.
PROCESSORS: dict[str, Callable[..., ad.AnnData]] = {
    "BeatAML2": process_beataml_transcriptomics,
    "PDX_Bruna": process_bruna_transcriptomics,
    "PRISM_19Q4": process_prism_transcriptomics,
    "PRISM_20Q2": process_prism_transcriptomics,
}

# Value semantics of X per dataset, recorded in adata.uns.
EXPRESSION_UNITS: dict[str, str] = {
    "CCLE": "log2_tpm_plus_1",
    "CTRPv1": "log2_tpm_plus_1",
    "CTRPv2": "log2_tpm_plus_1",
    "GDSC1": "log2_tpm_plus_1",
    "GDSC2": "log2_tpm_plus_1",
    "BeatAML2": "log2_cpm",
    "PDX_Bruna": "log2_microarray",
    "PRISM_19Q4": "log2_tpm_plus_1",
    "PRISM_20Q2": "log2_tpm_plus_1",
}


def main(dataset: str) -> None:
    """Process transcriptomics for the given dataset into an AnnData .h5ad file."""
    output_path = layout.omics("transcriptomics", dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if dataset in PROCESSORS:
        adata = PROCESSORS[dataset](layout.raw_dir("transcriptomics", dataset))
    elif dataset in EXPRESSION_UNITS and is_cmp_dataset(dataset):
        adata = _process_cmp_rnaseq(layout.cmp_raw("rnaseq"), cmp_model_list_path())
    else:
        msg = f"No transcriptomics processor implemented for dataset: {dataset}"
        raise CurationError(msg)
    adata.uns["expression_unit"] = EXPRESSION_UNITS[dataset]
    adata.write_h5ad(str(output_path))
