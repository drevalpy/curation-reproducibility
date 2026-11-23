"""Process raw transcriptomics data into a standardized AnnData file.

Handles dataset-specific processing:
- GDSC1/GDSC2 (``omics.transcriptomics: gdsc``): the GDSC1000 RMA-normalised
  microarray expression, keyed by COSMIC ID; see ``transcriptomics/gdsc.py``.
- CCLE/CTRPv1/CTRPv2 (``omics.transcriptomics: ccle``): the Broad's CCLE 2019
  RNA-seq, log2(TPM+1) as ``X`` with the read counts as a layer, keyed by CCLE
  name; see ``transcriptomics/ccle.py``.
- BeatAML2: log2-normalized RNA-seq keyed by patient subject.
- PDX_Bruna: log2 microarray intensities keyed by PDTC sample.
- Any dataset with ``transcriptomics.format: standard``: the long
  ``expression.parquet`` of the standard contract, processed like the CMP
  RNA-seq (log2(TPM+1) as ``X``, counts as a layer in the configured
  ``counts_unit``); see ``transcriptomics/standard.py``.

The value semantics differ per source, so each processor records its unit in
``adata.uns["expression_unit"]`` and, where layers exist, in
``adata.uns["layer_units"]``.
"""

from collections.abc import Callable

import anndata as ad

from data_curation.cell_lines.download_model_list import cmp_model_list_path
from data_curation.core import layout, standard
from data_curation.core.config import omics_source, source_config
from data_curation.core.errors import CurationError
from data_curation.omics.transcriptomics.beataml import process_beataml_transcriptomics
from data_curation.omics.transcriptomics.bruna import process_bruna_transcriptomics
from data_curation.omics.transcriptomics.ccle import process_ccle_transcriptomics
from data_curation.omics.transcriptomics.gdsc import process_gdsc_transcriptomics
from data_curation.omics.transcriptomics.prism import process_prism_transcriptomics
from data_curation.omics.transcriptomics.standard import process_standard_transcriptomics

#: Processors of the datasets with their own transcriptomics download. The
#: datasets that take it from a shared source (``omics.transcriptomics`` in
#: ``config/sources.yaml``) use that source's processor, in ``SHARED_PROCESSORS``.
PROCESSORS: dict[str, Callable[..., ad.AnnData]] = {
    "BeatAML2": process_beataml_transcriptomics,
    "PDX_Bruna": process_bruna_transcriptomics,
    "PRISM_19Q4": process_prism_transcriptomics,
    "PRISM_20Q2": process_prism_transcriptomics,
}

#: Processors of the shared transcriptomics sources, called with the source's raw
#: directory and the model list.
SHARED_PROCESSORS: dict[str, Callable[..., ad.AnnData]] = {
    "ccle": process_ccle_transcriptomics,
    "gdsc": process_gdsc_transcriptomics,
}

#: Unit of the counts layer of a standard dataset without ``counts_unit`` in its config.
DEFAULT_STANDARD_COUNTS_UNIT = "counts"

# Value semantics of X per dataset, recorded in adata.uns.
EXPRESSION_UNITS: dict[str, str] = {
    "CCLE": "log2_tpm_plus_1",
    "CTRPv1": "log2_tpm_plus_1",
    "CTRPv2": "log2_tpm_plus_1",
    "GDSC1": "log2_rma",
    "GDSC2": "log2_rma",
    "BeatAML2": "log2_cpm",
    "PDX_Bruna": "log2_microarray",
    "PRISM_19Q4": "log2_tpm_plus_1",
    "PRISM_20Q2": "log2_tpm_plus_1",
}


def _standard_config(dataset: str) -> dict | None:
    """Return the dataset's transcriptomics block if it selects the standard format, else ``None``."""
    block = source_config("datasets").get(dataset) or {}
    config = block.get("transcriptomics") or {}
    return config if config.get("format") == standard.STANDARD else None


def main(dataset: str) -> None:
    """Process transcriptomics for the given dataset into an AnnData .h5ad file."""
    output_path = layout.omics("transcriptomics", dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = _standard_config(dataset)
    if config is not None:
        counts_unit = str(config.get("counts_unit", DEFAULT_STANDARD_COUNTS_UNIT))
        adata = process_standard_transcriptomics(layout.raw_dir("transcriptomics", dataset), counts_unit)
        adata.write_h5ad(str(output_path))
        return
    source = omics_source(dataset, "transcriptomics")
    if source in SHARED_PROCESSORS and dataset in EXPRESSION_UNITS:
        adata = SHARED_PROCESSORS[source](layout.source_raw(source, "transcriptomics"), cmp_model_list_path())
    elif dataset in PROCESSORS:
        adata = PROCESSORS[dataset](layout.raw_dir("transcriptomics", dataset))
    else:
        msg = f"No transcriptomics processor implemented for dataset: {dataset}"
        raise CurationError(msg)
    adata.uns["expression_unit"] = EXPRESSION_UNITS[dataset]
    adata.write_h5ad(str(output_path))
