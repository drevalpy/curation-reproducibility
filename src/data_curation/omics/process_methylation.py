"""Process raw DNA methylation data into a standardized AnnData file.

As in drevalpy v1, methylation comes from each screen's own omics release, and
both releases share one feature space, the CpG islands of the GDSC arrays:

- GDSC1/GDSC2 (``omics.methylation: gdsc``): Illumina 450K beta values averaged
  per CpG island; see ``methylation/gdsc.py``.
- CCLE, CTRPv1/v2 and the PRISM releases (``omics.methylation: ccle``): CCLE RRBS
  beta values of the CpG clusters around each transcription start site, each
  island taking the cluster that overlaps it most; see ``methylation/ccle.py``.
  This is why the CCLE stage also reads the GDSC methylation download.

Missing beta values are filled with 0.0 as in v1; ``adata.uns`` records the fill
value and the fraction of values it replaced.
"""

from data_curation.cell_lines.download_model_list import cmp_model_list_path
from data_curation.core import layout
from data_curation.core.config import omics_source
from data_curation.core.errors import CurationError
from data_curation.omics.methylation.ccle import process_ccle_methylation
from data_curation.omics.methylation.gdsc import process_gdsc_methylation


def main(dataset: str) -> None:
    """Process methylation for the given dataset into an AnnData .h5ad file.

    Raises:
        CurationError: If the dataset takes methylation from no supported source.
    """
    source = omics_source(dataset, "methylation")
    if source == "gdsc":
        adata = process_gdsc_methylation(
            layout.source_raw("gdsc", "methylation"),
            layout.source_raw("gdsc", "methylation_samples"),
            cmp_model_list_path(),
        )
    elif source == "ccle":
        adata = process_ccle_methylation(
            layout.source_raw("ccle", "methylation"),
            layout.source_raw("gdsc", "methylation"),
            cmp_model_list_path(),
        )
    else:
        msg = f"No methylation processor for dataset {dataset} (source: {source})"
        raise CurationError(msg)

    output_path = layout.omics("methylation", dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(str(output_path))
