"""Process CNV GISTIC data into a standardized AnnData file.

Parses the SNP6 GISTIC scores from Cell Model Passports into
a cell line x gene matrix of discrete copy number values.
"""

import anndata as ad
from upath import UPath

from data_curation.cell_lines.download_model_list import cmp_model_list_path
from data_curation.core import layout
from data_curation.core.paths import find_first
from data_curation.omics.cmp_matrix import read_cmp_matrix, to_cell_line_anndata

# model_id / model_name / (gene_id, symbol) precede the per-gene data rows.
CNV_HEADER_ROWS = 3

# Gene annotation column holding the HGNC symbol; the SIDG accession in the
# other annotation column cannot be aligned with the symbol-keyed expression
# and mutation views.
SYMBOL_COLUMN = "symbol"


def _find_cnv_file(raw_dir: UPath) -> UPath:
    """Locate the CNV GISTIC CSV in the raw directory."""
    return find_first(raw_dir, ("*gistic*", "*cnv*.csv"))


def _process_cnv(raw_dir: UPath, model_list_path: UPath) -> ad.AnnData:
    """Process CMP CNV GISTIC data into AnnData.

    Genes are rows and cell lines are columns in the raw file; the result is
    transposed so cell lines are obs and genes are var, keyed by HGNC symbol.
    """
    matrix = read_cmp_matrix(_find_cnv_file(raw_dir), CNV_HEADER_ROWS)
    return to_cell_line_anndata(
        values=matrix.values,
        gene_names=matrix.annotations[SYMBOL_COLUMN],
        model_list_path=model_list_path,
    )


def main(dataset: str) -> None:
    """Process the shared CMP CNV download into the given dataset's AnnData .h5ad file.

    Every CMP dataset reads the same file, so *dataset* only names the output; the
    screened subset is taken at assembly.
    """
    raw_dir = layout.cmp_raw("cnv")
    output_path = layout.omics("cnv", dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    adata = _process_cnv(raw_dir, cmp_model_list_path())
    adata.write_h5ad(str(output_path))
