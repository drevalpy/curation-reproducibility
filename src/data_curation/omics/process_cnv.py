"""Process CNV GISTIC data into a standardized AnnData file.

Parses the SNP6 GISTIC scores from Cell Model Passports into a cell line x gene
matrix of discrete copy number values. The file has genes as rows and models as
columns, under three header rows: the SIDM model IDs, the model names, and the
names of the two gene annotation columns (``gene_id``, ``symbol``).
"""

import anndata as ad
import numpy as np
import pandas as pd
from upath import UPath

from data_curation.cell_lines.download_model_list import cmp_model_list_path
from data_curation.cell_lines.sidm_mapping import build_id_to_cellosaurus_map
from data_curation.core import layout
from data_curation.core.paths import find_first
from data_curation.omics.samples import to_sample_anndata

#: Header rows below the SIDM row: the model names and the annotation column names.
_SKIPPED_HEADER_ROWS = [1, 2]

#: Leading annotation columns: the SIDG gene accession and the HGNC symbol. Genes are
#: named by symbol, the identifier the expression and mutation views share.
_ANNOTATION_COLUMNS = 2
_SYMBOL_POSITION = 1


def _process_cnv(raw_dir: UPath, model_list_path: UPath) -> ad.AnnData:
    """Read the CMP CNV GISTIC matrix as cell lines x genes, keyed by Cellosaurus and HGNC symbol."""
    path = find_first(raw_dir, ("*gistic*", "*cnv*.csv"))
    table = pd.read_csv(str(path), skiprows=_SKIPPED_HEADER_ROWS, low_memory=False)
    values = table.iloc[:, _ANNOTATION_COLUMNS:].astype(np.float32)
    values.index = pd.Index(table.iloc[:, _SYMBOL_POSITION])
    ids = build_id_to_cellosaurus_map(model_list_path, "model_id")
    return to_sample_anndata(values.T, ids, var_name="gene_symbol")


def main(dataset: str) -> None:
    """Process the shared CMP CNV download into the given dataset's AnnData .h5ad file.

    Every CMP dataset reads the same file, so *dataset* only names the output; the
    screened subset is taken at assembly.
    """
    output_path = layout.omics("cnv", dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _process_cnv(layout.cmp_raw("cnv"), cmp_model_list_path()).write_h5ad(str(output_path))
