"""Process proteomics data into a standardized AnnData file.

Reads DIA-MS protein intensity data from Cell Model Passports and creates a
cell line x protein matrix.

The ``var`` index is named ``gene_symbol`` rather than ``protein``: the source
aggregates protein groups to a single HGNC symbol per row, so the axis holds the
same identifier space as the transcriptomics, mutation and CNV modalities and is
named accordingly.
"""

import anndata as ad
import numpy as np
import pandas as pd
from upath import UPath

from data_curation.cell_lines.download_model_list import cmp_model_list_path
from data_curation.cell_lines.sidm_mapping import build_sidm_to_cellosaurus_map
from data_curation.core import layout
from data_curation.core.errors import CurationError
from data_curation.core.paths import find_first


def _find_proteomics_file(raw_dir: UPath) -> UPath:
    """Locate the proteomics CSV in the raw directory."""
    return find_first(raw_dir, ("*proteomics*",))


def _process_proteomics(raw_dir: UPath, model_list_path: UPath) -> ad.AnnData:
    """Process CMP proteomics data into AnnData.

    The file is a long-format CSV with model_id, symbol, and intensity columns.
    We pivot to a wide matrix of cell lines x proteins.
    """
    proteomics_file = _find_proteomics_file(raw_dir)

    df = pd.read_csv(str(proteomics_file), low_memory=False)

    # Identify the relevant columns (names vary between releases)
    model_col = "model_id" if "model_id" in df.columns else df.columns[0]
    symbol_col = next((c for c in df.columns if c in ("symbol", "gene_symbol", "gene_name", "protein")), None)
    intensity_col = next((c for c in df.columns if "intensity" in c.lower() or "lfq" in c.lower()), None)

    if symbol_col is None or intensity_col is None:
        msg = f"Cannot identify columns. Available: {list(df.columns)}"
        raise CurationError(msg)

    # Drop rows without a gene symbol or model_id
    df = df.dropna(subset=[model_col, symbol_col, intensity_col])

    # Pivot: rows = model_id, columns = gene/protein symbol, values = intensity
    # For duplicates (multiple proteins per gene), take the mean
    pivot = df.pivot_table(index=model_col, columns=symbol_col, values=intensity_col, aggfunc="mean")

    sidm_to_cvcl = build_sidm_to_cellosaurus_map(model_list_path)

    valid_rows = [idx for idx in pivot.index if str(idx) in sidm_to_cvcl]
    pivot = pivot.loc[valid_rows]

    cellosaurus_ids = [sidm_to_cvcl[str(idx)][0] for idx in valid_rows]
    cell_line_names = [sidm_to_cvcl[str(idx)][1] for idx in valid_rows]

    adata = ad.AnnData(
        X=np.array(pivot.values, dtype=np.float32),
        obs=pd.DataFrame({"cell_line_name": cell_line_names}, index=pd.Index(cellosaurus_ids)),
        var=pd.DataFrame(index=pd.Index(pivot.columns, name="gene_symbol")),
    )
    adata.obs_names.name = layout.SAMPLE_INDEX_NAME
    adata.var_names_make_unique()
    adata = adata[~adata.obs_names.duplicated(keep="first")].copy()

    return adata


def main(dataset: str) -> None:
    """Process the shared CMP proteomics download into the given dataset's AnnData .h5ad file."""
    raw_dir = layout.cmp_raw("proteomics")
    output_path = layout.omics("proteomics", dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    adata = _process_proteomics(raw_dir, cmp_model_list_path())
    adata.write_h5ad(str(output_path))
