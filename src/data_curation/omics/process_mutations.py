"""Process mutations data into a standardized binary AnnData file.

Filters for coding, non-silent mutations and creates a binary
(mutated/not-mutated) matrix per gene per cell line.
"""

import anndata as ad
import numpy as np
import pandas as pd
from upath import UPath

from data_curation.cell_lines.download_model_list import cmp_model_list_path
from data_curation.cell_lines.sidm_mapping import build_sidm_to_cellosaurus_map
from data_curation.core import layout
from data_curation.core.paths import find_first

_USED_COLUMNS = frozenset({"model_id", "gene_symbol", "coding", "effect"})


def _find_mutations_file(raw_dir: UPath) -> UPath:
    """Locate the mutations CSV in the raw directory."""
    return find_first(raw_dir, ("*mutations_all*", "*mutation*"))


def _process_mutations(raw_dir: UPath, model_list_path: UPath) -> ad.AnnData:
    """Process CMP mutations data into a binary AnnData."""
    mutations_file = _find_mutations_file(raw_dir)

    # The file is ~1.8 GB of repeated strings: read only the columns used below, as
    # categoricals, so the peak memory stays far below that of the full object table.
    df = pd.read_csv(str(mutations_file), usecols=lambda column: column in _USED_COLUMNS, dtype="category")

    # Filter: coding and non-silent
    if "coding" in df.columns:
        df = df[df["coding"] == "t"]
    if "effect" in df.columns:
        df = df[df["effect"] != "silent"]

    # Create binary presence: (model_id, gene_symbol) -> 1
    mut_pairs = df[["model_id", "gene_symbol"]].drop_duplicates().astype(object)
    mut_pairs["mutated"] = 1

    pivot = mut_pairs.pivot_table(index="model_id", columns="gene_symbol", values="mutated", fill_value=0)

    sidm_to_cvcl = build_sidm_to_cellosaurus_map(model_list_path)

    valid_rows = [idx for idx in pivot.index if idx in sidm_to_cvcl]
    pivot = pivot.loc[valid_rows]

    cellosaurus_ids = [sidm_to_cvcl[idx][0] for idx in valid_rows]
    cell_line_names = [sidm_to_cvcl[idx][1] for idx in valid_rows]

    adata = ad.AnnData(
        X=np.array(pivot.values, dtype=np.float32),
        obs=pd.DataFrame({"cell_line_name": cell_line_names}, index=pd.Index(cellosaurus_ids)),
        var=pd.DataFrame(index=pd.Index(pivot.columns, name="gene_symbol")),
    )
    adata.obs_names.name = layout.SAMPLE_INDEX_NAME
    adata = adata[~adata.obs_names.duplicated(keep="first")].copy()

    return adata


def main(dataset: str) -> None:
    """Process the shared CMP mutations download into the given dataset's AnnData .h5ad file."""
    raw_dir = layout.cmp_raw("mutations")
    output_path = layout.omics("mutations", dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    adata = _process_mutations(raw_dir, cmp_model_list_path())
    adata.write_h5ad(str(output_path))
