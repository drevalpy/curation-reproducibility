"""Process PDX_Bruna microarray expression into a standardized AnnData file.

The published matrix (``ExpressionSamples.txt``) is genes x samples of
log2 microarray intensities and covers both PDTXs and PDTCs, using sample names
that differ from the drug-screen sample names (e.g. ``HCI001-X4`` in expression
vs ``HCI001-X4C`` in response).

The correspondence is not derivable by rule (some passages/regrowths only match
approximately, e.g. ``HCI002-X3R`` -> ``HCI002-X3C``), so it is maintained as a
reviewable curated CSV alongside this module. It covers 25 of the 37 screened
PDTCs; the remaining 12 have no microarray profile.
"""

import logging

import anndata as ad
import numpy as np
import pandas as pd
from upath import UPath

from data_curation.core import layout
from data_curation.core.paths import find_file

logger = logging.getLogger(__name__)

# Packaged alongside the omics code rather than under data/, which holds only
# DVC-tracked artifacts. Anchored on the omics package directory (this module's
# grandparent) so it keeps resolving if this file moves within the subpackage.
_OMICS_ROOT = UPath(__file__).parent.parent
_SAMPLE_MAP_PATH = _OMICS_ROOT / "data" / "PDX_Bruna_expression_sample_map.csv"


def _load_sample_map() -> dict[str, str]:
    """Load the curated expression-sample -> response-sample mapping."""
    df = pd.read_csv(str(_SAMPLE_MAP_PATH))
    return dict(zip(df["expression_sample"], df["response_sample"], strict=True))


def process_bruna_transcriptomics(raw_dir: UPath) -> ad.AnnData:
    """Process PDX_Bruna microarray expression into AnnData keyed by PDTC sample."""
    expression_file = find_file(raw_dir, "ExpressionSamples*")
    logger.info("Reading PDX_Bruna expression from %s", expression_file)

    df = pd.read_csv(str(expression_file), sep="\t", index_col=0, low_memory=False)

    sample_map = _load_sample_map()
    valid_cols = [col for col in df.columns if col in sample_map]
    logger.info("Matched %d/%d expression samples to screened PDTCs", len(valid_cols), len(sample_map))
    df = df[valid_cols]

    response_ids = [sample_map[col] for col in valid_cols]

    # Transpose: samples become obs, genes become var
    adata = ad.AnnData(
        X=df.values.astype(np.float32).T,
        obs=pd.DataFrame(index=pd.Index(response_ids)),
        var=pd.DataFrame(index=pd.Index(df.index, name="gene_symbol")),
    )
    adata.obs_names.name = layout.SAMPLE_INDEX_NAME
    adata.var_names_make_unique()
    adata = adata[~adata.obs_names.duplicated(keep="first")].copy()

    return adata
