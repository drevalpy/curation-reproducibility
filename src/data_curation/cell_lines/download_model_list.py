"""Location of the Cell Model Passports model list.

The model list is a single CSV that carries cell-line annotations (tissue,
cancer type, patient ID) and the ``RRID`` used to join to Cellosaurus
accessions. It is shared across all CMP-based omics and response modalities.
``omics.download.download_cmp`` fetches it with the other Cell Model Passports
files into ``data/01_raw/cmp/model_list/model_list.csv``.
"""

from upath import UPath

from data_curation.core import layout

#: Single shared path for the Cell Model Passports model list.
MODEL_LIST_PATH = layout.cmp_raw("model_list") / "model_list.csv"


def cmp_model_list_path() -> UPath:
    """Return the shared Cell Model Passports model list path.

    Returns:
        ``data/01_raw/cmp/model_list/model_list.csv``
    """
    return MODEL_LIST_PATH
