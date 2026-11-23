"""Download raw source files for an omics or response modality.

Delegates to ``core.download`` for the actual retrieval and unpacking; this
module owns the coordination with the dataset config. The caller is
responsible for supplying the output directory.

Files shared by several datasets are declared once per source in
``config/sources.yaml`` (``cell_model_passports``, ``ccle``, ``gdsc``) and fetched
by :func:`download_shared` rather than once per dataset.
"""

import logging
import shutil

from upath import UPath

from data_curation.core import layout
from data_curation.core.config import dataset_config, omics_source, shared_source
from data_curation.core.download import prune_unused, retrieve_file_list, retrieve_named_sources
from data_curation.core.errors import CurationError

logger = logging.getLogger(__name__)

#: Directory macOS zip tools add next to the real members; no stage reads it.
_MACOS_METADATA_DIR = "__MACOSX"


def download_omics_data(dataset: str, omics_key: str, output_dir: UPath) -> None:
    """Download raw files for one dataset/modality combination.

    The output directory must already exist; the caller is responsible for
    creating it and for choosing where in the filesystem it lives.

    Args:
        dataset: Dataset name, e.g. ``"PDX_Bruna"``.
        omics_key: Modality or section key in the dataset config, e.g.
            ``"transcriptomics"`` or ``"response"``.
        output_dir: Destination directory for the downloaded files.

    Raises:
        CurationError: If *dataset* takes the modality from a shared source,
            which :func:`download_shared` fetches once for all datasets.
    """
    source = None if omics_key == "response" else omics_source(dataset, omics_key)
    if source is not None:
        msg = f"The {omics_key} data of {dataset} comes from {source}; run `omics download-shared {source}` instead"
        raise CurationError(msg)
    ds_params = dataset_config(dataset, omics_key)
    retrieve_file_list(ds_params.get("files") or [], output_dir)
    retrieve_named_sources(ds_params, output_dir)
    prune_unused(output_dir, ds_params.get("keep") or [])


def download_shared(source: str, kinds: tuple[str, ...] | None = None) -> None:
    """Download every file of a shared source once into ``data/01_raw/<source>/<kind>/``.

    Each kind is fetched into its own directory, unpacked if it is an archive and
    pruned to the files its ``keep`` patterns name. Pooch verifies the pinned hash
    of each download.

    Args:
        source: Shared source block, e.g. ``"cell_model_passports"`` or ``"ccle"``.
        kinds: Entries of the block to fetch; every kind of the source when None.
    """
    for kind in kinds or layout.SHARED_SOURCES[source][1]:
        entry = shared_source(source, kind)
        output_dir = layout.source_raw(source, kind)
        output_dir.mkdir(parents=True, exist_ok=True)
        files = entry.get("files") or [{key: entry.get(key) for key in ("url", "fname", "known_hash")}]
        logger.info("Downloading %s %s (%d file(s))", source, kind, len(files))
        retrieve_file_list(files, output_dir)
        shutil.rmtree(str(output_dir / _MACOS_METADATA_DIR), ignore_errors=True)
        prune_unused(output_dir, entry.get("keep") or [])


def download_cmp(kinds: tuple[str, ...] = layout.CMP_KINDS) -> None:
    """Download every Cell Model Passports file once into ``data/01_raw/cmp/<kind>/``.

    Args:
        kinds: Entries of the ``cell_model_passports`` config block to fetch.
    """
    download_shared("cell_model_passports", kinds)
