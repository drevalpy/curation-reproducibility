"""Download raw source files for an omics or response modality.

Delegates to ``core.download`` for the actual retrieval and unpacking; this
module owns the coordination with the dataset config. The caller is
responsible for supplying the output directory.

Files shared by every Cell Model Passports dataset are declared once under
``cell_model_passports`` in ``config/sources.yaml`` and fetched by
:func:`download_cmp` rather than once per dataset.
"""

import logging

from upath import UPath

from data_curation.core import layout
from data_curation.core.config import cmp_source, dataset_config, is_cmp_dataset
from data_curation.core.download import prune_unused, retrieve_file_list, retrieve_named_sources
from data_curation.core.errors import CurationError

logger = logging.getLogger(__name__)


def download_omics_data(dataset: str, omics_key: str, output_dir: UPath) -> None:
    """Download raw files for one dataset/modality combination.

    The output directory must already exist; the caller is responsible for
    creating it and for choosing where in the filesystem it lives.

    Args:
        dataset: Dataset name, e.g. ``"GDSC2"``.
        omics_key: Modality or section key in the dataset config, e.g.
            ``"transcriptomics"``, ``"cnv"`` or ``"response"``.
        output_dir: Destination directory for the downloaded files.

    Raises:
        CurationError: If the omics of *dataset* come from Cell Model Passports,
            which :func:`download_cmp` fetches once for all datasets.
    """
    if omics_key != "response" and is_cmp_dataset(dataset):
        msg = f"The {omics_key} data of {dataset} comes from Cell Model Passports; run `omics download-cmp` instead"
        raise CurationError(msg)
    ds_params = dataset_config(dataset, omics_key)
    retrieve_file_list(ds_params.get("files") or [], output_dir)
    retrieve_named_sources(ds_params, output_dir)
    prune_unused(output_dir, ds_params.get("keep") or [])


def download_cmp(kinds: tuple[str, ...] = layout.CMP_KINDS) -> None:
    """Download every Cell Model Passports file once into ``data/01_raw/cmp/<kind>/``.

    Each kind is fetched into its own directory, unpacked if it is an archive and
    pruned to the files its ``keep`` patterns name. Pooch verifies the pinned hash
    of each download.

    Args:
        kinds: Entries of the ``cell_model_passports`` config block to fetch.
    """
    for kind in kinds:
        entry = cmp_source(kind)
        output_dir = layout.cmp_raw(kind)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading Cell Model Passports %s from %s", kind, entry["url"])
        retrieve_file_list([{key: entry.get(key) for key in ("url", "fname", "known_hash")}], output_dir)
        prune_unused(output_dir, entry.get("keep") or [])
