"""Access to the pipeline's Python-owned configuration files.

``config/sources.yaml`` holds pinned download URLs, hashes and per-dataset
modality blocks. ``config/processing.yaml`` holds the toy subsample knobs and
the CurveCurator settings. Both are properties of the working tree rather than
of the installed package, so the paths are resolved relative to the current
working directory - DVC runs every stage from the repository root.

``params.yaml`` is deliberately not read here. That file only lists the dataset
names DVC expands ``foreach`` over; the Python package never needs those lists
because a dataset is valid for a modality when its modality block is present.
"""

from typing import Any

import yaml
from upath import UPath

from data_curation.core import layout
from data_curation.core.errors import CurationError

SOURCES_PATH = UPath("config/sources.yaml")
PROCESSING_PATH = UPath("config/processing.yaml")

#: Top-level blocks of ``config/sources.yaml`` holding omics downloads shared by
#: several datasets. A dataset names one of them per modality in its ``omics``
#: mapping instead of carrying a ``<modality>`` block of its own.
SHARED_SOURCES = tuple(layout.SHARED_SOURCES)


def _load(path: UPath) -> dict[str, Any]:
    """Read and parse a config YAML from the current working directory.

    Args:
        path: Config file to open.

    Returns:
        The parsed mapping.

    Raises:
        FileNotFoundError: If *path* is missing, which means the caller is not
            at the repository root.
    """
    if not path.is_file():
        msg = f"No {path} found; stages must run from the repository root"
        raise FileNotFoundError(msg)
    with path.open() as handle:
        return yaml.safe_load(handle)


def source_config(key: str) -> Any:
    """Return one top-level block from ``config/sources.yaml``.

    Args:
        key: Top-level key, e.g. ``"cellosaurus"`` or ``"drug_ontology"``.

    Returns:
        The value under *key*.

    Raises:
        KeyError: If *key* is not present.
        FileNotFoundError: If the sources file is missing.
    """
    return _load(SOURCES_PATH)[key]


def processing_config(key: str) -> Any:
    """Return one top-level block from ``config/processing.yaml``.

    Args:
        key: Top-level key, e.g. ``"toy"`` or ``"curvecurator"``.

    Returns:
        The value under *key*.

    Raises:
        KeyError: If *key* is not present.
        FileNotFoundError: If the processing file is missing.
    """
    return _load(PROCESSING_PATH)[key]


def dataset_config(dataset: str, section: str | None = None) -> dict[str, Any]:
    """Return one dataset's config block, optionally narrowed to a section.

    Args:
        dataset: Dataset name, e.g. ``"GDSC2"``.
        section: Modality key within the dataset block, e.g. ``"response"``.
            When omitted the whole dataset block is returned.

    Returns:
        The requested config block.

    Raises:
        CurationError: If the dataset, or the section within it, is not configured.
    """
    datasets = source_config("datasets")
    if dataset not in datasets:
        known = ", ".join(sorted(datasets))
        msg = f"Dataset '{dataset}' is not configured in {SOURCES_PATH}; known datasets: {known}"
        raise CurationError(msg)

    block = datasets[dataset]
    if section is None:
        return block
    if section not in block:
        msg = f"Dataset '{dataset}' has no '{section}' block in {SOURCES_PATH}"
        raise CurationError(msg)
    return block[section]


def omics_source(dataset: str, modality: str) -> str | None:
    """Return the shared source a dataset takes one omics modality from.

    Args:
        dataset: Dataset name, e.g. ``"GDSC2"``.
        modality: Omics modality, e.g. ``"methylation"``.

    Returns:
        The source named in ``datasets.<dataset>.omics.<modality>`` (one of
        :data:`SHARED_SOURCES`), or None when the dataset has its own
        ``<modality>`` block or no such modality.

    Raises:
        CurationError: If the dataset is not configured, or names an unknown source.
    """
    source = (dataset_config(dataset).get("omics") or {}).get(modality)
    if source is not None and source not in SHARED_SOURCES:
        msg = f"Dataset '{dataset}' takes {modality} from unknown source '{source}'; expected one of {SHARED_SOURCES}"
        raise CurationError(msg)
    return source


def shared_source(source: str, kind: str) -> dict[str, Any]:
    """Return one download entry of a shared source block, e.g. ``shared_source("ccle", "methylation")``.

    Args:
        source: One of :data:`SHARED_SOURCES`.
        kind: Entry key within the block.

    Returns:
        The entry: ``url``, ``known_hash`` and optionally ``fname``, or a ``files``
        list of those; optionally ``keep``.

    Raises:
        KeyError: If *source* or *kind* is not declared.
    """
    return source_config(source)[kind]


def dataset_names(modality: str | None = None) -> list[str]:
    """List the datasets configured for a modality, or every dataset.

    A dataset is valid for a modality when that modality key is present under
    its block in ``config/sources.yaml``, or when its ``omics`` mapping names a
    shared source for the modality. Falls back to an empty list when the
    file is unreadable, so that shell completion and ``--help`` still work
    outside the repository.

    Args:
        modality: Modality key such as ``"cnv"`` or ``"response"``, or None for
            every dataset under the ``datasets`` key.

    Returns:
        Dataset names, sorted.
    """
    try:
        datasets = source_config("datasets") or {}
    except (FileNotFoundError, OSError, yaml.YAMLError, KeyError):
        return []

    if modality is None:
        return sorted(datasets)
    return sorted(
        name
        for name, block in datasets.items()
        if modality in (block or {}) or modality in ((block or {}).get("omics") or {})
    )
