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

from data_curation.core.errors import CurationError

SOURCES_PATH = UPath("config/sources.yaml")
PROCESSING_PATH = UPath("config/processing.yaml")

#: Value of a dataset's ``omics_source`` key when its omics come from the shared
#: Cell Model Passports downloads rather than from per-dataset ``<modality>`` blocks.
CMP_SOURCE = "cell_model_passports"

#: Modalities a CMP-sourced dataset provides, which have no per-dataset block.
CMP_MODALITIES = frozenset({"transcriptomics", "mutations", "cnv", "proteomics"})


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


def is_cmp_dataset(dataset: str) -> bool:
    """Whether a dataset's omics are the shared Cell Model Passports downloads.

    Args:
        dataset: Dataset name, e.g. ``"GDSC2"``.

    Returns:
        True when ``datasets.<dataset>.omics_source`` is ``cell_model_passports``.

    Raises:
        CurationError: If the dataset is not configured.
    """
    return dataset_config(dataset).get("omics_source") == CMP_SOURCE


def cmp_source(kind: str) -> dict[str, Any]:
    """Return one download entry of the ``cell_model_passports`` block.

    Args:
        kind: Entry key, e.g. ``"mutations"`` or ``"model_list"``.

    Returns:
        The entry (``url``, ``known_hash`` and optionally ``fname``, ``keep``).

    Raises:
        KeyError: If *kind* is not declared.
    """
    return source_config(CMP_SOURCE)[kind]


def dataset_names(modality: str | None = None) -> list[str]:
    """List the datasets configured for a modality, or every dataset.

    A dataset is valid for a modality when that modality key is present under
    its block in ``config/sources.yaml``, or when it is sourced from Cell Model
    Passports and the modality is one of its omics. Falls back to an empty list when the
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
        if modality in (block or {}) or (modality in CMP_MODALITIES and (block or {}).get("omics_source") == CMP_SOURCE)
    )
