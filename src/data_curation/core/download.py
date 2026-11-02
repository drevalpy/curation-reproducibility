"""Shared download utilities for the raw source files every domain fetches.

Two config styles are supported per dataset/omics entry:

- A generic ``files`` list of ``{url, fname, known_hash}`` entries, used by
  datasets that need several arbitrarily named source files.
- The legacy fixed keys (``url``, ``compounds_url``,
  ``fitted_url``) used by the Cell Model Passports based datasets.

Both may be combined; ``files`` is retrieved first.

Archives are deleted once extracted, and an entry may declare ``keep`` glob
patterns to prune the extracted files a stage never reads. Cell Model Passports
ships one zip holding every RNA-seq quantification, of which the pipeline reads
two: without pruning, each dataset retains a 1.2 GiB archive plus a 4.9 GiB
merged CSV that no stage globs for.
"""

import logging
from collections.abc import Sequence

import pooch
from upath import UPath

logger = logging.getLogger(__name__)

# Legacy fixed config keys: (url key, hash key, local filename).
# A local filename of None keeps whatever name the remote provides.
_NAMED_SOURCES: list[tuple[str, str, str | None]] = [
    ("url", "known_hash", None),
    ("compounds_url", "compounds_hash", "compounds.csv"),
    ("fitted_url", "fitted_hash", "fitted_dose_response.xlsx"),
]

# Files retrieved under a fixed name by _NAMED_SOURCES are always kept: they are
# addressed by name rather than by glob, so no `keep` pattern needs to list them.
_ALWAYS_KEEP = frozenset(fname for _, _, fname in _NAMED_SOURCES if fname)


def _retrieve(url: str, known_hash: str | None, output_dir: UPath, fname: str | None = None) -> list[str]:
    """Retrieve a single file, unzipping it in place when it is an archive.

    Args:
        url: Source URL.
        known_hash: Expected hash, or None to skip verification.
        output_dir: Destination directory.
        fname: Local filename, or None to keep the remote name.

    Returns:
        The paths pooch produced: the extracted members for an archive,
        otherwise the single retrieved file.
    """
    kwargs: dict = {"url": url, "known_hash": known_hash, "path": str(output_dir)}
    if fname:
        kwargs["fname"] = fname
    is_archive = url.endswith(".zip")
    if is_archive:
        kwargs["processor"] = pooch.Unzip(extract_dir="")

    result = pooch.retrieve(**kwargs)
    if isinstance(result, str):
        return [result]
    produced = [str(item) for item in result]
    if is_archive:
        _delete_archives(output_dir)
    return produced


def _delete_archives(output_dir: UPath) -> None:
    """Delete retained archives after their contents have been extracted."""
    for archive in output_dir.glob("*.zip"):
        size_mib = archive.stat().st_size / 1024**2
        archive.unlink()
        logger.info("Deleted extracted archive %s (%.0f MiB)", archive.name, size_mib)


def prune_unused(output_dir: UPath, keep: Sequence[str]) -> None:
    """Delete files in *output_dir* that no ``keep`` pattern matches.

    Extracted ``.zip`` archives are always removed, even when a ``keep`` pattern
    happens to match them: the GDSC archive is named
    ``GDSC2_public_raw_data_27Oct23.csv.zip``, so the very pattern that selects
    the extracted CSV also matches the redundant archive. Compressed data files
    read in place, such as ``proteomics_latest.csv.gz``, are not archives and are
    kept normally.

    Args:
        output_dir: Directory to prune.
        keep: Glob patterns for the files the process stages actually read.
    """
    if not keep:
        return

    kept: set[UPath] = set()
    for pattern in keep:
        kept.update(output_dir.glob(pattern))
    kept.update(output_dir / name for name in _ALWAYS_KEEP)
    kept.difference_update(output_dir.glob("*.zip"))

    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path in kept:
            continue
        size_mib = path.stat().st_size / 1024**2
        path.unlink()
        logger.info("Pruned unused source %s (%.0f MiB)", path.name, size_mib)


def retrieve_file_list(files: list[dict], output_dir: UPath) -> None:
    """Retrieve an explicit list of named files."""
    for entry in files:
        _retrieve(entry["url"], entry.get("known_hash"), output_dir, entry.get("fname"))


def retrieve_named_sources(ds_params: dict, output_dir: UPath) -> None:
    """Retrieve the fixed-key sources that are present in the config."""
    for url_key, hash_key, fname in _NAMED_SOURCES:
        url = ds_params.get(url_key)
        if url:
            _retrieve(url, ds_params.get(hash_key), output_dir, fname)


def download_file_list(files: list[dict], output_dir: UPath) -> None:
    """Retrieve an explicit list of named files into a directory.

    Args:
        files: Entries of ``{url, fname, known_hash}``.
        output_dir: Destination directory, created if missing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    retrieve_file_list(files, output_dir)
