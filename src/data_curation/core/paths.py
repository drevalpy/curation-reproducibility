"""Locating raw input files by glob pattern.

Download stages land their sources under names carrying an upstream release date
(``rnaseq_merged_rsem_tpm_20260323.csv``), so process stages address them by
pattern rather than by literal name.

This lives in its own module because both the response and the omics stages need
it. It used to sit in the response preparers' shared module, which made every
transcriptomics stage declare a dependency on the response code and rebuild
whenever an unrelated response preparer changed.

Paths throughout the package are :class:`~upath.UPath`, so a stage can address a
remote filesystem without changing. Note that a remote ``UPath`` is *not* an
``os.PathLike``: ``os.fspath()`` on an ``S3Path`` raises, so pandas, anndata and
``gzip`` are handed ``str(path)`` (a URL they resolve through fsspec) or an open
file object rather than the path object itself.
"""

from collections.abc import Sequence

from upath import UPath


def find_file(raw_dir: UPath, pattern: str) -> UPath:
    """Locate a single file in a raw directory by glob pattern.

    Args:
        raw_dir: Directory to search.
        pattern: Glob pattern, e.g. ``"*rsem_tpm*"``.

    Returns:
        The first matching path, in sorted order so the choice is deterministic
        when several files match.

    Raises:
        FileNotFoundError: If nothing matches *pattern*.
    """
    candidates = sorted(raw_dir.glob(pattern))
    if not candidates:
        msg = f"No file matching '{pattern}' in {raw_dir}"
        raise FileNotFoundError(msg)
    return candidates[0]


def find_first(raw_dir: UPath, patterns: Sequence[str]) -> UPath:
    """Locate a file by trying glob patterns in order, most specific first.

    Args:
        raw_dir: Directory to search.
        patterns: Glob patterns; the first one with a match decides.

    Returns:
        The first match of the first matching pattern, in sorted order.

    Raises:
        FileNotFoundError: If no pattern matches.
    """
    for pattern in patterns:
        candidates = sorted(raw_dir.glob(pattern))
        if candidates:
            return candidates[0]
    present = sorted(path.name for path in raw_dir.iterdir()) if raw_dir.is_dir() else []
    msg = f"No file matching {list(patterns)} found in {raw_dir}. Files: {present}"
    raise FileNotFoundError(msg)
