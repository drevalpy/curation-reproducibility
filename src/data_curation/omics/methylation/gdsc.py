"""Process the GDSC methylation (Iorio et al. 2016, Illumina 450K) into a standardized AnnData file.

Two raw downloads contribute:

- ``F2_METH_CELL_DATA.txt`` from the GDSC "Methylation" archive: beta values of 14,726
  CpG islands (rows, named by their GRCh37 range ``chr1:10003165-10003585``) over 1,029
  arrays (columns, named by Sentrix ID ``8221924127_R04C02``). The file is tab-separated
  with bare CR line endings.
- The GEO series matrix ``GSE68379_series_matrix.txt.gz``, the only public map from
  Sentrix ID to cell line: each sample carries its COSMIC ID among the
  ``!Sample_characteristics_ch1`` rows and its Sentrix ID in the name of its idat file.

Sentrix IDs resolve to COSMIC IDs through GEO and from there to Cellosaurus accessions
through the Cell Model Passports model list. A few cell lines were run on two arrays; the
first one in file order is kept (see :func:`data_curation.omics.samples.to_sample_anndata`).
The features keep the island ranges as names, which is what the CCLE processor and drevalpy's
``methylation_intersection`` gene list use.
"""

import csv
import gzip
import io
import logging
import re
from collections.abc import Iterator

import anndata as ad
import numpy as np
import pandas as pd
from upath import UPath

from data_curation.cell_lines.native_ids import resolve_cosmic_ids
from data_curation.core.errors import CurationError
from data_curation.core.paths import find_first
from data_curation.omics.samples import to_sample_anndata

logger = logging.getLogger(__name__)

#: Name of the ``var`` index: the GDSC CpG island range.
ISLAND_INDEX_NAME = "cpg_island"

#: Value semantics of ``X``, recorded in ``uns["methylation_unit"]``.
METHYLATION_UNIT = "beta"

#: Value that replaces missing betas, as in drevalpy v1's curation.
MISSING_FILL_VALUE = 0.0

#: ``uns["feature_space"]`` of both sources.
FEATURE_SPACE = "gdsc_cpg_islands"

_ISLAND_PATTERNS = ("F2_METH_CELL_DATA.txt", "*/F2_METH_CELL_DATA.txt")
_SERIES_MATRIX_PATTERNS = ("*series_matrix.txt.gz", "*series_matrix.txt")

_COSMIC = re.compile(r"^cosmic_id:\s*(\d+)\s*$")
_SENTRIX = re.compile(r"_(\d+_R\d\dC\d\d)_Grn\.idat(?:\.gz)?$")
_ISLAND = re.compile(r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$")


def read_gdsc_islands(raw_dir: UPath) -> pd.DataFrame:
    """Read the GDSC beta-value matrix of CpG islands.

    Args:
        raw_dir: Directory holding the extracted ``F2_METH_CELL_DATA.txt``.

    Returns:
        Islands x arrays beta values, the index the island ranges in file order and
        the columns the Sentrix IDs.

    Raises:
        FileNotFoundError: If the file is missing.
        CurationError: If an island name is not a ``chrN:start-end`` range or an array appears twice.
    """
    path = find_first(raw_dir, _ISLAND_PATTERNS)
    frame = pd.read_csv(str(path), sep="\t", index_col=0)
    frame.index = frame.index.astype(str).str.strip()
    frame.index.name = ISLAND_INDEX_NAME
    frame.columns = frame.columns.astype(str).str.strip()
    malformed = [name for name in frame.index if not _ISLAND.match(name)]
    if malformed:
        msg = f"{path} has island names that are not 'chrN:start-end' ranges: {malformed[:5]}"
        raise CurationError(msg)
    if frame.empty or frame.index.has_duplicates:
        msg = f"{path} has no islands or repeats an island"
        raise CurationError(msg)
    return frame


def parse_islands(names: pd.Index) -> pd.DataFrame:
    """Split island ranges into ``chrom``, ``start`` and ``end`` columns, in the order of *names*.

    Args:
        names: Island ranges such as ``chr1:10003165-10003585``.

    Returns:
        Frame indexed like *names* with the chromosome and integer coordinates.

    Raises:
        CurationError: If a name is not a range.
    """
    parts = names.to_series().astype(str).str.extract(_ISLAND)
    if parts.isna().any(axis=None):
        msg = f"Island names that are not 'chrN:start-end' ranges: {list(names[parts['chrom'].isna()][:5])}"
        raise CurationError(msg)
    return pd.DataFrame(
        {"chrom": parts["chrom"], "start": parts["start"].astype("int64"), "end": parts["end"].astype("int64")}
    )


def _open_text(path: UPath) -> Iterator[str]:
    """Yield the lines of a possibly gzipped text file."""
    with path.open("rb") as handle:
        text = (
            gzip.open(handle, "rt", encoding="utf-8")
            if path.suffix == ".gz"
            else io.TextIOWrapper(handle, encoding="utf-8")
        )
        yield from text


def _series_matrix_rows(path: UPath) -> dict[str, list[list[str]]]:
    """Collect the values of every ``!Sample_*`` header row of a series matrix, keyed by row name."""
    rows: dict[str, list[list[str]]] = {}
    lines = (line.rstrip("\r\n") for line in _open_text(path))
    for fields in csv.reader(lines, delimiter="\t"):
        if not fields:
            continue
        if fields[0].startswith("!series_matrix_table_begin"):
            break
        if fields[0].startswith("!Sample_"):
            rows.setdefault(fields[0], []).append(fields[1:])
    return rows


def read_sentrix_to_cosmic(samples_dir: UPath) -> dict[str, str]:
    """Map Sentrix array IDs to COSMIC IDs using the GEO series matrix of GSE68379.

    The ``cosmic_id:`` entry is looked up among all characteristics rows of each sample,
    because their positions differ between samples and some samples lack fields. The
    Sentrix ID is the part of the green-channel idat file name between the GEO accession
    and ``_Grn``.

    Args:
        samples_dir: Directory holding ``GSE68379_series_matrix.txt.gz``.

    Returns:
        ``{sentrix_id: cosmic_id}`` for the samples that state both.

    Raises:
        FileNotFoundError: If the series matrix is missing.
        CurationError: If the file has no sample rows.
    """
    path = find_first(samples_dir, _SERIES_MATRIX_PATTERNS)
    rows = _series_matrix_rows(path)
    characteristics = rows.get("!Sample_characteristics_ch1", [])
    files = rows.get("!Sample_supplementary_file", [])
    n_samples = max((len(row) for row in [*characteristics, *files]), default=0)
    if n_samples == 0:
        msg = f"{path} has no !Sample_characteristics_ch1 or !Sample_supplementary_file rows"
        raise CurationError(msg)

    mapping: dict[str, str] = {}
    for sample in range(n_samples):
        cosmic = _first_match(_COSMIC, (row[sample] for row in characteristics if sample < len(row)))
        sentrix = _first_match(_SENTRIX, (row[sample] for row in files if sample < len(row)))
        if cosmic and sentrix:
            mapping.setdefault(sentrix, cosmic)
    logger.info("GEO maps %d of %d samples to a Sentrix ID and COSMIC ID", len(mapping), n_samples)
    return mapping


def _first_match(pattern: re.Pattern[str], values: Iterator[str]) -> str | None:
    """Return the first capture group of the first value matching *pattern*, or ``None``."""
    for value in values:
        match = pattern.search(value.strip())
        if match:
            return match.group(1)
    return None


def finalize_beta(adata: ad.AnnData) -> ad.AnnData:
    """Record the missing fraction of a beta matrix, fill the gaps and set the methylation ``uns`` keys.

    Args:
        adata: Beta values, possibly with NaN.

    Returns:
        *adata*, modified in place: ``X`` filled with :data:`MISSING_FILL_VALUE`, and ``uns`` carrying
        ``methylation_unit``, ``missing_filled_with``, ``missing_fraction`` (of ``X`` before filling)
        and ``feature_space``.
    """
    matrix = np.asarray(adata.X, dtype=np.float32)
    missing = np.isnan(matrix)
    adata.uns["methylation_unit"] = METHYLATION_UNIT
    adata.uns["missing_filled_with"] = MISSING_FILL_VALUE
    adata.uns["missing_fraction"] = float(missing.mean()) if missing.size else 0.0
    adata.uns["feature_space"] = FEATURE_SPACE
    matrix[missing] = MISSING_FILL_VALUE
    adata.X = matrix
    return adata


def process_gdsc_methylation(raw_dir: UPath, samples_dir: UPath, model_list_path: UPath) -> ad.AnnData:
    """Process the GDSC 450K methylation into an AnnData of beta values per CpG island.

    Args:
        raw_dir: Directory holding the extracted ``F2_METH_CELL_DATA.txt``.
        samples_dir: Directory holding the GEO series matrix that maps arrays to COSMIC IDs.
        model_list_path: Cell Model Passports ``model_list.csv``.

    Returns:
        AnnData with Cellosaurus accessions as ``obs_names``, GDSC island ranges as ``var_names``
        and beta values (missing ones filled with 0.0) as ``X``.

    Raises:
        CurationError: If no array maps to a Cellosaurus accession.
    """
    islands = read_gdsc_islands(raw_dir)
    sentrix_to_cosmic = read_sentrix_to_cosmic(samples_dir)
    arrays = [array for array in islands.columns if array in sentrix_to_cosmic]
    logger.info("%d of %d arrays are on GEO", len(arrays), islands.shape[1])

    by_cosmic = resolve_cosmic_ids((sentrix_to_cosmic[array] for array in arrays), model_list_path)
    ids = {array: by_cosmic[sentrix_to_cosmic[array]] for array in arrays if sentrix_to_cosmic[array] in by_cosmic}
    if not ids:
        msg = "No GDSC methylation array maps to a Cellosaurus accession; check the GEO series matrix and model list"
        raise CurationError(msg)

    adata = to_sample_anndata(islands.T, ids, var_name=ISLAND_INDEX_NAME)
    logger.info("GDSC methylation: %d cell lines x %d islands", adata.n_obs, adata.n_vars)
    return finalize_beta(adata)
