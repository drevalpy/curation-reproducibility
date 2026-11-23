"""Process the CCLE 2019 RRBS methylation into beta values over the GDSC CpG islands.

``CCLE_RRBS_TSS_CpG_clusters_20180614.txt`` gives, per transcription-start-site CpG cluster,
the mean beta value in each cell line (843 CCLE names; missing values are written
``    NaN``). The clusters are named by gene, not by position, so they are mapped onto the
GDSC islands, the feature space of drevalpy's ``methylation_intersection`` gene list,
the way drevalpy v1 did (``CCLE/methylation/preprocess_methylation.py``):

1. A cluster spans the first to the last CpG site of ``CpG_sites_hg19``. Clusters of at
   most 200 bp and clusters on chrX or chrY are dropped, as are exact duplicate rows.
2. Each cluster maps to the island on the same chromosome that overlaps it most
   (``min(ends) - max(starts)``), the first one in GDSC file order on a tie. Clusters
   that overlap no island are dropped.
3. Each island keeps one cluster: the largest overlap, then the largest ``avg_coverage``
   (missing counts as 0), then the first. Islands are ordered by their first appearance
   among the clusters.

v1 matched with a row-wise ``apply`` over all islands; here the same rule is evaluated
per chromosome with array operations.
"""

import logging
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from upath import UPath

from data_curation.cell_lines.cellosaurus import CELLOSAURUS_PATH
from data_curation.cell_lines.native_ids import resolve_ccle_names
from data_curation.core.errors import CurationError
from data_curation.core.paths import find_first
from data_curation.omics.methylation.gdsc import ISLAND_INDEX_NAME, finalize_beta, parse_islands, read_gdsc_islands
from data_curation.omics.samples import to_sample_anndata

logger = logging.getLogger(__name__)

_RRBS_PATTERNS = ("*RRBS_TSS_CpG_clusters*.txt", "*RRBS*.txt*")
_MISSING_TOKEN = "    NaN"  # noqa: S105
_METADATA_COLUMNS = ["cluster_id", "gene_name", "RefSeq_id", "CpG_sites_hg19", "avg_coverage"]

#: Clusters up to this length (last minus first site, in bp) are dropped.
MIN_CLUSTER_LENGTH = 200

#: Chromosomes dropped because the GDSC islands cover the autosomes only.
EXCLUDED_CHROMOSOMES = ("chrX", "chrY")

#: Rows of the cluster x island overlap matrix computed at once; bounds its memory.
_MATCH_CHUNK = 2000


def read_rrbs(path: UPath) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the RRBS table.

    Args:
        path: ``CCLE_RRBS_TSS_CpG_clusters_20180614.txt``.

    Returns:
        The cluster metadata (``cluster_id``, ``gene_name``, ``RefSeq_id``, ``CpG_sites_hg19``,
        ``avg_coverage``) and the float32 beta values (clusters x CCLE names), both with the
        file's row order and a shared integer index.

    Raises:
        CurationError: If a metadata column is missing.
    """
    header = pd.read_csv(str(path), sep="\t", nrows=0)
    missing = [column for column in _METADATA_COLUMNS if column not in header.columns]
    if missing:
        msg = f"{path} lacks the columns {missing}"
        raise CurationError(msg)
    sample_columns = [str(column) for column in header.columns if column not in _METADATA_COLUMNS]
    dtypes: dict[Any, Any] = dict.fromkeys(sample_columns, np.dtype("float32"))
    table = pd.read_csv(str(path), sep="\t", na_values=[_MISSING_TOKEN], dtype=dtypes)
    metadata = table[_METADATA_COLUMNS].copy()
    table.drop(columns=_METADATA_COLUMNS, inplace=True)
    return metadata, table


def cluster_ranges(sites: pd.Series) -> pd.DataFrame:
    """Derive the genomic range of each cluster from its ``chrom:pos;chrom:pos`` site list.

    The range runs from the position of the first site to the position of the last site
    in the list, on the chromosome of the first site.

    Args:
        sites: ``CpG_sites_hg19`` values, e.g. ``"1:66998970;1:66998973"``.

    Returns:
        Frame with ``chrom`` (``"chr1"``), ``start``, ``end`` and ``length`` (``end - start``).

    Raises:
        CurationError: If a site list is malformed.
    """
    first = sites.astype(str).str.extract(r"^\s*([^:;]+):(\d+)")
    last = sites.astype(str).str.extract(r":(\d+)\s*$")[0]
    malformed = first.isna().any(axis=1) | last.isna()
    if malformed.any():
        msg = f"Malformed CpG_sites_hg19 values (expected 'chrom:position;...'): {sites[malformed].head(3).tolist()}"
        raise CurationError(msg)
    ranges = pd.DataFrame({"chrom": "chr" + first[0], "start": first[1].astype("int64"), "end": last.astype("int64")})
    ranges["length"] = ranges["end"] - ranges["start"]
    return ranges


def _drop_duplicate_rows(metadata: pd.DataFrame, values: pd.DataFrame) -> np.ndarray:
    """Return a mask of the rows that are not exact repeats of an earlier row (metadata and values).

    Rows can only repeat when their metadata repeats, so the wide value columns are compared
    for those candidates only.
    """
    keep = np.ones(len(metadata), dtype=bool)
    candidates = metadata.duplicated(keep=False).to_numpy()
    if candidates.any():
        positions = np.flatnonzero(candidates)
        full = pd.concat([metadata.iloc[positions], values.iloc[positions]], axis=1)
        keep[positions[full.duplicated(keep="first").to_numpy()]] = False
    return keep


def match_best_island(clusters: pd.DataFrame, islands: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Find, for each cluster, the island on its chromosome with the largest overlap.

    Args:
        clusters: Frame with ``chrom``, ``start`` and ``end`` per cluster.
        islands: Frame with ``chrom``, ``start`` and ``end`` per island, in GDSC file order.

    Returns:
        ``(island_position, overlap)`` per cluster: the row position in *islands* of the best
        match (first one on ties) and its overlap, with ``-1`` and ``-1`` for clusters that
        overlap no island.
    """
    best = np.full(len(clusters), -1, dtype=np.int64)
    overlap = np.full(len(clusters), -1, dtype=np.int64)
    for chrom, island_rows in islands.groupby("chrom", sort=False).indices.items():
        cluster_rows = np.flatnonzero((clusters["chrom"] == chrom).to_numpy())
        if cluster_rows.size == 0:
            continue
        island_start = islands["start"].to_numpy()[island_rows]
        island_end = islands["end"].to_numpy()[island_rows]
        for chunk in np.array_split(cluster_rows, max(1, -(-cluster_rows.size // _MATCH_CHUNK))):
            start = clusters["start"].to_numpy()[chunk][:, None]
            end = clusters["end"].to_numpy()[chunk][:, None]
            touching = (end >= island_start) & (start <= island_end)
            score = np.where(touching, np.minimum(end, island_end) - np.maximum(start, island_start), -1)
            winner = score.argmax(axis=1)
            top = score[np.arange(len(chunk)), winner]
            matched = top >= 0
            best[chunk[matched]] = island_rows[winner[matched]]
            overlap[chunk[matched]] = top[matched]
    return best, overlap


def select_per_island(island: np.ndarray, overlap: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """Pick one cluster per island and order the picks by the island's first appearance.

    The winner is the cluster with the largest overlap, then the largest coverage, then the
    first in cluster order.

    Args:
        island: Matched island position per cluster.
        overlap: Overlap with that island.
        coverage: ``avg_coverage`` per cluster, with missing values already set to 0.

    Returns:
        Cluster positions of the winners, ordered by the first cluster that matched their island.
    """
    position = np.arange(len(island))
    order = np.lexsort((position, -coverage, -overlap, island))
    first_of_island = np.unique(island[order], return_index=True)[1]
    winners = order[first_of_island]
    first_seen = pd.Series(position).groupby(island).min().reindex(island[winners]).to_numpy()
    return winners[np.argsort(first_seen, kind="stable")]


def process_ccle_methylation(
    raw_dir: UPath,
    gdsc_raw_dir: UPath,
    model_list_path: UPath,
    cellosaurus_path: UPath = CELLOSAURUS_PATH,
) -> ad.AnnData:
    """Process the CCLE RRBS methylation into beta values per GDSC CpG island.

    Args:
        raw_dir: Directory holding ``CCLE_RRBS_TSS_CpG_clusters_20180614.txt``.
        gdsc_raw_dir: Directory holding the extracted GDSC ``F2_METH_CELL_DATA.txt``, whose
            islands are the feature space.
        model_list_path: Cell Model Passports ``model_list.csv``.
        cellosaurus_path: Parsed Cellosaurus table for the CCLE-name fallback.

    Returns:
        AnnData with Cellosaurus accessions as ``obs_names``, GDSC island ranges as ``var_names``
        and beta values (missing ones filled with 0.0) as ``X``.

    Raises:
        CurationError: If no cluster maps to an island or no cell line resolves.
    """
    metadata, values = read_rrbs(find_first(raw_dir, _RRBS_PATTERNS))
    islands = parse_islands(read_gdsc_islands(gdsc_raw_dir).index)
    ranges = cluster_ranges(metadata["CpG_sites_hg19"])

    keep = (ranges["length"] > MIN_CLUSTER_LENGTH) & ~ranges["chrom"].isin(EXCLUDED_CHROMOSOMES)
    keep &= _drop_duplicate_rows(metadata, values)
    kept = np.flatnonzero(keep.to_numpy())
    logger.info("RRBS clusters: %d, after length and chromosome filters and de-duplication: %d", len(keep), len(kept))

    best, overlap = match_best_island(ranges.iloc[kept].reset_index(drop=True), islands)
    matched = best >= 0
    logger.info("%d of %d clusters overlap a GDSC island", int(matched.sum()), len(kept))
    if not matched.any():
        msg = "No CCLE RRBS cluster overlaps a GDSC CpG island; check the genome build and chromosome names"
        raise CurationError(msg)

    kept, best, overlap = kept[matched], best[matched], overlap[matched]
    coverage = metadata["avg_coverage"].iloc[kept].fillna(0.0).to_numpy()
    winners = select_per_island(best, overlap, coverage)
    rows = kept[winners]
    logger.info("%d GDSC islands carry a CCLE cluster", len(rows))

    features = islands.index[best[winners]]
    frame = values.iloc[rows].T
    frame.columns = features

    ids = resolve_ccle_names(frame.index, model_list_path, cellosaurus_path)
    if not ids:
        msg = "No CCLE methylation cell line resolves to a Cellosaurus accession"
        raise CurationError(msg)
    adata = to_sample_anndata(frame, ids, var_name=ISLAND_INDEX_NAME)
    logger.info("CCLE methylation: %d cell lines x %d islands", adata.n_obs, adata.n_vars)
    return finalize_beta(adata)
