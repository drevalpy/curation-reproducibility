"""Tests of the GDSC and CCLE methylation processors, built from small literal files."""

import gzip

import numpy as np
import pandas as pd
import pytest
from upath import UPath

from data_curation.core.errors import CurationError
from data_curation.omics.methylation import ccle, gdsc

MODELS = [
    {"model_id": "SIDM00001", "model_name": "A549", "RRID": "CVCL_0023", "COSMIC_ID": "905949", "CCLE_ID": "A549_LUNG"},
    {
        "model_id": "SIDM00002",
        "model_name": "MEC-1",
        "RRID": "CVCL_1870",
        "COSMIC_ID": "111;112",
        "CCLE_ID": "MEC1_BLOOD",
    },
    {"model_id": "SIDM00003", "model_name": "HeLa", "RRID": "CVCL_0030", "COSMIC_ID": "999", "CCLE_ID": "HELA_CERVIX"},
]
MODEL_COLUMNS = ["model_id", "model_name", "RRID", "tissue", "synonyms", "COSMIC_ID", "CCLE_ID"]

ISLANDS = ["chr1:100-500", "chr1:400-900", "chr2:100-500", "chr3:100-500"]


@pytest.fixture
def model_list(write_model_list) -> UPath:
    return write_model_list(MODELS, columns=MODEL_COLUMNS)


def _write_gdsc(raw_dir: UPath, arrays: dict[str, list[float | str]]) -> None:
    """Write F2_METH_CELL_DATA.txt with CR line endings: islands as rows, arrays as columns."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    lines = ["\t" + "\t".join(arrays)]
    for row, island in enumerate(ISLANDS):
        lines.append(island + "\t" + "\t".join(str(values[row]) for values in arrays.values()))
    (raw_dir / "F2_METH_CELL_DATA.txt").write_bytes("\r".join(lines).encode())


def _write_series_matrix(samples_dir: UPath, samples: list[tuple[str, str | None, bool]]) -> None:
    """Write a GEO series matrix; samples are ``(sentrix, cosmic, leading_field)``.

    ``leading_field`` adds an extra characteristics row for the sample only, shifting where its ``cosmic_id`` sits.
    """
    samples_dir.mkdir(parents=True, exist_ok=True)

    def row(name: str, values: list[str]) -> str:
        return name + "\t" + "\t".join(f'"{value}"' for value in values)

    count = len(samples)
    lines = [
        '!Series_title\t"Methylation"',
        row("!Sample_title", [f"s{i}" for i in range(count)]),
        row("!Sample_characteristics_ch1", [f"cell line: s{i}" for i in range(count)]),
        row("!Sample_characteristics_ch1", ["primary site: x" if lead else "" for _, _, lead in samples]),
        row("!Sample_characteristics_ch1", [f"cosmic_id: {cosmic}" if cosmic else "" for _, cosmic, _ in samples]),
        row(
            "!Sample_supplementary_file",
            [f"ftp://host/GSM{i}/suppl/GSM{i}_{sentrix}_Grn.idat.gz" for i, (sentrix, _, _) in enumerate(samples)],
        ),
        row(
            "!Sample_supplementary_file",
            [f"ftp://host/GSM{i}/suppl/GSM{i}_{sentrix}_Red.idat.gz" for i, (sentrix, _, _) in enumerate(samples)],
        ),
        "!series_matrix_table_begin",
        '"ID_REF"\t"GSM0"',
    ]
    with gzip.open(str(samples_dir / "GSE68379_series_matrix.txt.gz"), "wt") as handle:
        handle.write("\n".join(lines) + "\n")


class TestGdsc:
    def test_it_reads_cr_terminated_islands_in_file_order(self, tmp_dir) -> None:
        _write_gdsc(tmp_dir / "raw", {"1_R01C01": [0.1, 0.2, 0.3, 0.4], "2_R01C01": [0.5, 0.6, 0.7, 0.8]})

        islands = gdsc.read_gdsc_islands(tmp_dir / "raw")

        assert islands.shape == (4, 2)
        assert list(islands.index) == ISLANDS
        assert list(islands.columns) == ["1_R01C01", "2_R01C01"]
        assert islands.loc["chr2:100-500", "2_R01C01"] == pytest.approx(0.7)

    def test_it_maps_sentrix_ids_to_cosmic_ids_and_skips_samples_without_one(self, tmp_dir) -> None:
        _write_series_matrix(
            tmp_dir / "geo",
            [("11_R01C01", "905949", False), ("22_R02C01", None, True), ("33_R03C01", "111", True)],
        )

        mapping = gdsc.read_sentrix_to_cosmic(tmp_dir / "geo")

        assert mapping == {"11_R01C01": "905949", "33_R03C01": "111"}

    def test_it_requires_the_series_matrix(self, tmp_dir) -> None:
        (tmp_dir / "geo").mkdir()

        with pytest.raises(FileNotFoundError):
            gdsc.read_sentrix_to_cosmic(tmp_dir / "geo")

    def test_it_builds_an_anndata_keyed_by_cellosaurus_over_islands(self, tmp_dir, model_list) -> None:
        _write_gdsc(
            tmp_dir / "raw",
            {
                "11_R01C01": [0.1, 0.2, 0.3, 0.4],  # A549
                "22_R01C01": [0.5, 0.6, 0.7, 0.8],  # no cosmic id on GEO
                "33_R01C01": [0.9, 0.8, 0.7, 0.6],  # MEC-1
                "44_R01C01": [0.0, 0.0, 0.0, 0.0],  # second array of A549: dropped
                "55_R01C01": [0.3, 0.3, 0.3, 0.3],  # not on GEO
                "66_R01C01": [0.3, 0.3, 0.3, 0.3],  # cosmic id absent from the model list
            },
        )
        _write_series_matrix(
            tmp_dir / "geo",
            [
                ("11_R01C01", "905949", False),
                ("22_R01C01", None, True),
                ("33_R01C01", "112", True),
                ("44_R01C01", "905949", False),
                ("66_R01C01", "424242", False),
            ],
        )

        adata = gdsc.process_gdsc_methylation(tmp_dir / "raw", tmp_dir / "geo", model_list)

        assert list(adata.obs_names) == ["CVCL_0023", "CVCL_1870"]
        assert list(adata.obs["cell_line_name"]) == ["A549", "MEC-1"]
        assert list(adata.var_names) == ISLANDS
        assert adata.var_names.name == "cpg_island"
        assert np.asarray(adata.X).dtype == np.float32
        np.testing.assert_allclose(np.asarray(adata.X)[0], [0.1, 0.2, 0.3, 0.4])
        assert adata.uns["methylation_unit"] == "beta"
        assert adata.uns["feature_space"] == "gdsc_cpg_islands"
        assert adata.uns["missing_filled_with"] == 0.0
        assert adata.uns["missing_fraction"] == 0.0

    def test_it_fills_missing_values_and_records_their_fraction(self, tmp_dir, model_list) -> None:
        _write_gdsc(
            tmp_dir / "raw",
            {"11_R01C01": [0.1, "", 0.3, 0.4], "33_R01C01": [0.9, 0.8, "", ""]},
        )
        _write_series_matrix(tmp_dir / "geo", [("11_R01C01", "905949", False), ("33_R01C01", "111", False)])

        adata = gdsc.process_gdsc_methylation(tmp_dir / "raw", tmp_dir / "geo", model_list)

        assert adata.uns["missing_fraction"] == pytest.approx(3 / 8)
        np.testing.assert_allclose(np.asarray(adata.X), [[0.1, 0.0, 0.3, 0.4], [0.9, 0.8, 0.0, 0.0]])

    def test_it_fails_when_no_array_resolves(self, tmp_dir, model_list) -> None:
        _write_gdsc(tmp_dir / "raw", {"11_R01C01": [0.1, 0.2, 0.3, 0.4]})
        _write_series_matrix(tmp_dir / "geo", [("11_R01C01", "424242", False)])

        with pytest.raises(CurationError, match="Cellosaurus"):
            gdsc.process_gdsc_methylation(tmp_dir / "raw", tmp_dir / "geo", model_list)


class TestClusterRanges:
    def test_it_spans_the_first_to_the_last_site(self) -> None:
        sites = pd.Series(["1:5000;1:5100;1:5300", "X:10;X:20", "7:900"])

        ranges = ccle.cluster_ranges(sites)

        assert list(ranges["chrom"]) == ["chr1", "chrX", "chr7"]
        assert list(ranges["start"]) == [5000, 10, 900]
        assert list(ranges["end"]) == [5300, 20, 900]
        assert list(ranges["length"]) == [300, 10, 0]

    def test_it_takes_the_first_and_last_site_in_listed_order_not_the_min_and_max(self) -> None:
        ranges = ccle.cluster_ranges(pd.Series(["2:800;2:700;2:900;2:750"]))

        assert (ranges["start"][0], ranges["end"][0]) == (800, 750)

    def test_it_rejects_malformed_site_lists(self) -> None:
        with pytest.raises(CurationError, match="Malformed"):
            ccle.cluster_ranges(pd.Series(["1:100;oops"]))


class TestMatchBestIsland:
    @staticmethod
    def _frame(rows: list[tuple[str, int, int]]) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=pd.Index(["chrom", "start", "end"]))

    def test_it_picks_the_island_with_the_largest_overlap(self) -> None:
        islands = self._frame([("chr1", 100, 500), ("chr1", 400, 900)])
        clusters = self._frame([("chr1", 450, 880), ("chr1", 120, 480)])

        best, overlap = ccle.match_best_island(clusters, islands)

        assert list(best) == [1, 0]
        assert list(overlap) == [430, 360]

    def test_it_takes_the_first_island_in_file_order_on_ties(self) -> None:
        islands = self._frame([("chr1", 100, 500), ("chr1", 100, 500)])

        best, overlap = ccle.match_best_island(self._frame([("chr1", 200, 300)]), islands)

        assert list(best) == [0]
        assert list(overlap) == [100]

    def test_it_requires_the_same_chromosome_and_touching_ranges(self) -> None:
        islands = self._frame([("chr1", 100, 500), ("chr2", 100, 500)])
        clusters = self._frame([("chr3", 100, 500), ("chr1", 501, 900), ("chr2", 500, 900)])

        best, overlap = ccle.match_best_island(clusters, islands)

        assert list(best) == [-1, -1, 1]
        assert list(overlap) == [-1, -1, 0]

    def test_it_matches_islands_across_chunks(self, monkeypatch) -> None:
        monkeypatch.setattr(ccle, "_MATCH_CHUNK", 2)
        islands = self._frame([("chr1", 0, 100), ("chr1", 200, 300)])
        clusters = self._frame([("chr1", 210, 290)] * 3 + [("chr1", 10, 90)] * 2)

        best, _ = ccle.match_best_island(clusters, islands)

        assert list(best) == [1, 1, 1, 0, 0]


class TestSelectPerIsland:
    def test_it_prefers_overlap_then_coverage_then_the_first_cluster(self) -> None:
        island = np.array([0, 0, 1, 1, 1, 2, 2])
        overlap = np.array([10, 20, 5, 5, 5, 7, 7])
        coverage = np.array([99.0, 1.0, 1.0, 3.0, 3.0, 2.0, 2.0])

        winners = ccle.select_per_island(island, overlap, coverage)

        assert list(winners) == [1, 3, 5]

    def test_it_orders_islands_by_their_first_appearing_cluster(self) -> None:
        island = np.array([2, 0, 2, 1])
        overlap = np.array([1, 1, 9, 1])
        coverage = np.zeros(4)

        winners = ccle.select_per_island(island, overlap, coverage)

        assert list(winners) == [2, 1, 3]


def _write_rrbs(raw_dir: UPath, clusters: list[tuple], samples: list[str]) -> None:
    """Write an RRBS table; clusters are ``(id, sites, avg_coverage, values)`` with ``None`` for missing."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(["cluster_id", "gene_name", "RefSeq_id", "CpG_sites_hg19", "avg_coverage", *samples])]
    for cluster_id, sites, coverage, values in clusters:
        cells = ["    NaN" if value is None else f"{value:.5f}" for value in values]
        lines.append("\t".join([cluster_id, "GENE", "NM_1", sites, f"{coverage:7.2f}", *cells]))
    (raw_dir / "CCLE_RRBS_TSS_CpG_clusters_20180614.txt").write_text("\n".join(lines) + "\n")


class TestCcle:
    SAMPLES = ["A549_LUNG", "MEC1_BLOOD", "UNKNOWN_SKIN"]

    @pytest.fixture
    def gdsc_dir(self, tmp_dir) -> UPath:
        _write_gdsc(tmp_dir / "gdsc", {"11_R01C01": [0.1, 0.2, 0.3, 0.4]})
        return tmp_dir / "gdsc"

    def _run(self, tmp_dir, gdsc_dir, model_list, clusters):
        _write_rrbs(tmp_dir / "rrbs", clusters, self.SAMPLES)
        cellosaurus = tmp_dir / "cellosaurus.parquet"
        pd.DataFrame(
            {
                "name": ["Other"],
                "cellosaurus_id": ["CVCL_9999"],
                "synonyms": [""],
                "problem_description": [""],
                "discontinued": [""],
                "misspelling": [""],
                "problematic": [False],
            }
        ).to_parquet(str(cellosaurus), index=False)
        return ccle.process_ccle_methylation(tmp_dir / "rrbs", gdsc_dir, model_list, cellosaurus)

    def test_it_maps_clusters_onto_gdsc_islands_with_v1_semantics(self, tmp_dir, gdsc_dir, model_list) -> None:
        clusters = [
            # first and last site give chr1:150-480 (best overlap with island 0: 330 vs island 1: 80)
            ("c_a", "1:150;1:300;1:480", 10.0, [0.1, 0.2, None]),
            # same island 0 with a smaller overlap: loses despite far higher coverage
            ("c_b", "1:200;1:420", 500.0, [0.9, 0.9, 0.9]),
            # island 1 (chr1:400-900) through a tie-free overlap of 480
            ("c_c", "1:410;1:890", 5.0, [0.3, None, 0.5]),
            # too short (length 200 is not > 200)
            ("c_d", "2:100;2:300", 5.0, [0.7, 0.7, 0.7]),
            # chrX is excluded
            ("c_e", "X:100;X:900", 5.0, [0.7, 0.7, 0.7]),
            # island 2 (chr2:100-500)
            ("c_f", "2:90;2:600", 5.0, [0.4, 0.5, 0.6]),
            # chromosome without an island
            ("c_g", "9:100;9:900", 5.0, [0.7, 0.7, 0.7]),
        ]

        adata = self._run(tmp_dir, gdsc_dir, model_list, clusters)

        assert list(adata.var_names) == ["chr1:100-500", "chr1:400-900", "chr2:100-500"]
        assert list(adata.obs_names) == ["CVCL_0023", "CVCL_1870"]
        np.testing.assert_allclose(np.asarray(adata.X)[0], [0.1, 0.3, 0.4], rtol=1e-6)
        np.testing.assert_allclose(adata.X[1], [0.2, 0.0, 0.5], rtol=1e-6)
        assert np.asarray(adata.X).dtype == np.float32

    def test_it_records_the_missing_fraction_before_filling(self, tmp_dir, gdsc_dir, model_list) -> None:
        clusters = [
            ("c_a", "1:150;1:480", 10.0, [0.1, None, 0.2]),
            ("c_b", "2:90;2:600", 5.0, [None, None, 0.6]),
        ]

        adata = self._run(tmp_dir, gdsc_dir, model_list, clusters)

        # UNKNOWN_SKIN is dropped as unresolved, so 3 of the 4 resolved values are missing.
        assert adata.uns["missing_fraction"] == pytest.approx(3 / 4)
        assert adata.uns["missing_filled_with"] == 0.0
        assert adata.uns["methylation_unit"] == "beta"
        assert adata.uns["feature_space"] == "gdsc_cpg_islands"
        assert not np.isnan(adata.X).any()

    def test_it_breaks_island_ties_by_coverage_then_by_position(self, tmp_dir, gdsc_dir, model_list) -> None:
        clusters = [
            ("c_a", "1:150;1:480", 5.0, [0.1, 0.1, 0.1]),
            ("c_b", "1:150;1:480", 9.0, [0.2, 0.2, 0.2]),
            ("c_c", "1:150;1:480", 9.0, [0.3, 0.3, 0.3]),
        ]

        adata = self._run(tmp_dir, gdsc_dir, model_list, clusters)

        assert list(adata.var_names) == ["chr1:100-500"]
        np.testing.assert_allclose(adata.X[:, 0], [0.2, 0.2], rtol=1e-6)

    def test_it_drops_exact_duplicate_rows(self) -> None:
        metadata = pd.DataFrame({"cluster_id": ["a", "a", "a"], "avg_coverage": [1.0, 1.0, 1.0]})
        values = pd.DataFrame({"s": [0.5, 0.5, 0.6], "t": [np.nan, np.nan, np.nan]})

        keep = ccle._drop_duplicate_rows(metadata, values)

        assert list(keep) == [True, False, True]

    def test_it_fails_when_no_cluster_overlaps_an_island(self, tmp_dir, gdsc_dir, model_list) -> None:
        with pytest.raises(CurationError, match="overlaps"):
            self._run(tmp_dir, gdsc_dir, model_list, [("c_a", "9:100;9:900", 5.0, [0.1, 0.2, 0.3])])
