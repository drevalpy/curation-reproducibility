"""Tests for the report's backed-mode reading and summary tables."""

from pathlib import Path

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
from drevalpy.curation._replicate_fits import ID_COLUMNS, METRICS, to_schema
from drevalpy.data._replicate_fits import KEY
from upath import UPath

from data_curation.report import summary
from data_curation.report.reading import (
    count_finite,
    discover_datasets,
    finite_counts,
    read_snapshot,
    read_snapshots,
    replicate_counts,
)

md.set_options(pull_on_update=False)

NAN = np.nan


def _anndata(obs: list[str], x: np.ndarray, prefix: str = "f") -> ad.AnnData:
    return ad.AnnData(
        X=np.asarray(x, dtype=np.float32),
        obs=pd.DataFrame(index=pd.Index(obs)),
        var=pd.DataFrame(index=pd.Index([f"{prefix}{i}" for i in range(x.shape[1])])),
    )


def _fits(rows: list[tuple[str, str, str, int]]) -> pd.DataFrame:
    """A replicate-fits table from (sample, drug, status, biological_replicate) rows."""
    table = pd.DataFrame({column: pd.Series(["x"] * len(rows), dtype=object) for column in ID_COLUMNS})
    table["sample_id"] = [row[0] for row in rows]
    table["pubchem_id"] = [row[1] for row in rows]
    table["status"] = [row[2] for row in rows]
    table["biological_replicate"] = [row[3] for row in rows]
    table["technical_replicate"] = table.groupby(["sample_id", "pubchem_id", "biological_replicate"]).cumcount()
    table["replicate"] = range(len(rows))
    table["n_doses"] = 8
    for metric in METRICS:
        table[metric] = 1.0
    return to_schema(table)


def _with_fits(adata: ad.AnnData, rows: list[tuple[str, str, str, int]]) -> ad.AnnData:
    adata.uns[KEY] = _fits(rows)
    return adata


def _write(path: Path, modalities: dict[str, ad.AnnData], tissues: dict[str, str]) -> UPath:
    mdata = md.MuData(modalities)
    mdata.obs["tissue"] = pd.Series(tissues).reindex(mdata.obs_names).astype("object")
    mdata.write(str(path))
    return UPath(path)


@pytest.fixture
def two_datasets(tmp_path: Path) -> list[UPath]:
    """Dataset A: 3 samples (one without any response), expression + mutations.

    A's replicate fits: (c1, d0) has two fitted series, (c2, d1) only one fitted and one failed.

    Dataset B: 2 samples, one shared with A, expression only, single-series fits only.
    """
    a_fits = [
        ("c1", "d0", "fitted", 0),
        ("c1", "d0", "fitted", 1),
        ("c2", "d1", "fitted", 0),
        ("c2", "d1", "fit_failed", 0),
    ]
    a = _write(
        tmp_path / "A.h5mu",
        {
            "response": _with_fits(_anndata(["c1", "c2", "c3"], np.array([[1, 2], [NAN, 3], [NAN, NAN]]), "d"), a_fits),
            "gene_expression": _anndata(["c1", "c2", "c3"], np.ones((3, 4))),
            "mutations": _anndata(["c1"], np.ones((1, 5))),
        },
        {"c1": "Lung", "c2": "Skin", "c3": "Lung"},
    )
    b = _write(
        tmp_path / "B.h5mu",
        {
            "response": _with_fits(
                _anndata(["c2", "c4"], np.array([[1.0], [2.0]]), "d"),
                [("c2", "d0", "fitted", 0), ("c4", "d0", "fitted", 0)],
            ),
            "gene_expression": _anndata(["c4"], np.ones((1, 4))),
        },
        {"c2": "Skin", "c4": ""},
    )
    return [a, b]


class TestCountFinite:
    @pytest.mark.parametrize("chunk_rows", [1, 2, 3, 100])
    def test_result_does_not_depend_on_chunk_size(self, chunk_rows: int) -> None:
        matrix = np.array([[1, NAN], [NAN, NAN], [2, 3], [np.inf, 0]])
        assert count_finite(matrix, chunk_rows=chunk_rows).tolist() == [1, 0, 2, 1]
        assert finite_counts(matrix, chunk_rows=chunk_rows)[1].tolist() == [2, 2]

    def test_empty_matrix(self) -> None:
        assert count_finite(np.empty((0, 3))).tolist() == []


class TestDiscoverDatasets:
    def test_excluded_names_are_skipped(self, two_datasets: list[UPath]) -> None:
        found = discover_datasets(two_datasets[0].parent, exclude=("B",))
        assert [p.stem for p in found] == ["A"]


class TestReadSnapshot:
    def test_backed_read_reduces_to_counts(self, two_datasets: list[UPath]) -> None:
        snap = read_snapshot(two_datasets[0])
        assert list(snap.response_samples) == ["c1", "c2"]
        assert snap.n_drugs == 2
        assert snap.n_measured == 3
        assert snap.tissues.to_dict() == {"c1": "Lung", "c2": "Skin"}
        assert snap.omics_features == {"gene_expression": 4, "mutations": 5}
        assert list(snap.replicates.samples) == ["c1"]
        assert snap.replicates.n_pairs == 1
        assert snap.replicates.n_biological_pairs == 1

    def test_blank_tissue_becomes_unknown(self, two_datasets: list[UPath]) -> None:
        snap = read_snapshot(two_datasets[1])
        assert snap.tissues["c4"] == "Unknown"


class TestReplicateCounts:
    def test_a_pair_counts_from_two_fitted_series(self) -> None:
        counts = replicate_counts(
            _fits(
                [
                    ("s1", "d1", "fitted", 0),
                    ("s1", "d1", "fitted", 0),
                    ("s1", "d1", "fitted", 1),
                    ("s1", "d2", "fitted", 0),
                    ("s1", "d2", "too_few_doses", 1),
                    ("s2", "d1", "fitted", 0),
                    ("s2", "d1", "fitted", 0),
                    ("s3", "d1", "fit_failed", 0),
                    ("s3", "d1", "fit_failed", 1),
                ]
            )
        )
        assert list(counts.samples) == ["s1", "s2"]
        assert counts.n_pairs == 2

    def test_pairs_split_by_their_fitted_biological_replicates(self) -> None:
        counts = replicate_counts(
            _fits(
                [
                    ("s1", "d1", "fitted", 0),
                    ("s1", "d1", "fitted", 0),
                    ("s1", "d1", "fitted", 1),
                    ("s2", "d1", "fitted", 0),
                    ("s2", "d1", "fitted", 0),
                    ("s2", "d1", "fit_failed", 1),
                ]
            )
        )
        assert (counts.n_biological_pairs, counts.n_technical_only_pairs) == (1, 1)
        assert counts.median_biological == 1.5
        assert counts.median_technical == 2.0

    def test_no_table_means_no_replicates(self) -> None:
        counts = replicate_counts(None)
        assert counts.n_pairs == 0
        assert counts.samples.empty
        assert np.isnan(counts.median_biological)

    def test_a_dataset_without_a_table_reads_as_unreplicated(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "C.h5mu", {"response": _anndata(["c1"], np.array([[1.0]]), "d")}, {"c1": "Lung"})
        snap = read_snapshot(path)
        assert snap.replicates.n_pairs == 0


class TestSummaryTables:
    def test_overview(self, two_datasets: list[UPath]) -> None:
        overview = summary.dataset_overview(read_snapshots(two_datasets))
        assert overview.loc["A", "samples"] == 2
        assert overview.loc["A", "density"] == pytest.approx(0.75)
        assert overview.loc["A", "replicate_pairs"] == 1
        assert overview.loc["A", "replicate_samples"] == 1
        assert overview.loc["B", "replicate_pairs"] == 0

    def test_replicate_structure_has_a_row_per_dataset(self, two_datasets: list[UPath]) -> None:
        table = summary.replicate_structure(read_snapshots(two_datasets))
        assert table[["replicate_pairs", "biological_pairs", "technical_only_pairs"]].to_dict("index") == {
            "A": {"replicate_pairs": 1, "biological_pairs": 1, "technical_only_pairs": 0},
            "B": {"replicate_pairs": 0, "biological_pairs": 0, "technical_only_pairs": 0},
        }
        assert table.loc["A", "median_technical"] == 1.0
        assert np.isnan(table.loc["B", "median_biological"])

    def test_omics_coverage_is_a_full_grid_relative_to_response_lines(self, two_datasets: list[UPath]) -> None:
        coverage = summary.omics_coverage(read_snapshots(two_datasets)).set_index(["dataset", "modality"])
        # c3 has expression but no response, so it does not count.
        assert coverage.loc[("A", "gene_expression"), "samples"] == 2
        assert coverage.loc[("A", "mutations"), "fraction"] == pytest.approx(0.5)
        assert coverage.loc[("B", "mutations"), "samples"] == 0

    def test_union_coverage_and_overlap(self, two_datasets: list[UPath]) -> None:
        snaps = read_snapshots(two_datasets)
        union = summary.union_coverage(summary.modality_membership(snaps))
        assert union.loc["response", "unique_samples"] == 3
        assert union.loc["gene_expression", "unique_samples"] == 3
        indicators = summary.dataset_indicators(snaps)
        assert list(indicators.columns) == ["A", "B"]
        assert indicators.loc["c2"].tolist() == [True, True]
        assert indicators.loc["c4"].tolist() == [False, True]
        assert summary.dataset_multiplicity(snaps).to_dict() == {1: 2, 2: 1}

    def test_drug_indicators_count_only_measured_drugs(self, tmp_path: Path) -> None:
        nan_column = np.array([[1.0, NAN], [2.0, NAN]])
        a = _write(
            tmp_path / "A.h5mu", {"response": _anndata(["c1", "c2"], nan_column, "d")}, {"c1": "Lung", "c2": "Lung"}
        )
        b = _write(
            tmp_path / "B.h5mu", {"response": _anndata(["c1"], np.array([[1.0, 2.0, 3.0]]), "d")}, {"c1": "Lung"}
        )
        indicators = summary.dataset_indicators(read_snapshots([a, b]), "drugs")
        # d1 is an all-NaN column in A, so only B measured it.
        assert indicators.to_dict("index") == {
            "d0": {"A": True, "B": True},
            "d1": {"A": False, "B": True},
            "d2": {"A": False, "B": True},
        }
        assert indicators.index.name == "drug"

    def test_tissue_counts(self, two_datasets: list[UPath]) -> None:
        counts = summary.tissue_counts(read_snapshots(two_datasets))
        assert set(map(tuple, counts.to_numpy().tolist())) == {
            ("A", "Lung", 1),
            ("A", "Skin", 1),
            ("B", "Skin", 1),
            ("B", "Unknown", 1),
        }
