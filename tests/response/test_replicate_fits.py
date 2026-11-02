"""Tests for re-keying and restricting the replicate-fits table."""

import numpy as np
import pandas as pd
import pytest

from data_curation.response.replicate_fits import DRUG_COLUMN, canonicalize, restrict


def _native_table() -> pd.DataFrame:
    """A table as ``curate()`` writes it: keyed by native labels, text as categoricals."""
    samples = ["c1", "c2", "c1", "c3"]
    drugs = ["d1", "d2", "d2", "d1"]
    table = pd.DataFrame(
        {
            "sample_id": samples,
            "pubchem_id": drugs,
            "native_sample": samples,
            "native_drug": drugs,
            "replicate": np.arange(4, dtype=np.int32),
            "status": ["fitted", "fitted", "too_few_doses", "fitted"],
            "pEC50": np.array([1.0, 2.0, np.nan, 3.0], dtype=np.float32),
        }
    )
    for column in ("sample_id", "pubchem_id", "native_sample", "native_drug", "status"):
        table[column] = table[column].astype("category")
    return table


_SAMPLES = pd.Series({"c1": "C1", "c2": "C2"})
_DRUGS = pd.Series({"d1": "1", "d2": "2"})


class TestCanonicalize:
    def test_keys_are_renamed_and_natives_kept(self) -> None:
        table = canonicalize(_native_table(), _SAMPLES, _DRUGS)
        assert table[["sample_id", DRUG_COLUMN, "native_sample"]].astype(str).values.tolist() == [
            ["C1", "1", "c1"],
            ["C2", "2", "c2"],
            ["C1", "2", "c1"],
        ]

    def test_unresolved_rows_are_dropped_and_their_categories_with_them(self) -> None:
        table = canonicalize(_native_table(), _SAMPLES, _DRUGS)
        assert "c3" not in table["native_sample"].cat.categories
        assert list(table["status"].cat.categories) == ["fitted", "too_few_doses"]

    def test_the_column_order_is_kept(self) -> None:
        assert list(canonicalize(_native_table(), _SAMPLES, _DRUGS).columns) == list(_native_table().columns)

    def test_two_natives_onto_one_canonical_keep_their_own_rows(self) -> None:
        table = canonicalize(_native_table(), pd.Series({"c1": "C", "c2": "C", "c3": "C"}), _DRUGS)
        assert table["sample_id"].astype(str).tolist() == ["C"] * 4
        assert table["native_sample"].astype(str).tolist() == ["c1", "c2", "c1", "c3"]


def _canonical_table() -> pd.DataFrame:
    return canonicalize(_native_table(), pd.Series({"c1": "C1", "c2": "C2", "c3": "C3"}), _DRUGS)


class TestRestrict:
    def test_only_listed_samples_and_drugs_are_kept(self) -> None:
        table = restrict(_canonical_table(), pd.Index(["C2"]), pd.Index(["1", "2"]))
        assert table[["sample_id", DRUG_COLUMN]].astype(str).values.tolist() == [["C2", "2"]]

    def test_unused_categories_are_dropped(self) -> None:
        table = restrict(_canonical_table(), pd.Index(["C2"]), pd.Index(["2"]))
        assert list(table["sample_id"].cat.categories) == ["C2"]

    @pytest.mark.parametrize("samples", [pd.Index([]), pd.Index(["nope"])])
    def test_nothing_matching_gives_an_empty_table(self, samples: pd.Index) -> None:
        assert restrict(_canonical_table(), samples, pd.Index(["1"])).empty
