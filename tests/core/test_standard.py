"""Tests for the standard-format file contract."""

import numpy as np
import pandas as pd
import pytest
from upath import UPath

from data_curation.core import standard
from data_curation.core.errors import CurationError


def _wells(**overrides) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "sample": ["S1", "S1"],
            "drug": ["d1", "d1"],
            "concentration": [0.1, 1.0],
            "intensity": [1.0, 0.5],
        }
    )
    for column, values in overrides.items():
        frame[column] = values
    return frame


class TestValidate:
    def test_a_minimal_well_table_passes_with_normalised_dtypes(self) -> None:
        frame = _wells(concentration=[1, 10])
        out = standard.validate("wells", frame)
        assert out["concentration"].dtype == np.float64
        assert out["sample"].tolist() == ["S1", "S1"]

    def test_optional_and_metadata_columns_are_kept(self) -> None:
        out = standard.validate("wells", _wells(experiment=["e", "e"], replicate=[1, 2], plate=["p", "p"]))
        assert out["replicate"].tolist() == ["1", "2"]
        assert out["plate"].tolist() == ["p", "p"]

    def test_a_missing_required_column_is_named(self) -> None:
        with pytest.raises(CurationError, match="intensity"):
            standard.validate("wells", _wells().drop(columns="intensity"))

    def test_a_missing_key_value_is_rejected(self) -> None:
        with pytest.raises(CurationError, match="'sample' has 1 missing"):
            standard.validate("wells", _wells(sample=["S1", None]))

    def test_an_empty_key_value_is_rejected(self) -> None:
        with pytest.raises(CurationError, match="empty"):
            standard.validate("wells", _wells(drug=["d1", " "]))

    def test_a_non_positive_concentration_is_rejected(self) -> None:
        with pytest.raises(CurationError, match="not positive"):
            standard.validate("wells", _wells(concentration=[0.0, 1.0]))

    def test_a_missing_intensity_is_rejected(self) -> None:
        with pytest.raises(CurationError, match="'intensity' has 1 missing or infinite"):
            standard.validate("wells", _wells(intensity=[np.nan, 1.0]))

    def test_a_non_numeric_column_is_rejected(self) -> None:
        with pytest.raises(CurationError, match="not numeric"):
            standard.validate("wells", _wells(intensity=["a", "b"]))

    def test_duplicate_drugs_are_rejected(self) -> None:
        frame = pd.DataFrame({"drug": ["a", "a"], "drug_name": ["A", "A"]})
        with pytest.raises(CurationError, match="repeat"):
            standard.validate("drugs", frame)

    def test_published_metrics_become_floats_and_may_be_missing(self) -> None:
        frame = pd.DataFrame({"sample": ["S1"], "drug": ["d1"], "AUC": [None]})
        assert np.isnan(standard.validate("published", frame)["AUC"].iloc[0])

    def test_published_needs_a_metric(self) -> None:
        with pytest.raises(CurationError, match="no metric"):
            standard.validate("published", pd.DataFrame({"sample": ["S1"], "drug": ["d1"]}))

    def test_sample_extras_are_strings(self) -> None:
        frame = pd.DataFrame({"sample_id": ["S1", "S2"], "tissue": ["Lung", "Lung"], "age": [3, None]})
        assert standard.validate("samples", frame)["age"].tolist() == ["3", ""]

    def test_duplicate_samples_are_rejected(self) -> None:
        frame = pd.DataFrame({"sample_id": ["S1", "S1"], "tissue": ["Lung", "Lung"]})
        with pytest.raises(CurationError, match="repeat"):
            standard.validate("samples", frame)

    def test_negative_tpm_is_rejected(self) -> None:
        frame = pd.DataFrame({"sample_id": ["S1"], "gene_symbol": ["TP53"], "tpm": [-1.0]})
        with pytest.raises(CurationError, match="negative"):
            standard.validate("expression", frame)

    def test_counts_may_be_missing(self) -> None:
        frame = pd.DataFrame({"sample_id": ["S1"], "gene_symbol": ["TP53"], "tpm": [1.0], "counts": [None]})
        assert np.isnan(standard.validate("expression", frame)["counts"].iloc[0])


class TestRead:
    def test_reads_and_validates_a_file(self, tmp_dir: UPath) -> None:
        _wells().to_parquet(str(standard.path("wells", tmp_dir)))
        frame = standard.read("wells", tmp_dir)
        assert frame is not None
        assert len(frame) == 2

    def test_errors_name_the_file(self, tmp_dir: UPath) -> None:
        _wells().drop(columns="drug").to_parquet(str(standard.path("wells", tmp_dir)))
        with pytest.raises(CurationError, match=r"wells\.parquet lacks"):
            standard.read("wells", tmp_dir)

    def test_a_missing_required_file_is_an_error(self, tmp_dir: UPath) -> None:
        with pytest.raises(CurationError, match="missing"):
            standard.read("drugs", tmp_dir)

    def test_a_missing_optional_file_is_none(self, tmp_dir: UPath) -> None:
        assert standard.read("published", tmp_dir) is None
