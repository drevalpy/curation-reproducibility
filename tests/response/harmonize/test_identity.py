"""Tests for native -> canonical identity resolution in harmonize_response."""

import pandas as pd
import pytest
from upath import UPath

from data_curation.cell_lines.cellosaurus import build_fallback_name_map
from data_curation.core.errors import CurationError
from data_curation.response.harmonize import identity as identity_module
from data_curation.response.harmonize.identity import (
    drop_unresolved,
    resolve_drugs,
    resolve_samples,
)

CELLOSAURUS_COLUMNS = [
    "name",
    "cellosaurus_id",
    "sex",
    "age_at_sampling",
    "category",
    "synonyms",
    "species",
    "cross_references",
    "secondary_accessions",
    "derived_from",
    "same_individual",
    "disease",
    "problem_description",
    "discontinued",
    "misspelling",
    "problematic",
]


@pytest.fixture
def raw_dir(write_model_list) -> UPath:
    """A raw response directory carrying a one-line Cell Model Passports list."""
    return write_model_list(
        [{"model_id": "SIDM00001", "model_name": "MEC-1", "RRID": "CVCL_1870", "synonyms": "MEC1"}]
    ).parent


@pytest.fixture
def cellosaurus_only_line(tmp_dir: UPath, monkeypatch) -> None:
    """Point the Cellosaurus fallback at a table holding one CMP-less line.

    ``BT-159`` stands in for the 34 ``BT*`` glioma stem-cell lines Cell Model
    Passports does not carry at all. The default argument of
    ``build_fallback_name_map`` is bound at import time, so the lookup is
    redirected by rebinding the name the module under test resolves.
    """
    frame = pd.DataFrame([{"name": "BT-159", "cellosaurus_id": "CVCL_WW42", "synonyms": "BT159"}])
    for column in CELLOSAURUS_COLUMNS:
        if column not in frame.columns:
            frame[column] = False if column == "problematic" else ""
    path = tmp_dir / "cellosaurus.parquet"
    frame[CELLOSAURUS_COLUMNS].to_parquet(str(path), index=False)
    monkeypatch.setattr(
        identity_module,
        "build_fallback_name_map",
        lambda: build_fallback_name_map(path),
    )


class TestResolveSamplesBySidm:
    def test_the_accession_and_name_are_both_resolved(self, raw_dir: UPath) -> None:
        identity = resolve_samples(pd.Index(["SIDM00001"]), "gdsc", raw_dir / "model_list.csv")
        assert identity.canonical["SIDM00001"] == "CVCL_1870"
        assert identity.cell_line_names["CVCL_1870"] == "MEC-1"

    def test_an_unknown_model_id_stays_unresolved(self, raw_dir: UPath) -> None:
        identity = resolve_samples(pd.Index(["SIDM99999"]), "gdsc", raw_dir / "model_list.csv")
        assert identity.canonical["SIDM99999"] is None


class TestResolveSamplesByName:
    @pytest.mark.parametrize("fmt", ["ctrpv1", "ctrpv2", "ccle"])
    def test_cmp_names_resolve_first(self, raw_dir: UPath, cellosaurus_only_line, fmt: str) -> None:
        identity = resolve_samples(pd.Index(["MEC1"]), fmt, raw_dir / "model_list.csv")
        assert identity.canonical["MEC1"] == "CVCL_1870"

    def test_cellosaurus_rescues_what_cmp_lacks(self, raw_dir: UPath, cellosaurus_only_line) -> None:
        """The 5.7% of CTRPv2 measurements only Cellosaurus can resolve."""
        identity = resolve_samples(pd.Index(["BT-159"]), "ctrpv2", raw_dir / "model_list.csv")
        assert identity.canonical["BT-159"] == "CVCL_WW42"

    def test_a_rescued_line_carries_its_name(self, raw_dir: UPath, cellosaurus_only_line) -> None:
        """Rescued lines have no CMP entry, so their name has to come back too."""
        identity = resolve_samples(pd.Index(["BT-159"]), "ctrpv2", raw_dir / "model_list.csv")
        assert identity.cell_line_names["CVCL_WW42"] == "BT-159"

    def test_the_rescue_is_logged(self, raw_dir: UPath, cellosaurus_only_line, caplog) -> None:
        with caplog.at_level("INFO"):
            resolve_samples(pd.Index(["BT-159"]), "ctrpv2", raw_dir / "model_list.csv")
        assert "Cellosaurus resolved" in caplog.text

    def test_an_unknown_name_stays_unresolved(self, raw_dir: UPath, cellosaurus_only_line) -> None:
        identity = resolve_samples(pd.Index(["NeverHeardOfIt"]), "ccle", raw_dir / "model_list.csv")
        assert pd.isna(identity.canonical["NeverHeardOfIt"])


class TestResolveSamplesIdentity:
    @pytest.mark.parametrize("fmt", ["bruna", "beataml"])
    def test_specimen_ids_pass_straight_through(self, fmt: str) -> None:
        """PDTCs and patient specimens have no accession, so the ID is canonical."""
        identity = resolve_samples(pd.Index(["2100", "2101"]), fmt)
        assert identity.canonical.to_dict() == {"2100": "2100", "2101": "2101"}
        assert identity.cell_line_names == {}


class TestUnknownFormat:
    def test_an_unroutable_format_is_rejected(self, tmp_dir: UPath) -> None:
        with pytest.raises(CurationError, match="No sample resolution route"):
            resolve_samples(pd.Index(["x"]), "nonesuch", tmp_dir / "model_list.csv")


class TestResolveDrugs:
    MAPPING = pd.DataFrame(
        {
            "DRUG_ID": ["1001", "1002", "1003"],
            "pubchem_id": ["31703", "31703", "GDSC_1003"],
            "canonical_smiles": ["", "", ""],
        }
    )

    def test_native_drug_ids_become_cids(self) -> None:
        resolved = resolve_drugs(pd.Index(["1001"]), self.MAPPING)
        assert resolved["1001"] == "31703"

    def test_repeat_screens_share_one_cid(self) -> None:
        """The GDSC1 case: 402 DRUG_IDs onto 378 CIDs, all repeat screens."""
        resolved = resolve_drugs(pd.Index(["1001", "1002"]), self.MAPPING)
        assert resolved.tolist() == ["31703", "31703"]

    def test_pseudo_ids_are_kept(self) -> None:
        """An unresolvable drug keeps its pseudo-ID so its data is not dropped."""
        assert resolve_drugs(pd.Index(["1003"]), self.MAPPING)["1003"] == "GDSC_1003"

    def test_numeric_native_ids_match_a_string_mapping(self) -> None:
        """CurveCurator round-trips labels as strings, the mapping CSV as ints."""
        assert resolve_drugs(pd.Index([1001]), self.MAPPING)["1001"] == "31703"

    def test_an_unmapped_drug_stays_unresolved(self) -> None:
        assert pd.isna(resolve_drugs(pd.Index(["9999"]), self.MAPPING)["9999"])


class TestDropUnresolved:
    def test_everything_resolved_is_a_no_op(self) -> None:
        canonical = pd.Series({"a": "CVCL_A"})
        assert drop_unresolved(canonical, axis_name="sample", failure_fraction=0.1).to_dict() == {"a": "CVCL_A"}

    def test_unresolved_entries_are_dropped(self) -> None:
        canonical = pd.Series({"a": "CVCL_A", "b": None, "c": "CVCL_C"})
        kept = drop_unresolved(canonical, axis_name="sample", failure_fraction=0.5)
        assert list(kept.index) == ["a", "c"]

    def test_the_loss_is_reported_with_its_share(self) -> None:
        canonical = pd.Series({"a": "CVCL_A", "b": None})
        with pytest.raises(CurationError, match="50.00%"):
            drop_unresolved(canonical, axis_name="sample", failure_fraction=0.1)

    def test_a_small_loss_only_warns(self, caplog) -> None:
        canonical = pd.Series({f"a{i}": "CVCL_A" for i in range(99)} | {"b": None})
        with caplog.at_level("WARNING"):
            kept = drop_unresolved(canonical, axis_name="drug", failure_fraction=0.1)
        assert len(kept) == 99
        assert "Unmapped drug" in caplog.text
