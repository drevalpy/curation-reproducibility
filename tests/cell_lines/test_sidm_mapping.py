"""Tests for cell-line identifier mapping."""

import pytest
from upath import UPath

from data_curation.cell_lines.sidm_mapping import (
    build_name_to_cellosaurus_map,
    build_sidm_to_cellosaurus_map,
    normalize_cell_line_name,
)


class TestNormalizeCellLineName:
    @pytest.mark.parametrize(
        "raw",
        ["MDA-MB-231", "MDAMB231", "mda_mb_231", "MDA MB 231", "mda.mb.231", "MDA/MB/231"],
    )
    def test_punctuation_variants_collapse_to_one_key(self, raw: str) -> None:
        assert normalize_cell_line_name(raw) == "MDAMB231"

    def test_digits_are_preserved(self) -> None:
        assert normalize_cell_line_name("NCI-H460") == "NCIH460"

    def test_non_string_input_is_coerced(self) -> None:
        assert normalize_cell_line_name(2004) == "2004"

    def test_empty_and_punctuation_only_normalize_to_empty(self) -> None:
        assert normalize_cell_line_name("") == ""
        assert normalize_cell_line_name("---") == ""


class TestBuildSidmToCellosaurusMap:
    def test_maps_sidm_to_accession_and_name(self, write_model_list) -> None:
        path = write_model_list([{"model_id": "SIDM00001", "model_name": "MEC-1", "RRID": "CVCL_1870"}])
        assert build_sidm_to_cellosaurus_map(path) == {"SIDM00001": ("CVCL_1870", "MEC-1")}

    def test_rows_without_a_cvcl_are_skipped(self, write_model_list) -> None:
        path = write_model_list(
            [
                {"model_id": "SIDM00001", "model_name": "MEC-1", "RRID": "CVCL_1870"},
                {"model_id": "SIDM00002", "model_name": "NoRRID", "RRID": None},
                {"model_id": "SIDM00003", "model_name": "Bogus", "RRID": "NOTACVCL"},
            ]
        )
        assert list(build_sidm_to_cellosaurus_map(path)) == ["SIDM00001"]

    def test_missing_model_name_becomes_empty_string(self, write_model_list) -> None:
        path = write_model_list([{"model_id": "SIDM00001", "model_name": None, "RRID": "CVCL_1870"}])
        assert build_sidm_to_cellosaurus_map(path)["SIDM00001"] == ("CVCL_1870", "")


class TestMultiAccessionRrid:
    """A single RRID field holding two accessions must never become an ID.

    Real row ``SIDM00400`` (SC-1) carries ``RRID = "CVCL_2717;CVCL_1888"``.
    Validation used to be a bare ``startswith("CVCL")``, so that literal string
    passed and leaked into GDSC1 as a cell-line identifier.
    """

    AMBIGUOUS = "CVCL_2717;CVCL_1888"

    def test_sidm_map_rejects_multi_accession(self, write_model_list) -> None:
        path = write_model_list([{"model_id": "SIDM00400", "model_name": "SC-1", "RRID": self.AMBIGUOUS}])
        assert build_sidm_to_cellosaurus_map(path) == {}

    def test_name_map_rejects_multi_accession(self, write_model_list) -> None:
        path = write_model_list(
            [{"model_id": "SIDM00400", "model_name": "SC-1", "RRID": self.AMBIGUOUS, "synonyms": "SC1"}]
        )
        assert build_name_to_cellosaurus_map(path) == {}

    @pytest.mark.parametrize("separator", [";", ",", "|", " "])
    def test_every_separator_is_rejected(self, write_model_list, separator: str) -> None:
        rrid = f"CVCL_2717{separator}CVCL_1888"
        path = write_model_list([{"model_id": "SIDM00400", "model_name": "SC-1", "RRID": rrid}])
        assert build_sidm_to_cellosaurus_map(path) == {}

    def test_a_clean_accession_is_still_accepted(self, write_model_list) -> None:
        path = write_model_list([{"model_id": "SIDM00400", "model_name": "SC-1", "RRID": " CVCL_2717 "}])
        assert build_sidm_to_cellosaurus_map(path)["SIDM00400"][0] == "CVCL_2717"

    def test_rejection_is_logged(self, write_model_list, caplog) -> None:
        path = write_model_list([{"model_id": "SIDM00400", "model_name": "SC-1", "RRID": self.AMBIGUOUS}])
        with caplog.at_level("WARNING"):
            build_sidm_to_cellosaurus_map(path)
        assert "SIDM00400" in caplog.text


class TestBuildNameToCellosaurusMap:
    def test_primary_name_takes_precedence_over_a_synonym(self, write_model_list) -> None:
        """A name that is one row's primary and another's synonym must resolve to the primary."""
        rows = [
            {"model_id": "SIDM2", "model_name": "OTHER", "RRID": "CVCL_0002", "synonyms": "TARGET"},
            {"model_id": "SIDM1", "model_name": "TARGET", "RRID": "CVCL_0001"},
        ]
        assert build_name_to_cellosaurus_map(write_model_list(rows))["TARGET"] == ("CVCL_0001", "TARGET")

    def test_precedence_holds_in_the_reverse_row_order(self, write_model_list) -> None:
        rows = [
            {"model_id": "SIDM1", "model_name": "TARGET", "RRID": "CVCL_0001"},
            {"model_id": "SIDM2", "model_name": "OTHER", "RRID": "CVCL_0002", "synonyms": "TARGET"},
        ]
        assert build_name_to_cellosaurus_map(write_model_list(rows))["TARGET"] == ("CVCL_0001", "TARGET")

    def test_synonyms_are_split_on_semicolons_and_normalized(self, write_model_list) -> None:
        path = write_model_list(
            [{"model_id": "SIDM1", "model_name": "MEC-1", "RRID": "CVCL_1870", "synonyms": "MEC 1;MEC1;Mec-1"}]
        )
        mapping = build_name_to_cellosaurus_map(path)
        assert mapping["MEC1"] == ("CVCL_1870", "MEC-1")
        assert mapping["MEC1"] == mapping[normalize_cell_line_name("mec_1")]

    def test_first_row_wins_for_a_duplicated_name(self, write_model_list) -> None:
        rows = [
            {"model_id": "SIDM1", "model_name": "DUP", "RRID": "CVCL_0001"},
            {"model_id": "SIDM2", "model_name": "DUP", "RRID": "CVCL_0002"},
        ]
        assert build_name_to_cellosaurus_map(write_model_list(rows))["DUP"][0] == "CVCL_0001"

    def test_empty_model_list_yields_an_empty_map(self, write_model_list) -> None:
        assert build_name_to_cellosaurus_map(write_model_list([])) == {}

    def test_missing_file_raises(self, tmp_dir: UPath) -> None:
        with pytest.raises(FileNotFoundError):
            build_name_to_cellosaurus_map(tmp_dir / "absent.csv")
