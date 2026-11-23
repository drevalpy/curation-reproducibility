"""Tests for resolving COSMIC IDs and CCLE names to Cellosaurus accessions."""

import logging

import pandas as pd
import pytest
from upath import UPath

from data_curation.cell_lines.native_ids import (
    build_id_to_cellosaurus_map,
    resolve_ccle_names,
    resolve_cosmic_ids,
    strip_ccle_name,
)

MODELS = [
    {"model_id": "SIDM00001", "model_name": "A549", "RRID": "CVCL_0023", "COSMIC_ID": 905949, "CCLE_ID": "A549_LUNG"},
    {
        "model_id": "SIDM00002",
        "model_name": "MEC-1",
        "RRID": "CVCL_1870",
        "COSMIC_ID": "1331031;1331030",
        "CCLE_ID": "MEC1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE",
    },
    {"model_id": "SIDM00003", "model_name": "NoAccession", "RRID": pd.NA, "COSMIC_ID": 111, "CCLE_ID": "NOACC_LUNG"},
    {"model_id": "SIDM00004", "model_name": "Twin", "RRID": "CVCL_0004", "COSMIC_ID": 905949, "CCLE_ID": "A549_LUNG"},
    {"model_id": "SIDM00005", "model_name": "NoIds", "RRID": "CVCL_0005", "COSMIC_ID": pd.NA, "CCLE_ID": pd.NA},
    {"model_id": "SIDM00006", "model_name": "HCC-56", "RRID": "CVCL_0006"},
]

COLUMNS = ["model_id", "model_name", "RRID", "tissue", "synonyms", "COSMIC_ID", "CCLE_ID"]


@pytest.fixture
def model_list(write_model_list) -> UPath:
    return write_model_list(MODELS, columns=COLUMNS)


@pytest.fixture
def cellosaurus(tmp_dir: UPath) -> UPath:
    """A Cellosaurus table that knows two lines the model list lacks."""
    columns = ["name", "cellosaurus_id", "synonyms", "problem_description", "discontinued", "misspelling"]
    frame = pd.DataFrame(
        [
            {"name": "NCI-H854", "cellosaurus_id": "CVCL_1579", "synonyms": "NCIH854"},
            {"name": "OE-LE", "cellosaurus_id": "CVCL_OELE", "synonyms": ""},
        ]
    )
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    frame["problematic"] = False
    path = tmp_dir / "cellosaurus.parquet"
    frame.to_parquet(str(path), index=False)
    return path


class TestBuildIdMap:
    def test_every_value_of_a_multi_valued_cell_maps_to_the_model(self, model_list: UPath) -> None:
        mapping = build_id_to_cellosaurus_map(model_list, "COSMIC_ID")

        assert mapping["1331031"] == ("CVCL_1870", "MEC-1")
        assert mapping["1331030"] == ("CVCL_1870", "MEC-1")

    def test_the_first_model_wins_a_shared_id(self, model_list: UPath) -> None:
        assert build_id_to_cellosaurus_map(model_list, "COSMIC_ID")["905949"] == ("CVCL_0023", "A549")

    def test_models_without_an_accession_or_the_id_are_skipped(self, model_list: UPath) -> None:
        mapping = build_id_to_cellosaurus_map(model_list, "COSMIC_ID")

        assert "111" not in mapping
        assert set(mapping) == {"905949", "1331031", "1331030"}


class TestResolveCosmicIds:
    def test_integers_and_strings_resolve_to_the_same_key(self, model_list: UPath) -> None:
        resolved = resolve_cosmic_ids([905949, "1331030"], model_list)

        assert resolved == {"905949": ("CVCL_0023", "A549"), "1331030": ("CVCL_1870", "MEC-1")}

    def test_a_float_suffix_from_a_gappy_column_is_dropped(self, model_list: UPath) -> None:
        assert resolve_cosmic_ids(["905949.0"], model_list) == {"905949": ("CVCL_0023", "A549")}

    def test_unknown_ids_are_left_out_and_warned_about(self, model_list: UPath, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            resolved = resolve_cosmic_ids(["905949", "999", "905954.1"], model_list)

        assert list(resolved) == ["905949"]
        assert "2 of 3 COSMIC IDs" in caplog.text


class TestStripCcleName:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("NCIH854_LUNG", "NCIH854"),
            ("MEC1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE", "MEC1"),
            ("NOSUFFIX", "NOSUFFIX"),
        ],
    )
    def test_it_keeps_the_part_before_the_tissue(self, name: str, expected: str) -> None:
        assert strip_ccle_name(name) == expected


class TestResolveCcleNames:
    def test_names_in_the_model_list_resolve_by_ccle_id(self, model_list: UPath, cellosaurus: UPath) -> None:
        resolved = resolve_ccle_names(["A549_LUNG", "MEC1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE"], model_list, cellosaurus)

        assert resolved["A549_LUNG"] == ("CVCL_0023", "A549")
        assert resolved["MEC1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE"] == ("CVCL_1870", "MEC-1")

    def test_a_missing_name_falls_back_to_the_stripped_name_in_the_model_list(
        self, model_list: UPath, cellosaurus: UPath
    ) -> None:
        resolved = resolve_ccle_names(["HCC56_LUNG"], model_list, cellosaurus)

        assert resolved == {"HCC56_LUNG": ("CVCL_0006", "HCC-56")}

    def test_a_name_only_cellosaurus_knows_resolves_with_the_stripped_name(
        self, model_list: UPath, cellosaurus: UPath
    ) -> None:
        resolved = resolve_ccle_names(["NCIH854_LUNG"], model_list, cellosaurus)

        assert resolved == {"NCIH854_LUNG": ("CVCL_1579", "NCIH854")}

    def test_names_nothing_knows_are_left_out_and_warned_about(
        self, model_list: UPath, cellosaurus: UPath, caplog
    ) -> None:
        with caplog.at_level(logging.WARNING):
            resolved = resolve_ccle_names(["A549_LUNG", "GHOST_SKIN"], model_list, cellosaurus)

        assert list(resolved) == ["A549_LUNG"]
        assert "1 of 2 CCLE names" in caplog.text
