"""Tests for Cellosaurus-backed name resolution and quality flags."""

import pandas as pd
import pytest
from upath import UPath

from data_curation.cell_lines.cellosaurus import (
    build_fallback_name_map,
    load_cellosaurus,
    quality_flags,
    resolve_names,
)

COLUMNS = [
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
def cellosaurus_csv(tmp_dir: UPath) -> UPath:
    """A minimal parsed Cellosaurus table covering the interesting cases."""
    rows = [
        {"name": "BT-159", "cellosaurus_id": "CVCL_WW42", "synonyms": "BT159"},
        {"name": "Hs 729", "cellosaurus_id": "CVCL_0871", "synonyms": "HS-729; Hs729.T"},
        {
            "name": "U-87MG ATCC",
            "cellosaurus_id": "CVCL_0022",
            "synonyms": "U87MG",
            "problem_description": "Misidentified.",
            "problematic": True,
        },
        {"name": "PrimaryWins", "cellosaurus_id": "CVCL_0001"},
        {"name": "OtherLine", "cellosaurus_id": "CVCL_0002", "synonyms": "PrimaryWins"},
    ]
    frame = pd.DataFrame(rows)
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = "" if column != "problematic" else False
    frame["problematic"] = frame["problematic"].astype("boolean").fillna(False).astype(bool)
    path = tmp_dir / "cellosaurus.parquet"
    frame[COLUMNS].to_parquet(path, index=False)
    return path


class TestLoadCellosaurus:
    def test_missing_file_names_the_stage_to_run(self, tmp_dir: UPath) -> None:
        with pytest.raises(FileNotFoundError, match="download_cellosaurus"):
            load_cellosaurus(tmp_dir / "absent.parquet")

    def test_problematic_is_boolean(self, cellosaurus_csv: UPath) -> None:
        assert load_cellosaurus(cellosaurus_csv)["problematic"].dtype == bool


class TestBuildFallbackNameMap:
    def test_indexes_primary_names_and_synonyms(self, cellosaurus_csv: UPath) -> None:
        mapping = build_fallback_name_map(cellosaurus_csv)
        assert mapping["BT159"] == "CVCL_WW42"
        assert mapping["HS729"] == "CVCL_0871"
        assert mapping["HS729T"] == "CVCL_0871"

    def test_a_primary_name_beats_another_row_synonym(self, cellosaurus_csv: UPath) -> None:
        assert build_fallback_name_map(cellosaurus_csv)["PRIMARYWINS"] == "CVCL_0001"


class TestResolveNames:
    PRIMARY = {"KNOWN": ("CVCL_1111", "Known")}

    def test_primary_map_is_used_when_it_matches(self, cellosaurus_csv: UPath) -> None:
        names = pd.Series(["Known"])
        resolved, rescued = resolve_names(names, self.PRIMARY, build_fallback_name_map(cellosaurus_csv))
        assert list(resolved) == ["CVCL_1111"]
        assert rescued == {}

    def test_fallback_rescues_what_the_primary_map_lacks(self, cellosaurus_csv: UPath) -> None:
        names = pd.Series(["Known", "BT159"])
        resolved, rescued = resolve_names(names, self.PRIMARY, build_fallback_name_map(cellosaurus_csv))
        assert list(resolved) == ["CVCL_1111", "CVCL_WW42"]
        assert rescued == {"BT159": "CVCL_WW42"}

    def test_primary_wins_over_the_fallback(self, cellosaurus_csv: UPath) -> None:
        """Accessions must agree with the omics modalities, which are CMP-keyed."""
        primary = {"BT159": ("CVCL_PRIMARY", "BT-159")}
        resolved, rescued = resolve_names(pd.Series(["BT159"]), primary, build_fallback_name_map(cellosaurus_csv))
        assert list(resolved) == ["CVCL_PRIMARY"]
        assert rescued == {}

    def test_without_a_fallback_nothing_is_rescued(self) -> None:
        resolved, rescued = resolve_names(pd.Series(["BT159"]), self.PRIMARY, None)
        assert resolved.isna().all()
        assert rescued == {}

    def test_unresolvable_names_stay_missing(self, cellosaurus_csv: UPath) -> None:
        resolved, rescued = resolve_names(
            pd.Series(["NeverHeardOfIt"]), self.PRIMARY, build_fallback_name_map(cellosaurus_csv)
        )
        assert resolved.isna().all()
        assert rescued == {}

    def test_punctuation_differences_still_match(self, cellosaurus_csv: UPath) -> None:
        resolved, _ = resolve_names(pd.Series(["bt-159"]), {}, build_fallback_name_map(cellosaurus_csv))
        assert list(resolved) == ["CVCL_WW42"]

    def test_nan_names_are_left_alone(self, cellosaurus_csv: UPath) -> None:
        names = pd.Series(["BT159", None])
        resolved, rescued = resolve_names(names, {}, build_fallback_name_map(cellosaurus_csv))
        assert resolved.iloc[0] == "CVCL_WW42"
        assert pd.isna(resolved.iloc[1])
        assert rescued == {"BT159": "CVCL_WW42"}


class TestQualityFlags:
    def test_flags_a_problematic_line(self, cellosaurus_csv: UPath) -> None:
        flags = quality_flags(pd.Index(["CVCL_0022", "CVCL_WW42"]), cellosaurus_csv)
        assert bool(flags.loc["CVCL_0022", "problematic"]) is True
        assert bool(flags.loc["CVCL_WW42", "problematic"]) is False

    def test_unknown_accessions_default_to_not_problematic(self, cellosaurus_csv: UPath) -> None:
        flags = quality_flags(pd.Index(["CVCL_NOPE"]), cellosaurus_csv)
        assert bool(flags.loc["CVCL_NOPE", "problematic"]) is False
        assert flags.loc["CVCL_NOPE", "problem_description"] == ""

    def test_output_is_aligned_to_the_requested_index(self, cellosaurus_csv: UPath) -> None:
        index = pd.Index(["CVCL_WW42", "CVCL_0022", "CVCL_NOPE"])
        assert list(quality_flags(index, cellosaurus_csv).index) == list(index)
