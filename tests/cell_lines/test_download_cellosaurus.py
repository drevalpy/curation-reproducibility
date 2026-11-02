"""Tests for the Cellosaurus flat-file parser."""

from upath import UPath

from data_curation.cell_lines.download_cellosaurus import _parse_cellosaurus

RECORD = """\
ID   U-87MG ATCC
AC   CVCL_0022
AS   CVCL_9999
SY   U-87MG; U-87 MG; U87MG
SX   Male
AG   44Y
CA   Cancer cell line
OX   NCBI_TaxID=9606; ! Homo sapiens (Human)
DI   NCIt; C3058; Glioblastoma
DR   ATCC; HTB-14
DR   DepMap; ACH-000075
HI   CVCL_D145 ! Some parent
OI   CVCL_D2Y8 ! Some sibling
CC   Problematic cell line: Misidentified. Not the original 1968 line.
CC   Discontinued: ECACC; 94050311; true.
CC   Derived from site: In situ; Brain; UBERON=UBERON_0000955.
//
ID   CleanLine
AC   CVCL_0001
SY   Clean-Line
SX   Female
CA   Cancer cell line
OX   NCBI_TaxID=9606; ! Homo sapiens (Human)
//
"""


def _write(tmp_dir: UPath, text: str) -> UPath:
    path = tmp_dir / "cellosaurus.txt"
    path.write_text(text, encoding="utf-8")
    return path


class TestParseCellosaurus:
    def test_parses_one_row_per_accession(self, tmp_dir: UPath) -> None:
        frame = _parse_cellosaurus(_write(tmp_dir, RECORD))
        assert list(frame["cellosaurus_id"]) == ["CVCL_0022", "CVCL_0001"]

    def test_scalar_fields_are_extracted(self, tmp_dir: UPath) -> None:
        row = _parse_cellosaurus(_write(tmp_dir, RECORD)).iloc[0]
        assert row["name"] == "U-87MG ATCC"
        assert row["sex"] == "Male"
        assert row["age_at_sampling"] == "44Y"
        assert row["category"] == "Cancer cell line"

    def test_multi_value_fields_are_joined(self, tmp_dir: UPath) -> None:
        row = _parse_cellosaurus(_write(tmp_dir, RECORD)).iloc[0]
        assert row["synonyms"] == "U-87MG; U-87 MG; U87MG"
        assert "ATCC; HTB-14" in row["cross_references"]
        assert "DepMap; ACH-000075" in row["cross_references"]

    def test_secondary_accessions_are_kept(self, tmp_dir: UPath) -> None:
        """Dropping AS is a silent correctness hole: a query by an old accession misses."""
        row = _parse_cellosaurus(_write(tmp_dir, RECORD)).iloc[0]
        assert row["secondary_accessions"] == "CVCL_9999"

    def test_problem_comment_sets_the_flag_and_description(self, tmp_dir: UPath) -> None:
        frame = _parse_cellosaurus(_write(tmp_dir, RECORD))
        flagged, clean = frame.iloc[0], frame.iloc[1]
        assert bool(flagged["problematic"]) is True
        assert "Misidentified" in flagged["problem_description"]
        assert bool(clean["problematic"]) is False
        assert clean["problem_description"] == ""

    def test_only_the_wanted_comment_topics_are_extracted(self, tmp_dir: UPath) -> None:
        """CC packs dozens of topics into one code; unrelated ones must not leak."""
        row = _parse_cellosaurus(_write(tmp_dir, RECORD)).iloc[0]
        assert row["discontinued"] == "ECACC; 94050311; true."
        assert "Derived from site" not in row["problem_description"]
        assert row["misspelling"] == ""

    def test_hierarchy_fields_are_captured(self, tmp_dir: UPath) -> None:
        row = _parse_cellosaurus(_write(tmp_dir, RECORD)).iloc[0]
        assert row["derived_from"].startswith("CVCL_D145")
        assert row["same_individual"].startswith("CVCL_D2Y8")

    def test_a_final_record_without_a_trailing_separator_is_kept(self, tmp_dir: UPath) -> None:
        text = "ID   Lonely\nAC   CVCL_7777\n"
        assert list(_parse_cellosaurus(_write(tmp_dir, text))["cellosaurus_id"]) == ["CVCL_7777"]

    def test_records_without_an_accession_are_dropped(self, tmp_dir: UPath) -> None:
        text = "ID   NoAccession\nSX   Male\n//\nID   Fine\nAC   CVCL_0001\n//\n"
        assert list(_parse_cellosaurus(_write(tmp_dir, text))["cellosaurus_id"]) == ["CVCL_0001"]

    def test_repeated_comments_of_one_topic_are_joined(self, tmp_dir: UPath) -> None:
        text = (
            "ID   Two\nAC   CVCL_0002\n"
            "CC   Problematic cell line: First problem.\n"
            "CC   Problematic cell line: Second problem.\n//\n"
        )
        row = _parse_cellosaurus(_write(tmp_dir, text)).iloc[0]
        assert row["problem_description"] == "First problem. | Second problem."

    def test_an_empty_file_yields_an_empty_frame_with_columns(self, tmp_dir: UPath) -> None:
        frame = _parse_cellosaurus(_write(tmp_dir, ""))
        assert frame.empty
        assert "problematic" in frame.columns
