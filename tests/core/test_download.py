"""Tests for the download helpers' archive deletion and source pruning."""

import pytest
from upath import UPath

from data_curation.core.download import _delete_archives, prune_unused


def _touch(directory: UPath, name: str, size: int = 1) -> UPath:
    path = directory / name
    path.write_bytes(b"x" * size)
    return path


class TestDeleteArchives:
    def test_removes_zip_and_leaves_everything_else(self, tmp_dir: UPath) -> None:
        _touch(tmp_dir, "sources.zip")
        kept = _touch(tmp_dir, "data.csv")
        gz = _touch(tmp_dir, "proteomics_latest.csv.gz")
        _delete_archives(tmp_dir)
        assert sorted(p.name for p in tmp_dir.iterdir()) == [kept.name, gz.name]

    def test_is_a_no_op_without_archives(self, tmp_dir: UPath) -> None:
        _touch(tmp_dir, "data.csv")
        _delete_archives(tmp_dir)
        assert [p.name for p in tmp_dir.iterdir()] == ["data.csv"]


class TestPruneUnused:
    def test_empty_keep_list_prunes_nothing(self, tmp_dir: UPath) -> None:
        """No `keep` must mean "keep everything", so existing entries are unaffected."""
        _touch(tmp_dir, "huge.csv")
        prune_unused(tmp_dir, [])
        assert [p.name for p in tmp_dir.iterdir()] == ["huge.csv"]

    def test_keeps_matches_and_deletes_the_rest(self, tmp_dir: UPath) -> None:
        _touch(tmp_dir, "rnaseq_merged_rsem_tpm_20260323.csv")
        _touch(tmp_dir, "rnaseq_merged_rsem_expected_count_20260323.csv")
        _touch(tmp_dir, "rnaseq_merged_20260323.csv")
        _touch(tmp_dir, "rnaseq_merged_rsem_fpkm_20260323.csv")
        prune_unused(tmp_dir, ["*rsem_tpm*", "*expected_count*"])
        assert sorted(p.name for p in tmp_dir.iterdir()) == [
            "rnaseq_merged_rsem_expected_count_20260323.csv",
            "rnaseq_merged_rsem_tpm_20260323.csv",
        ]

    def test_named_source_survives_without_being_named(self, tmp_dir: UPath) -> None:
        """compounds.csv is addressed by name, so no keep pattern lists it."""
        _touch(tmp_dir, "compounds.csv")
        _touch(tmp_dir, "junk.txt")
        prune_unused(tmp_dir, ["*gistic*"])
        assert [p.name for p in tmp_dir.iterdir()] == ["compounds.csv"]

    def test_archive_is_deleted_even_when_a_keep_pattern_matches_it(self, tmp_dir: UPath) -> None:
        """The real GDSC trap: the archive shares the extracted file's name.

        ``GDSC2_public_raw_data_27Oct23.csv.zip`` matches ``*public_raw_data*``,
        the very pattern that selects the extracted CSV, so a naive keep-set
        retains a redundant 291 MiB archive.
        """
        _touch(tmp_dir, "GDSC2_public_raw_data_27Oct23.csv")
        _touch(tmp_dir, "abc123-GDSC2_public_raw_data_27Oct23.csv.zip")
        prune_unused(tmp_dir, ["*public_raw_data*"])
        assert [p.name for p in tmp_dir.iterdir()] == ["GDSC2_public_raw_data_27Oct23.csv"]

    def test_mutations_archive_is_deleted_but_its_csv_kept(self, tmp_dir: UPath) -> None:
        _touch(tmp_dir, "mutations_all_20260724.csv")
        _touch(tmp_dir, "abc-mutations_all_20260724.zip")
        prune_unused(tmp_dir, ["*mutations_all*"])
        assert [p.name for p in tmp_dir.iterdir()] == ["mutations_all_20260724.csv"]

    def test_gzipped_data_file_is_kept(self, tmp_dir: UPath) -> None:
        """A .csv.gz is read in place; only .zip archives are extraction leftovers."""
        _touch(tmp_dir, "abc-proteomics_latest.csv.gz")
        prune_unused(tmp_dir, ["*proteomics*"])
        assert [p.name for p in tmp_dir.iterdir()] == ["abc-proteomics_latest.csv.gz"]

    def test_subdirectories_are_left_alone(self, tmp_dir: UPath) -> None:
        (tmp_dir / "nested").mkdir()
        _touch(tmp_dir / "nested", "inner.csv")
        _touch(tmp_dir, "junk.txt")
        prune_unused(tmp_dir, ["*gistic*"])
        assert [p.name for p in tmp_dir.iterdir()] == ["nested"]

    @pytest.mark.parametrize("pattern", ["*rsem_tpm*", "rnaseq_merged_rsem_tpm*", "*tpm*"])
    def test_equivalent_patterns_all_retain_the_target(self, tmp_dir: UPath, pattern: str) -> None:
        target = _touch(tmp_dir, "rnaseq_merged_rsem_tpm_20260323.csv")
        _touch(tmp_dir, "other.csv")
        prune_unused(tmp_dir, [pattern])
        assert [p.name for p in tmp_dir.iterdir()] == [target.name]
