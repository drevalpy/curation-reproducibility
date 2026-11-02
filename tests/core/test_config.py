"""Tests for config/sources.yaml and config/processing.yaml access."""

import pytest

from data_curation.core import config
from data_curation.core.errors import CurationError


class TestDatasetConfig:
    def test_returns_the_requested_section(self) -> None:
        assert config.dataset_config("GDSC2", "response")["format"] == "gdsc"

    def test_returns_the_whole_block_without_a_section(self) -> None:
        assert "response" in config.dataset_config("GDSC2")

    def test_an_unknown_dataset_lists_the_known_ones(self) -> None:
        with pytest.raises(CurationError, match="GDSC2") as exc_info:
            config.dataset_config("GDSC3")
        assert "not configured" in str(exc_info.value)

    def test_a_missing_section_names_the_dataset(self) -> None:
        with pytest.raises(CurationError, match="PDX_Bruna"):
            config.dataset_config("PDX_Bruna", "cnv")


class TestDatasetNames:
    def test_lists_every_dataset_by_default(self) -> None:
        assert "GDSC2" in config.dataset_names()

    def test_scopes_to_a_modality(self) -> None:
        """PDX_Bruna has transcriptomics but no CNV, which is what scoping is for."""
        assert "PDX_Bruna" in config.dataset_names("transcriptomics")
        assert "PDX_Bruna" not in config.dataset_names("cnv")

    def test_returns_empty_outside_the_repository(self, monkeypatch, tmp_path) -> None:
        """Completion and --help must still work when there is no sources.yaml."""
        monkeypatch.chdir(tmp_path)
        assert config.dataset_names() == []


class TestSourceAndProcessingConfig:
    def test_source_config_returns_a_top_level_block(self) -> None:
        assert "url" in config.source_config("cellosaurus")

    def test_processing_config_returns_a_top_level_block(self) -> None:
        assert config.processing_config("curvecurator")["fit_speed"] == "exhaustive"

    def test_execution_selects_an_executor_and_pins_the_pipeline(self) -> None:
        execution = config.processing_config("execution")
        assert execution["backend"] == "local"
        assert execution["pipeline"] == "drevalpy/curate"
        assert execution["revision"]
        assert {"slurm", "aws_batch", "max_workers", "chunk_cache"}.isdisjoint(execution)

    def test_pubchem_failure_budget_is_configured(self) -> None:
        assert config.processing_config("pubchem")["max_failure_fraction"] == 0.05

    def test_refuses_to_guess_outside_the_repository(self, monkeypatch, tmp_path) -> None:
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="repository root"):
            config.source_config("cellosaurus")
