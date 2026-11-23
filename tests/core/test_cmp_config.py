"""Tests for the shared omics source blocks of config/sources.yaml and their readers."""

from typing import Any

import pytest

from data_curation.core import config, layout
from data_curation.core.errors import CurationError


def _walk(node: Any, path: str = ""):
    """Yield (dotted path, key, value) for every mapping entry in a parsed YAML tree."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield f"{path}.{key}" if path else str(key), key, value
            yield from _walk(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for index, item in enumerate(node):
            yield from _walk(item, f"{path}[{index}]")


class TestPinnedSources:
    def test_no_source_is_left_unpinned(self) -> None:
        sources = config._load(config.SOURCES_PATH)
        unpinned = [path for path, key, value in _walk(sources) if key.endswith("known_hash") and not value]
        assert not unpinned, f"sources without a hash: {unpinned}"

    @pytest.mark.parametrize("kind", layout.CMP_KINDS)
    def test_each_cmp_download_has_a_sha256_pin(self, kind: str) -> None:
        entry = config.shared_source("cell_model_passports", kind)
        assert entry["known_hash"].startswith("sha256:")
        assert len(entry["known_hash"]) == len("sha256:") + 64
        assert entry["url"].startswith("https://")

    @pytest.mark.parametrize("source", ["ccle", "gdsc"])
    def test_each_shared_download_has_a_sha256_pin(self, source: str) -> None:
        for kind in layout.SHARED_SOURCES[source][1]:
            entry = config.shared_source(source, kind)
            for item in entry.get("files") or [entry]:
                assert item["known_hash"].startswith("sha256:")
                assert len(item["known_hash"]) == len("sha256:") + 64
                assert item["url"].startswith("https://")

    def test_only_the_unversioned_proteomics_url_has_no_date(self) -> None:
        """CMP publishes no dated proteomics file; every other CMP URL carries a release date."""
        urls = {kind: config.shared_source("cell_model_passports", kind)["url"] for kind in layout.CMP_KINDS}
        undated = [kind for kind, url in urls.items() if "latest" in url]
        assert undated == ["proteomics"]


class TestOmicsSources:
    @pytest.mark.parametrize(
        ("dataset", "source"),
        [("CCLE", "ccle"), ("CTRPv1", "ccle"), ("CTRPv2", "ccle"), ("GDSC1", "gdsc"), ("GDSC2", "gdsc")],
    )
    def test_expression_and_methylation_come_from_the_screens_own_release(self, dataset: str, source: str) -> None:
        assert config.omics_source(dataset, "transcriptomics") == source
        assert config.omics_source(dataset, "methylation") == source

    @pytest.mark.parametrize("dataset", ["CCLE", "CTRPv1", "CTRPv2", "GDSC1", "GDSC2"])
    @pytest.mark.parametrize("modality", ["mutations", "cnv", "proteomics"])
    def test_the_rest_comes_from_cell_model_passports(self, dataset: str, modality: str) -> None:
        assert config.omics_source(dataset, modality) == "cell_model_passports"

    @pytest.mark.parametrize("dataset", ["PRISM_19Q4", "PRISM_20Q2"])
    def test_prism_takes_ccle_methylation_but_its_own_expression(self, dataset: str) -> None:
        assert config.omics_source(dataset, "methylation") == "ccle"
        assert config.omics_source(dataset, "transcriptomics") is None
        assert "transcriptomics" in config.dataset_config(dataset)

    @pytest.mark.parametrize("dataset", ["PDX_Bruna", "BeatAML2"])
    def test_the_other_datasets_name_no_shared_source(self, dataset: str) -> None:
        assert config.omics_source(dataset, "transcriptomics") is None

    def test_an_unknown_dataset_is_rejected(self) -> None:
        with pytest.raises(CurationError, match="not configured"):
            config.omics_source("NOPE", "cnv")

    def test_every_named_source_is_a_declared_block(self) -> None:
        for dataset in config.dataset_names():
            for source in (config.dataset_config(dataset).get("omics") or {}).values():
                assert source in config.SHARED_SOURCES
                assert config.source_config(source)

    @pytest.mark.parametrize("modality", ["mutations", "cnv", "proteomics"])
    def test_cmp_omics_modalities_cover_exactly_the_cmp_datasets(self, modality: str) -> None:
        assert config.dataset_names(modality) == ["CCLE", "CTRPv1", "CTRPv2", "GDSC1", "GDSC2"]

    def test_methylation_covers_the_cmp_datasets_and_prism(self) -> None:
        assert config.dataset_names("methylation") == [
            "CCLE",
            "CTRPv1",
            "CTRPv2",
            "GDSC1",
            "GDSC2",
            "PRISM_19Q4",
            "PRISM_20Q2",
        ]

    def test_transcriptomics_adds_the_datasets_with_their_own_download(self) -> None:
        names = config.dataset_names("transcriptomics")
        assert {"CCLE", "GDSC2", "PDX_Bruna", "BeatAML2", "PRISM_19Q4", "PRISM_20Q2"} <= set(names)

    def test_a_shared_modality_has_no_per_dataset_block(self) -> None:
        for dataset in config.dataset_names():
            block = config.dataset_config(dataset)
            assert not set(block) & set(block.get("omics") or {}), dataset
