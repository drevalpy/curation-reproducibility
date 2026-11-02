"""Tests for the Cell Model Passports block of config/sources.yaml and its readers."""

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
        entry = config.cmp_source(kind)
        assert entry["known_hash"].startswith("sha256:")
        assert len(entry["known_hash"]) == len("sha256:") + 64
        assert entry["url"].startswith("https://")

    def test_only_the_unversioned_proteomics_url_has_no_date(self) -> None:
        """CMP publishes no dated proteomics file; every other CMP URL carries a release date."""
        undated = [kind for kind in layout.CMP_KINDS if "latest" in config.cmp_source(kind)["url"]]
        assert undated == ["proteomics"]


class TestCmpDatasets:
    @pytest.mark.parametrize("dataset", ["CCLE", "CTRPv1", "CTRPv2", "GDSC1", "GDSC2"])
    def test_the_cmp_datasets_are_flagged(self, dataset: str) -> None:
        assert config.is_cmp_dataset(dataset)

    @pytest.mark.parametrize("dataset", ["PDX_Bruna", "BeatAML2", "PRISM_19Q4", "PRISM_20Q2"])
    def test_the_other_datasets_are_not(self, dataset: str) -> None:
        assert not config.is_cmp_dataset(dataset)

    def test_an_unknown_dataset_is_rejected(self) -> None:
        with pytest.raises(CurationError, match="not configured"):
            config.is_cmp_dataset("NOPE")

    @pytest.mark.parametrize("modality", ["mutations", "cnv", "proteomics"])
    def test_cmp_omics_modalities_cover_exactly_the_cmp_datasets(self, modality: str) -> None:
        assert config.dataset_names(modality) == ["CCLE", "CTRPv1", "CTRPv2", "GDSC1", "GDSC2"]

    def test_transcriptomics_adds_the_datasets_with_their_own_download(self) -> None:
        names = config.dataset_names("transcriptomics")
        assert {"CCLE", "GDSC2", "PDX_Bruna", "BeatAML2", "PRISM_19Q4", "PRISM_20Q2"} <= set(names)

    def test_cmp_datasets_carry_no_per_dataset_omics_blocks(self) -> None:
        for dataset in ("CCLE", "CTRPv1", "CTRPv2", "GDSC1", "GDSC2"):
            block = config.dataset_config(dataset)
            assert not set(block) & config.CMP_MODALITIES
