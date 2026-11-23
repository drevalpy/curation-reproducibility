"""Tests for the prepare_viability stage's format dispatch."""

import pandas as pd
import pytest
from upath import UPath

from data_curation.core.errors import CurationError
from data_curation.response import prepare_viability
from data_curation.response.prepare.wells import SeriesSpec, Wells
from data_curation.response.prepare_viability import PREPARERS, prepare


class TestPreparerRegistry:
    def test_every_configured_format_has_a_preparer(self) -> None:
        assert set(PREPARERS) == {"ctrpv1", "ctrpv2", "ccle", "bruna", "beataml", "prism", "standard"}


class TestPrepare:
    def test_the_configured_format_is_dispatched(self, tmp_dir: UPath, monkeypatch) -> None:
        expected = Wells(pd.DataFrame({"sample": ["a"], "drug": ["b"]}), SeriesSpec(sample="sample", drug="drug"))
        monkeypatch.setitem(PREPARERS, "ccle", lambda _raw: expected)
        assert prepare("CCLE", {"format": "ccle"}, tmp_dir) is expected

    def test_gdsc_receives_its_control_tags(self, tmp_dir: UPath, monkeypatch) -> None:
        """GDSC's control tags differ per release and come from the params block."""
        seen: list[tuple[str, str]] = []

        def _record(_raw: UPath, *, negative_control_tag: str, positive_control_tag: str) -> Wells:
            seen.append((negative_control_tag, positive_control_tag))
            return Wells(pd.DataFrame(), SeriesSpec(sample="s", drug="d"))

        monkeypatch.setattr(prepare_viability, "prepare_gdsc", _record)
        params = {"format": "gdsc", "negative_control_tag": "NC-0", "positive_control_tag": "B"}
        prepare("GDSC1", params, tmp_dir)
        assert seen == [("NC-0", "B")]

    def test_an_unknown_format_is_rejected(self, tmp_dir: UPath) -> None:
        with pytest.raises(CurationError, match="Unknown response format 'nonesuch'"):
            prepare("Whatever", {"format": "nonesuch"}, tmp_dir)
