"""Tests for the prepare_viability stage's format dispatch."""

import pandas as pd
import pytest
from upath import UPath

from data_curation.core.errors import CurationError
from data_curation.response.prepare.wells import SeriesSpec, Wells
from data_curation.response.prepare_viability import PREPARERS, prepare


class TestPreparerRegistry:
    def test_every_configured_format_has_a_preparer(self) -> None:
        assert set(PREPARERS) == {"gdsc", "ctrpv1", "ctrpv2", "ccle", "bruna", "beataml", "prism", "standard"}


class TestPrepare:
    def test_the_configured_format_is_dispatched(self, tmp_dir: UPath, monkeypatch) -> None:
        expected = Wells(pd.DataFrame({"sample": ["a"], "drug": ["b"]}), SeriesSpec(sample="sample", drug="drug"))
        monkeypatch.setitem(PREPARERS, "gdsc", lambda _raw, _params: expected)
        assert prepare("GDSC2", {"format": "gdsc"}, tmp_dir) is expected

    def test_the_params_block_reaches_the_preparer(self, tmp_dir: UPath, monkeypatch) -> None:
        """GDSC's control tags differ per release and arrive this way."""
        seen: list[dict] = []

        def _record(_raw: UPath, params: dict) -> Wells:
            seen.append(params)
            return Wells(pd.DataFrame(), SeriesSpec(sample="s", drug="d"))

        monkeypatch.setitem(PREPARERS, "gdsc", _record)
        prepare("GDSC1", {"format": "gdsc", "negative_control_tag": "NC-0"}, tmp_dir)
        assert seen[0]["negative_control_tag"] == "NC-0"

    def test_an_unknown_format_is_rejected(self, tmp_dir: UPath) -> None:
        with pytest.raises(CurationError, match="Unknown response format 'nonesuch'"):
            prepare("Whatever", {"format": "nonesuch"}, tmp_dir)
