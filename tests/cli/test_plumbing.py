"""Tests for the CLI's logging and progress plumbing."""

import logging

import pytest

from data_curation.cli import _logging, _progress


class TestResolveLevel:
    def test_defaults_to_info(self) -> None:
        assert _logging.resolve_level(None, 0, 0) == logging.INFO

    def test_verbose_steps_down_to_debug(self) -> None:
        assert _logging.resolve_level(None, 1, 0) == logging.DEBUG

    def test_quiet_steps_up_to_warning(self) -> None:
        assert _logging.resolve_level(None, 0, 1) == logging.WARNING

    def test_an_explicit_level_wins_over_the_counters(self) -> None:
        assert _logging.resolve_level("ERROR", 3, 0) == logging.ERROR

    def test_the_level_name_is_case_insensitive(self) -> None:
        assert _logging.resolve_level("debug", 0, 0) == logging.DEBUG

    @pytest.mark.parametrize("count", [5, 12])
    def test_excess_flags_clamp_instead_of_failing(self, count: int) -> None:
        assert _logging.resolve_level(None, count, 0) == logging.DEBUG
        assert _logging.resolve_level(None, 0, count) == logging.CRITICAL

    def test_an_unknown_level_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown log level"):
            _logging.resolve_level("LOUD", 0, 0)


class TestTrack:
    def test_yields_every_item_unchanged(self) -> None:
        assert list(_progress.track([1, 2, 3], "counting")) == [1, 2, 3]

    def test_falls_back_to_log_lines_when_not_interactive(self, monkeypatch, caplog) -> None:
        """Under dvc a live bar would redraw itself into the log; a summary must not."""
        monkeypatch.setattr(_progress, "is_interactive", lambda: False)
        monkeypatch.setattr(_progress, "_LOG_EVERY", 2)

        with caplog.at_level(logging.INFO):
            list(_progress.track(range(4), "mapping"))

        messages = [record.message for record in caplog.records]
        assert any("mapping: 2/4" in message for message in messages)
        assert any("done" in message for message in messages)

    def test_uses_a_live_bar_when_interactive(self, monkeypatch) -> None:
        monkeypatch.setattr(_progress, "is_interactive", lambda: True)
        tracked: list[str] = []
        monkeypatch.setattr(
            _progress,
            "_tracked",
            lambda items, description, total, console: (tracked.append(description), iter(items))[1],
        )

        assert list(_progress.track([1, 2], "mapping")) == [1, 2]
        assert tracked == ["mapping"]

    def test_an_unsized_iterable_needs_no_total(self, monkeypatch, caplog) -> None:
        monkeypatch.setattr(_progress, "is_interactive", lambda: False)

        with caplog.at_level(logging.INFO):
            assert list(_progress.track((n for n in range(3)), "streaming")) == [0, 1, 2]

        assert any("streaming: 3 done" in record.message for record in caplog.records)
