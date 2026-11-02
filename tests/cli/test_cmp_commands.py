"""Tests for the commands that run the dataset-free Cell Model Passports stages."""

from typer.testing import CliRunner

from data_curation import cli

runner = CliRunner()


def test_download_cmp_calls_the_shared_download_once(monkeypatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr("data_curation.omics.download.download_cmp", lambda: calls.append(1))

    result = runner.invoke(cli.app, ["omics", "download-cmp"])

    assert result.exit_code == 0, result.output
    assert calls == [1]


def test_process_cmp_meta_calls_the_shared_stage(monkeypatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr("data_curation.cell_lines.process_sample_meta.main_cmp", lambda: calls.append(1))

    result = runner.invoke(cli.app, ["cell-lines", "process-cmp-meta"])

    assert result.exit_code == 0, result.output
    assert calls == [1]


def test_the_per_dataset_model_list_command_is_gone() -> None:
    assert runner.invoke(cli.app, ["cell-lines", "download-model-list"]).exit_code != 0


def test_a_cmp_dataset_is_still_accepted_for_its_omics_process_commands(monkeypatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr("data_curation.omics.process_cnv.main", seen.append)

    result = runner.invoke(cli.app, ["omics", "process", "cnv", "CCLE"])

    assert result.exit_code == 0, result.output
    assert seen == ["CCLE"]
