"""Tests for the CLI's command tree and argument handling."""

import pytest
from typer.testing import CliRunner

from data_curation import cli
from data_curation.core.errors import CurationError

runner = CliRunner()

# Every leaf command that dispatches through {module}.main, as (argv, module).
LEAF_COMMANDS = [
    (["cell-lines", "process-meta", "GDSC2"], "data_curation.cell_lines.process_sample_meta"),
    (["omics", "process", "cnv", "GDSC2"], "data_curation.omics.process_cnv"),
    (["omics", "process", "mutations", "GDSC2"], "data_curation.omics.process_mutations"),
    (["omics", "process", "proteomics", "GDSC2"], "data_curation.omics.process_proteomics"),
    (["omics", "process", "transcriptomics", "GDSC2"], "data_curation.omics.process_transcriptomics"),
    (["response", "prepare-viability", "GDSC2"], "data_curation.response.prepare_viability"),
    (["response", "fit-curves", "GDSC2"], "data_curation.response.fit_curves"),
    (["response", "harmonize", "GDSC2"], "data_curation.response.harmonize_response"),
    (["drugs", "download-ontology"], "data_curation.drugs.download_drug_ontology"),
    (["drugs", "map", "GDSC2"], "data_curation.drugs.map_drugs"),
    (["drugs", "resolve-structures"], "data_curation.drugs.resolve_structures"),
    (["drugs", "annotate"], "data_curation.drugs.annotate_drugs"),
    (["assembly", "check-drug-structures"], "data_curation.assembly.check_drug_structures"),
    (["assembly", "mudata", "GDSC2"], "data_curation.assembly.assemble_mudata"),
    (["assembly", "subsample-toy"], "data_curation.assembly.subsample_toy"),
    (["assembly", "record-toy-gene-lists"], "data_curation.assembly.record_toy_gene_lists"),
]

GROUPS = ["cell-lines", "omics", "response", "drugs", "assembly"]


class TestHelp:
    def test_the_root_lists_every_group(self) -> None:
        result = runner.invoke(cli.app, ["--help"])
        assert result.exit_code == 0
        for group in GROUPS:
            assert group in result.output

    @pytest.mark.parametrize("group", GROUPS)
    def test_each_group_help_succeeds(self, group: str) -> None:
        assert runner.invoke(cli.app, [group, "--help"]).exit_code == 0

    def test_no_arguments_shows_help_rather_than_failing_silently(self) -> None:
        assert "Usage" in runner.invoke(cli.app, []).output


class TestDispatch:
    @pytest.mark.parametrize(("argv", "module"), LEAF_COMMANDS)
    def test_each_command_calls_its_stage(self, monkeypatch, argv: list[str], module: str) -> None:
        """Guards the wiring: a command pointing at the wrong module is caught here."""
        calls: list[tuple] = []
        monkeypatch.setattr(f"{module}.main", lambda *args: calls.append(args))

        result = runner.invoke(cli.app, argv)

        assert result.exit_code == 0, result.output
        assert len(calls) == 1

    def test_download_cellosaurus_calls_its_stage(self, monkeypatch, tmp_path) -> None:
        """download-cellosaurus dispatches to download_cellosaurus(), not main()."""
        calls: list[int] = []

        monkeypatch.setattr(
            "data_curation.cell_lines.download_cellosaurus.download_cellosaurus",
            lambda path: calls.append(1),
        )
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(cli.app, ["cell-lines", "download-cellosaurus"])

        assert result.exit_code == 0, result.output
        assert len(calls) == 1

    def test_a_per_dataset_command_passes_the_dataset_through(self, monkeypatch) -> None:
        seen: list[str] = []
        monkeypatch.setattr("data_curation.response.fit_curves.main", seen.append)

        runner.invoke(cli.app, ["response", "fit-curves", "CTRPv2"])

        assert seen == ["CTRPv2"]

    def test_omics_download_passes_the_modality_through(self, monkeypatch, tmp_path) -> None:
        calls: list[tuple[str, str]] = []
        monkeypatch.setattr(
            "data_curation.omics.download.download_omics_data",
            lambda dataset, key, output_dir: calls.append((dataset, key)),
        )
        monkeypatch.chdir(tmp_path)

        runner.invoke(cli.app, ["omics", "download", "transcriptomics", "BeatAML2"])

        assert calls == [("BeatAML2", "transcriptomics")]


class TestValidation:
    def test_an_unknown_dataset_is_rejected_before_the_stage_runs(self, monkeypatch) -> None:
        ran: list[str] = []
        monkeypatch.setattr("data_curation.response.fit_curves.main", ran.append)

        result = runner.invoke(cli.app, ["response", "fit-curves", "NOPE"])

        assert not ran, "the stage must not run for an unknown dataset"
        assert result.exit_code != 0
        assert "NOPE" in result.output

    def test_the_rejection_lists_the_valid_datasets(self) -> None:
        """An operator should not have to open config/sources.yaml to find the right name."""
        result = runner.invoke(cli.app, ["response", "fit-curves", "GDSC3"])

        assert "GDSC2" in result.output

    def test_an_unknown_modality_is_rejected(self) -> None:
        result = runner.invoke(cli.app, ["omics", "process", "metabolomics", "GDSC2"])

        assert result.exit_code != 0

    def test_a_dataset_without_the_modality_is_rejected(self) -> None:
        """PDX_Bruna has transcriptomics but no CNV, so the CNV stage must refuse it."""
        result = runner.invoke(cli.app, ["omics", "process", "cnv", "PDX_Bruna"])

        assert result.exit_code != 0
        assert "PDX_Bruna" in result.output


class TestErrorRendering:
    def test_an_expected_failure_becomes_a_short_message(self, monkeypatch) -> None:
        def fail(_dataset: str) -> None:
            raise FileNotFoundError("no file matching '*rsem_tpm*'")

        monkeypatch.setattr("data_curation.response.fit_curves.main", fail)

        result = runner.invoke(cli.app, ["response", "fit-curves", "GDSC2"])

        assert result.exit_code == 1
        assert "rsem_tpm" in result.output
        assert "Traceback" not in result.output

    def test_a_curation_error_becomes_a_short_message(self, monkeypatch) -> None:
        def fail(_dataset: str) -> None:
            raise CurationError("target coverage collapsed")

        monkeypatch.setattr("data_curation.response.fit_curves.main", fail)

        result = runner.invoke(cli.app, ["response", "fit-curves", "GDSC2"])

        assert result.exit_code == 1
        assert "target coverage collapsed" in result.output
        assert "CurationError" in result.output
        assert "Traceback" not in result.output

    def test_traceback_opts_back_into_the_full_trace(self, monkeypatch) -> None:
        def fail(_dataset: str) -> None:
            raise CurationError("target coverage collapsed")

        monkeypatch.setattr("data_curation.response.fit_curves.main", fail)

        result = runner.invoke(cli.app, ["--traceback", "response", "fit-curves", "GDSC2"])

        assert isinstance(result.exception, CurationError)

    @pytest.mark.parametrize("error", [ValueError("bug"), RuntimeError("bug"), KeyError("bug"), TypeError("bug")])
    def test_other_exceptions_keep_their_traceback(self, monkeypatch, error: Exception) -> None:
        """Only FileNotFoundError and CurationError are expected; anything else is a bug."""

        def fail(_dataset: str) -> None:
            raise error

        monkeypatch.setattr("data_curation.response.fit_curves.main", fail)

        result = runner.invoke(cli.app, ["response", "fit-curves", "GDSC2"])

        assert result.exception is error


class TestClickControlFlow:
    """Click signals --help and exit codes with exceptions of its own.

    They are re-raised before the expected-error handling, so ordinary success
    never looks like a crash.
    """

    def test_group_help_still_succeeds(self) -> None:
        assert runner.invoke(cli.app, ["drugs", "--help"]).exit_code == 0

    def test_a_successful_command_exits_zero(self, monkeypatch) -> None:
        monkeypatch.setattr("data_curation.response.fit_curves.main", lambda _dataset: None)

        assert runner.invoke(cli.app, ["response", "fit-curves", "GDSC2"]).exit_code == 0
