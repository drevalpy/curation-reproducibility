"""Tests for running the drevalpy/curate Nextflow pipeline from fit_curves."""

import os
import shutil
import stat
import subprocess  # noqa: S404 - probes for a running Docker daemon
import textwrap

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from upath import UPath

from data_curation.core import config
from data_curation.core.errors import CurationError
from data_curation.response import pipeline
from data_curation.response.pipeline import (
    BACKEND_NAMES,
    NEXTFLOW_CONFIG_DIR,
    FitOptions,
    backend_name,
    fit_dataset,
    nextflow_command,
    pipeline_source,
    well_metadata,
)

OPTIONS = FitOptions(min_doses=4, fit_speed="fast", max_chunk_size=7)
EXECUTION = {"backend": "local", "pipeline": "drevalpy/curate", "revision": "abc123"}
REPO_CONFIG_DIR = NEXTFLOW_CONFIG_DIR.absolute()


def _wells(n_pairs: int = 1, *, metadata: bool = True) -> pd.DataFrame:
    """Wells of *n_pairs* sigmoidal curves, two plates each."""
    doses = np.logspace(-3, 1, 6)
    rows = []
    for pair in range(n_pairs):
        for plate in ("P1", "P2"):
            for dose in doses:
                intensity = 1 / (1 + dose / 10 ** (-1 - pair * 0.5))
                row = {
                    "sample": f"CL_{pair}",
                    "drug": "DrugX",
                    "concentration": dose,
                    "intensity": intensity,
                    "replicate": plate,
                }
                if metadata:
                    row["plate_date"] = "2020-01-01"
                rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def repo(tmp_dir: UPath, monkeypatch) -> UPath:
    """A repository root in ``tmp_dir`` with the shipped Nextflow configs and one well table."""
    shutil.copytree(str(REPO_CONFIG_DIR), str(tmp_dir / "config" / "nextflow"))
    wells = tmp_dir / "data" / "04_response" / "wells"
    wells.mkdir(parents=True)
    _wells().to_parquet(str(wells / "TOY.parquet"))
    monkeypatch.chdir(tmp_dir)
    return tmp_dir


@pytest.fixture
def pinned_execution() -> dict:
    """The real ``execution`` block, read before ``repo`` leaves the repository root."""
    return config.processing_config("execution")


def _fake_nextflow(directory: UPath, body: str) -> str:
    """Write an executable ``nextflow`` shell script into *directory* and return a PATH for it."""
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "nextflow"
    script.write_text("#!/bin/sh\n" + textwrap.dedent(body))
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return f"{directory}:/usr/bin:/bin"


class TestBackendName:
    def test_the_config_picks_the_backend(self) -> None:
        assert backend_name({"backend": "slurm"}, {}) == "slurm"

    def test_local_is_the_default(self) -> None:
        assert backend_name({}, {}) == "local"

    def test_the_environment_overrides_the_config(self) -> None:
        assert backend_name({"backend": "local"}, {"DATA_CURATION_BACKEND": "aws_batch"}) == "aws_batch"

    def test_an_empty_override_is_ignored(self) -> None:
        assert backend_name({"backend": "slurm"}, {"DATA_CURATION_BACKEND": ""}) == "slurm"

    def test_an_unknown_backend_lists_the_valid_names(self) -> None:
        with pytest.raises(CurationError, match="local, slurm, aws_batch"):
            backend_name({"backend": "kubernetes"}, {})

    def test_every_backend_has_a_shipped_config(self) -> None:
        assert {path.stem for path in REPO_CONFIG_DIR.glob("*.config")} == set(BACKEND_NAMES)


class TestPipelineSource:
    def test_the_pipeline_is_pinned_to_its_revision(self) -> None:
        assert pipeline_source(EXECUTION) == ("drevalpy/curate", "abc123")

    def test_the_default_pipeline_is_drevalpy_curate(self) -> None:
        assert pipeline_source({"revision": "abc123"}) == ("drevalpy/curate", "abc123")

    def test_a_missing_revision_is_refused(self) -> None:
        with pytest.raises(CurationError, match="execution.revision"):
            pipeline_source({"pipeline": "drevalpy/curate"})


class TestWellMetadata:
    def test_the_metadata_columns_come_from_the_schema_in_order(self, tmp_dir: UPath) -> None:
        _wells().assign(barcode="B1").to_parquet(str(tmp_dir / "w.parquet"))

        assert well_metadata(tmp_dir / "w.parquet") == ["plate_date", "barcode"]

    def test_a_table_without_metadata_has_none(self, tmp_dir: UPath) -> None:
        _wells(metadata=False).to_parquet(str(tmp_dir / "w.parquet"))

        assert well_metadata(tmp_dir / "w.parquet") == []


class TestNextflowCommand:
    def _command(
        self, revision: str = "abc123", *, normalize: bool = False, metadata: list[str] | None = None
    ) -> list[str]:
        return nextflow_command(
            "/bin/nextflow",
            "drevalpy/curate",
            revision,
            config=UPath("/c/local.config"),
            wells=UPath("/d/wells.parquet"),
            normalize=normalize,
            metadata=metadata or [],
            outdir=UPath("/l/results"),
            options=OPTIONS,
        )

    def test_it_passes_the_config_inputs_and_fit_options_and_resumes(self) -> None:
        cmd = self._command("abc123")

        assert cmd[:6] == ["/bin/nextflow", "run", "drevalpy/curate", "-r", "abc123", "-latest"]
        assert cmd[cmd.index("-c") + 1] == "/c/local.config"
        assert cmd[cmd.index("--input") + 1] == "/d/wells.parquet"
        assert cmd[cmd.index("--outdir") + 1] == "/l/results"
        assert "--replicate_fits" in cmd
        assert cmd[cmd.index("--min_doses") + 1] == "4"
        assert cmd[cmd.index("--fit_speed") + 1] == "fast"
        assert cmd[cmd.index("--max_chunk_size") + 1] == "7"
        assert cmd[-1] == "-resume"

    def test_normalize_and_metadata_are_passed_only_when_set(self) -> None:
        plain = self._command()
        assert "--normalize" not in plain
        assert "--metadata" not in plain

        cmd = self._command(normalize=True, metadata=["plate_date", "barcode"])
        assert "--normalize" in cmd
        assert cmd[cmd.index("--metadata") + 1] == "plate_date,barcode"


class TestFitDataset:
    def _environ(self, repo: UPath, body: str, **extra: str) -> dict[str, str]:
        return {"PATH": _fake_nextflow(repo / "bin", body), **extra}

    def test_it_runs_from_the_launch_dir_and_copies_the_curves(self, repo: UPath) -> None:
        environ = self._environ(
            repo,
            """
            echo "$PWD" > launched_from
            echo "$@" > argv
            out=$(echo "$@" | sed 's/.*--outdir \\([^ ]*\\).*/\\1/')
            mkdir -p "$out" && echo fitted > "$out/curves.h5ad"
            """,
        )

        output = fit_dataset("TOY", normalize=False, options=OPTIONS, execution=EXECUTION, environ=environ)

        launch = (repo / "data" / "cache" / "nextflow" / "TOY").absolute()
        assert output == UPath("data/04_response/curves/TOY.h5ad")
        assert output.read_text() == "fitted\n"
        assert UPath((launch / "launched_from").read_text().strip()).resolve() == launch.resolve()
        argv = (launch / "argv").read_text().split()
        assert argv[:3] == ["run", "drevalpy/curate", "-r"]
        assert argv[argv.index("-c") + 1] == str((repo / "config" / "nextflow" / "local.config").absolute())
        assert argv[argv.index("--input") + 1] == str(
            (repo / "data" / "04_response" / "wells" / "TOY.parquet").absolute()
        )
        assert argv[argv.index("--metadata") + 1] == "plate_date"

    def test_the_backend_override_selects_its_config(self, repo: UPath) -> None:
        environ = self._environ(repo, 'echo "$@" > argv; exit 1\n', DATA_CURATION_BACKEND="slurm")

        with pytest.raises(CurationError):
            fit_dataset("TOY", normalize=False, options=OPTIONS, execution=EXECUTION, environ=environ)

        argv = (repo / "data" / "cache" / "nextflow" / "TOY" / "argv").read_text().split()
        assert argv[argv.index("-c") + 1].endswith("config/nextflow/slurm.config")

    def test_its_output_is_streamed_to_the_log(self, repo: UPath, caplog) -> None:
        environ = self._environ(repo, "echo 'N E X T F L O W'; exit 3\n")

        with caplog.at_level("INFO", logger=pipeline.__name__), pytest.raises(CurationError, match="code 3"):
            fit_dataset("TOY", normalize=False, options=OPTIONS, execution=EXECUTION, environ=environ)

        assert "nextflow: N E X T F L O W" in caplog.text

    def test_a_run_without_curves_fails(self, repo: UPath) -> None:
        environ = self._environ(repo, "exit 0\n")

        with pytest.raises(CurationError, match="wrote no"):
            fit_dataset("TOY", normalize=False, options=OPTIONS, execution=EXECUTION, environ=environ)

    def test_a_missing_nextflow_is_a_clear_error(self, repo: UPath) -> None:
        with pytest.raises(CurationError, match="nextflow is not on the PATH"):
            fit_dataset("TOY", normalize=False, options=OPTIONS, execution=EXECUTION, environ={"PATH": str(repo)})


def _docker_is_running() -> bool:
    docker = shutil.which("docker")
    if docker is None:
        return False
    probe = subprocess.run([docker, "info"], capture_output=True, check=False)  # noqa: S603
    return probe.returncode == 0


@pytest.mark.slow
@pytest.mark.skipif(
    shutil.which("nextflow") is None or not _docker_is_running(),
    reason="needs nextflow and a running Docker daemon",
)
def test_the_pinned_pipeline_fits_a_tiny_screen(pinned_execution: dict, repo: UPath) -> None:
    _wells(n_pairs=2).to_parquet(str(repo / "data" / "04_response" / "wells" / "TOY.parquet"))
    environ = {**os.environ, "DATA_CURATION_BACKEND": "local"}

    output = fit_dataset(
        "TOY",
        normalize=False,
        options=FitOptions(min_doses=4, fit_speed="fast", max_chunk_size=1),
        execution=pinned_execution,
        environ=environ,
    )

    adata = ad.read_h5ad(str(output))
    assert sorted(adata.obs_names) == ["CL_0", "CL_1"]
    assert list(adata.var_names) == ["DrugX"]
    assert "replicate_fits" in adata.uns
