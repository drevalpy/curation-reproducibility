"""Run the drevalpy/curate Nextflow pipeline on one dataset's well table.

``fit_curves`` hands every curve of a dataset to the pipeline, which fits it chunk by chunk on
any Nextflow executor and writes the same AnnData ``drevalpy.curation.curate()`` would. This
module builds the ``nextflow run`` command for a dataset's well table and runs it.

Two settings come from the ``execution`` block of ``config/processing.yaml``:

- the **executor**, ``execution.backend`` (``local``, ``slurm`` or ``aws_batch``, overridable
  with ``DATA_CURATION_BACKEND``), which selects ``config/nextflow/<backend>.config``. It decides
  where tasks run, never what they fit, so it is not a DVC param.
- the **pipeline**, ``execution.pipeline`` at ``execution.revision`` (a DVC param: the revision
  pins the container and with it the fitting libraries). ``DATA_CURATION_PIPELINE`` replaces it
  with a local checkout, which Nextflow runs as it is, so ``-r`` is only passed for a remote one.

Every dataset is launched from its own directory (:func:`data_curation.core.layout.nextflow_launch_dir`),
which keeps its Nextflow history and work directory, so ``-resume`` reuses its fitted chunks.
"""

import logging
import os
import shutil
import subprocess  # noqa: S404 - runs the nextflow CLI with arguments built here
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pyarrow.parquet as pq
from upath import UPath

from data_curation.core import layout
from data_curation.core.errors import CurationError
from data_curation.response.prepare.wells import metadata_columns

logger = logging.getLogger(__name__)

BACKEND_ENV_VAR = "DATA_CURATION_BACKEND"
PIPELINE_ENV_VAR = "DATA_CURATION_PIPELINE"
BACKEND_NAMES = ("local", "slurm", "aws_batch")
NEXTFLOW_CONFIG_DIR = UPath("config/nextflow")
DEFAULT_PIPELINE = "drevalpy/curate"


@dataclass(frozen=True)
class FitOptions:
    """The pipeline parameters that decide (or, for ``max_chunk_size``, merely cut up) the fits.

    Attributes:
        min_doses: Series with fewer distinct doses are not fitted on their own.
        fit_speed: CurveCurator fit speed, e.g. ``"exhaustive"``.
        max_chunk_size: Most curves one fitting task gets.
    """

    min_doses: int
    fit_speed: str
    max_chunk_size: int


def backend_name(execution: Mapping[str, Any], environ: Mapping[str, str]) -> str:
    """The executor to run on: ``DATA_CURATION_BACKEND`` if set, else ``execution.backend``.

    Args:
        execution: The ``execution`` block of ``config/processing.yaml``.
        environ: Environment to read the override from.

    Returns:
        One of :data:`BACKEND_NAMES`.

    Raises:
        CurationError: If the name is not one of :data:`BACKEND_NAMES`.
    """
    name = environ.get(BACKEND_ENV_VAR) or execution.get("backend", "local")
    if name not in BACKEND_NAMES:
        msg = f"Unknown execution backend {name!r}; valid names: {', '.join(BACKEND_NAMES)}"
        raise CurationError(msg)
    return name


def pipeline_source(execution: Mapping[str, Any], environ: Mapping[str, str]) -> tuple[str, str | None]:
    """The pipeline to run and the revision to pass with ``-r``.

    Args:
        execution: The ``execution`` block of ``config/processing.yaml``.
        environ: Environment to read ``DATA_CURATION_PIPELINE`` from.

    Returns:
        The pipeline (a remote name, or a local checkout as an absolute path) and its revision,
        which is None for a local checkout: Nextflow runs that as it is.
    """
    source = environ.get(PIPELINE_ENV_VAR) or execution.get("pipeline", DEFAULT_PIPELINE)
    local = UPath(source).expanduser()
    if source.startswith(("/", ".", "~")) or local.is_dir():
        return str(local.absolute()), None
    return source, execution.get("revision")


def well_metadata(wells: UPath) -> list[str]:
    """The metadata columns of a well table, read from its Parquet schema only.

    Args:
        wells: The dataset's well table (Parquet).

    Returns:
        Every column curate does not read itself, in table order.
    """
    return metadata_columns(pq.read_schema(str(wells)).empty_table().to_pandas())


def nextflow_command(
    nextflow: str,
    pipeline: str,
    revision: str | None,
    *,
    config: UPath,
    wells: UPath,
    normalize: bool,
    metadata: list[str],
    outdir: UPath,
    options: FitOptions,
) -> list[str]:
    """The ``nextflow run`` command for one dataset.

    Args:
        nextflow: Path of the ``nextflow`` executable.
        pipeline: Remote pipeline name or local checkout.
        revision: Revision to pass with ``-r``, or None to pass none.
        config: Executor config passed with ``-c``.
        wells: The well table (``--input``).
        normalize: Whether the pipeline normalizes the wells before fitting (``--normalize``).
        metadata: Well-table columns kept per dose series (``--metadata``).
        outdir: The pipeline's output directory (``--outdir``).
        options: The fit parameters.

    Returns:
        The argv, ending in ``-resume``.
    """
    revision_args = ["-r", revision] if revision else []
    normalize_args = ["--normalize"] if normalize else []
    metadata_args = ["--metadata", ",".join(metadata)] if metadata else []
    return [
        nextflow,
        "run",
        pipeline,
        *revision_args,
        "-c",
        str(config),
        "-ansi-log",
        "false",
        "--input",
        str(wells),
        *normalize_args,
        *metadata_args,
        "--outdir",
        str(outdir),
        "--replicate_fits",
        "--min_doses",
        str(options.min_doses),
        "--fit_speed",
        options.fit_speed,
        "--max_chunk_size",
        str(options.max_chunk_size),
        "-resume",
    ]


def fit_dataset(
    dataset: str,
    *,
    normalize: bool,
    options: FitOptions,
    execution: Mapping[str, Any],
    environ: Mapping[str, str] | None = None,
) -> UPath:
    """Fit every curve of a dataset with the pipeline and copy the result to its layout path.

    Args:
        dataset: Dataset name.
        normalize: Whether the pipeline normalizes the wells before fitting.
        options: The fit parameters.
        execution: The ``execution`` block of ``config/processing.yaml``.
        environ: Environment for the overrides and for Nextflow; defaults to ``os.environ``.

    Returns:
        The written curve file, :func:`data_curation.core.layout.curves`.

    Raises:
        CurationError: If ``nextflow`` is not on the ``PATH``, the executor config is missing or
            the pipeline fails.
    """
    env = dict(os.environ if environ is None else environ)
    nextflow = shutil.which("nextflow", path=env.get("PATH"))
    if nextflow is None:
        msg = "nextflow is not on the PATH; install it (https://www.nextflow.io) to fit curves"
        raise CurationError(msg)
    backend = backend_name(execution, env)
    config = (NEXTFLOW_CONFIG_DIR / f"{backend}.config").absolute()
    if not config.is_file():
        msg = f"No Nextflow config {config} for execution backend {backend!r}"
        raise CurationError(msg)
    pipeline, revision = pipeline_source(execution, env)

    launch_dir = layout.nextflow_launch_dir(dataset).absolute()
    launch_dir.mkdir(parents=True, exist_ok=True)
    wells = layout.wells(dataset).absolute()
    outdir = launch_dir / "results"
    cmd = nextflow_command(
        nextflow,
        pipeline,
        revision,
        config=config,
        wells=wells,
        normalize=normalize,
        metadata=well_metadata(wells),
        outdir=outdir,
        options=options,
    )

    logger.info("Running %s on %s from %s", pipeline, backend, launch_dir)
    _run(cmd, launch_dir, env)

    produced = outdir / "curves.h5ad"
    if not produced.is_file():
        msg = f"Nextflow finished but wrote no {produced}"
        raise CurationError(msg)
    output = layout.curves(dataset)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(produced), str(output))
    return output


def _run(cmd: list[str], cwd: UPath, env: Mapping[str, str]) -> None:
    """Run *cmd* in *cwd*, streaming its output to the log line by line."""
    with subprocess.Popen(  # noqa: S603 - argv built by nextflow_command, no shell
        cmd, cwd=str(cwd), env=dict(env), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as process:
        assert process.stdout is not None  # noqa: S101 - stdout=PIPE guarantees it
        for line in process.stdout:
            logger.info("nextflow: %s", line.rstrip())
    if process.returncode != 0:
        msg = f"Nextflow exited with code {process.returncode}; see {cwd / '.nextflow.log'}"
        raise CurationError(msg)
