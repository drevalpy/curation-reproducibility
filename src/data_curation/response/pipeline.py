"""Run the drevalpy/curate Nextflow pipeline on one dataset's well table.

``fit_curves`` hands every curve of a dataset to the pipeline, which fits it chunk by chunk on
any Nextflow executor and writes the same AnnData ``drevalpy.curation.curate()`` would. This
module builds the ``nextflow run`` command for a dataset's well table and runs it.

Two settings come from the ``execution`` block of ``config/processing.yaml``:

- the **executor**, ``execution.backend`` (``local``, ``slurm`` or ``aws_batch``, overridable
  with ``DATA_CURATION_BACKEND``), which selects ``config/nextflow/<backend>.config``. It decides
  where tasks run, never what they fit, so it is not a DVC param.
- the **pipeline**, ``execution.pipeline`` at ``execution.revision``, run as
  ``nextflow run <pipeline> -r <revision>`` (DVC params: the revision pins the container and with
  it the fitting libraries).

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


def pipeline_source(execution: Mapping[str, Any]) -> tuple[str, str]:
    """The pipeline to run and the revision it is pinned to.

    Args:
        execution: The ``execution`` block of ``config/processing.yaml``.

    Returns:
        ``execution.pipeline`` (default ``drevalpy/curate``) and ``execution.revision``.

    Raises:
        CurationError: If no revision is set.
    """
    revision = execution.get("revision")
    if not revision:
        msg = "execution.revision is not set; pin the pipeline to a commit of drevalpy/curate"
        raise CurationError(msg)
    return execution.get("pipeline", DEFAULT_PIPELINE), str(revision)


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
    revision: str,
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
        pipeline: Pipeline name, e.g. ``drevalpy/curate``.
        revision: Commit to pass with ``-r``; ``-latest`` fetches it if the local clone lacks it.
        config: Executor config passed with ``-c``.
        wells: The well table (``--input``).
        normalize: Whether the pipeline normalizes the wells before fitting (``--normalize``).
        metadata: Well-table columns kept per dose series (``--metadata``).
        outdir: The pipeline's output directory (``--outdir``).
        options: The fit parameters.

    Returns:
        The argv, ending in ``-resume``.
    """
    normalize_args = ["--normalize"] if normalize else []
    metadata_args = ["--metadata", ",".join(metadata)] if metadata else []
    return [
        nextflow,
        "run",
        pipeline,
        "-r",
        revision,
        "-latest",
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
        environ: Environment for the backend override and for Nextflow; defaults to ``os.environ``.

    Returns:
        The written curve file, :func:`data_curation.core.layout.curves`.

    Raises:
        CurationError: If ``nextflow`` is not on the ``PATH``, the executor config or the revision
            is missing, or the pipeline fails.
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
    pipeline, revision = pipeline_source(execution)

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

    logger.info("Running %s at %s on %s from %s", pipeline, revision, backend, launch_dir)
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
