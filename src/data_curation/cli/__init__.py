"""The ``data-curation`` command line interface.

One entry point for the whole pipeline. The command tree mirrors the domain
packages - ``data-curation response fit-curves GDSC2`` runs
``data_curation.response.fit_curves`` - so the hierarchy is discoverable by
tabbing through ``--help`` instead of by reading ``dvc.yaml``.

The whole command surface lives here rather than as a ``cli.py`` inside each
domain: one module per group (:mod:`~data_curation.cli.response`,
:mod:`~data_curation.cli.omics`, ...) beside the shared plumbing, which is
underscore-prefixed. Keeping it in one place means the domain packages carry no
CLI code and so do not depend on this layer at all, and the whole tree can be
read without opening five directories.

``dvc.yaml`` invokes these commands too, so there is exactly one way to run a
stage. That is why no module here appears in a stage's ``deps``: this layer
parses arguments and configures logging, and cannot change what a stage
computes, so listing it would make an unrelated logging tweak re-run the ~9 h
curve fit.

Every leaf command imports its implementation *inside* the function body. That
deferral is load-bearing: importing them at registration time would drag
``anndata``, ``mudata``, ``pubchempy`` and ``drevalpy`` (hence torch) into
``data-curation --help``.
"""

from importlib.metadata import PackageNotFoundError, version
from typing import Annotated, Any

import click
import typer
from rich.panel import Panel
from typer.core import TyperGroup

from data_curation.cli import _logging
from data_curation.cli._console import error_console
from data_curation.core.errors import CurationError

# Failures the pipeline raises on purpose: a missing input file, or a
# CurationError (unconfigured dataset or format, tripped sanity threshold). These
# get a one-line message instead of a traceback, because the traceback tells the
# operator nothing they can act on. Anything else - including ValueError,
# RuntimeError and KeyError - is a bug and keeps its full traceback.
_EXPECTED_ERRORS = (FileNotFoundError, CurationError)

# Exceptions Click and Typer use for their own control flow: --help, an explicit
# exit code, a usage error. Typer 0.26 vendors its own Exit and Abort rather than
# reusing Click's, so both families have to be listed. Re-raising them before the
# expected-error clause keeps a successful --help from being rewritten into a
# failure should the expected set ever widen to cover their base classes.
_CONTROL_FLOW = (
    click.exceptions.Exit,
    click.exceptions.Abort,
    click.ClickException,
    typer.Exit,
    typer.Abort,
)


class _State:
    """Global options, shared with subcommands through the Click context."""

    def __init__(self) -> None:
        self.tracebacks = False
        self.console = error_console()


class _PipelineGroup(TyperGroup):
    """Root group that turns the pipeline's expected failures into short errors.

    Subclassing the group rather than wrapping each command keeps the handling in
    one place and, because Click routes every subcommand through the root group's
    ``invoke``, covers the whole tree - including under ``CliRunner`` in tests.
    """

    # Annotated as Any because Typer 0.26 vendors its own Click: the context it
    # passes is typer._click.core.Context, not click.core.Context, and naming the
    # private path here would break on any Typer reshuffle.
    def invoke(self, ctx: Any) -> Any:
        """Invoke the selected subcommand, rendering expected errors compactly."""
        try:
            return super().invoke(ctx)
        except _CONTROL_FLOW:
            raise
        except _EXPECTED_ERRORS as exc:
            state = ctx.ensure_object(_State)
            if state.tracebacks:
                raise
            state.console.print(
                Panel(
                    str(exc) or exc.__class__.__name__,
                    title=f"[bold red]{exc.__class__.__name__}",
                    border_style="red",
                )
            )
            raise typer.Exit(1) from exc


app = typer.Typer(
    cls=_PipelineGroup,
    help="Curate drug response datasets into MuData files for drevalpy.",
    no_args_is_help=True,
    add_completion=True,
    rich_markup_mode="rich",
)


def _version_callback(value: bool) -> None:
    """Print the installed package version and exit."""
    if not value:
        return
    try:
        typer.echo(version("data-curation"))
    except PackageNotFoundError:
        typer.echo("unknown (not installed)")
    raise typer.Exit


@app.callback()
def main(
    ctx: typer.Context,
    log_level: Annotated[
        str | None,
        typer.Option("--log-level", help="Explicit log level: CRITICAL, ERROR, WARNING, INFO or DEBUG."),
    ] = None,
    verbose: Annotated[int, typer.Option("--verbose", "-v", count=True, help="Increase verbosity.")] = 0,
    quiet: Annotated[int, typer.Option("--quiet", "-q", count=True, help="Decrease verbosity.")] = 0,
    tracebacks: Annotated[
        bool,
        typer.Option("--traceback", help="Show the full traceback instead of a one-line error."),
    ] = False,
    _version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show the version and exit."),
    ] = False,
) -> None:
    """Configure logging for whichever stage is about to run."""
    state = ctx.ensure_object(_State)
    state.tracebacks = tracebacks
    state.console = _logging.configure(_logging.resolve_level(log_level, verbose, quiet), tracebacks=tracebacks)


def _mount() -> None:
    """Attach each domain's sub-app.

    Imported lazily inside this function so the module-level import graph stays
    flat, and grouped here so the command tree can be read in one place.
    """
    from data_curation.cli.assembly import app as assembly_app
    from data_curation.cli.cell_lines import app as cell_lines_app
    from data_curation.cli.drugs import app as drugs_app
    from data_curation.cli.omics import app as omics_app
    from data_curation.cli.report import report
    from data_curation.cli.response import app as response_app

    app.add_typer(cell_lines_app, name="cell-lines")
    app.add_typer(omics_app, name="omics")
    app.add_typer(response_app, name="response")
    app.add_typer(drugs_app, name="drugs")
    app.add_typer(assembly_app, name="assembly")
    app.command("report")(report)


_mount()
