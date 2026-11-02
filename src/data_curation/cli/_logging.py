"""Logging configuration for the CLI.

Every stage logs through the stdlib ``logging`` module and knows nothing about
Rich; the handler is installed once here, from the root callback, so the
implementation modules stay free of presentation concerns. This replaces the nine
near-identical ``logging.basicConfig`` calls the stage modules used to carry,
which had already drifted apart in level and format.

Two settings matter for running under DVC rather than in a terminal:

- ``markup=False``, because log messages interpolate values the pipeline does not
  control. Drug names contain square brackets (``JQ-1:UNC0638 (2:1 mol/mol)`` is
  tame, but ChEBI role names are not), and Rich would try to parse those as
  console markup tags and mangle or drop them.
- The widened non-TTY console from :mod:`data_curation.cli._console`, so the wide
  aligned tables ``annotate_drugs`` logs are not hard-wrapped at 80 columns.
"""

import logging

from rich.console import Console
from rich.logging import RichHandler

from data_curation.cli._console import error_console

_LEVELS = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]


def resolve_level(level: str | None, verbose: int, quiet: int) -> int:
    """Resolve the effective log level from the global CLI options.

    An explicit ``--log-level`` wins outright; otherwise ``-v``/``-q`` step up
    and down from INFO, which is what the pipeline stages assume.

    Args:
        level: Explicit level name, case-insensitive, or None.
        verbose: Number of ``-v`` flags given.
        quiet: Number of ``-q`` flags given.

    Returns:
        A stdlib logging level.

    Raises:
        ValueError: If *level* is not a known level name.
    """
    if level is not None:
        name = level.upper()
        if name not in _LEVELS:
            msg = f"Unknown log level '{level}'; choose one of {', '.join(_LEVELS)}"
            raise ValueError(msg)
        return getattr(logging, name)

    index = _LEVELS.index("INFO") + verbose - quiet
    return getattr(logging, _LEVELS[min(max(index, 0), len(_LEVELS) - 1)])


def configure(level: int, *, tracebacks: bool) -> Console:
    """Install the Rich log handler on the root logger.

    Args:
        level: Effective log level.
        tracebacks: Whether to let Rich render tracebacks for logged exceptions.

    Returns:
        The console the handler writes to, reused for error panels.
    """
    console = error_console()
    handler = RichHandler(
        console=console,
        markup=False,
        rich_tracebacks=tracebacks,
        show_path=False,
        log_time_format="[%H:%M:%S]",
    )
    logging.basicConfig(level=level, format="%(message)s", handlers=[handler], force=True)
    return console
