"""Progress reporting that degrades gracefully when not attached to a terminal.

The drug stages spend most of their wall time in rate-limited PubChem loops -
``map_drugs`` sleeps 0.25 s between lookups and can run for over an hour on
GDSC1. Those loops used to log one INFO line per drug (roughly 1143 lines for a
full run), which is unreadable in a terminal and drowns out the warnings that
actually matter.

:func:`track` replaces that with a Rich progress bar interactively, and with one
periodic summary line under ``dvc repro``, where a bar redrawing itself into a
pipe would produce thousands of useless escape-laden lines.
"""

import logging
from collections.abc import Iterable, Iterator, Sized

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from data_curation.cli._console import error_console, is_interactive

# One log line per this many items when output is not a terminal. PubChem loops
# run at ~4 items/s at best, so this is a line every couple of minutes.
_LOG_EVERY = 100


def track[T](items: Iterable[T], description: str, *, total: int | None = None) -> Iterator[T]:
    """Iterate *items*, reporting progress in whichever way suits the output.

    Args:
        items: Iterable to consume. A summary line needs the length, so a
            non-sized iterable should pass *total* explicitly.
        description: Short label, e.g. ``"Mapping GDSC2 drugs"``.
        total: Item count, when ``len(items)`` is unavailable.

    Yields:
        The items of *items*, unchanged.
    """
    if total is None and isinstance(items, Sized):
        total = len(items)

    if is_interactive():
        yield from _tracked(items, description, total, error_console())
    else:
        yield from _logged(items, description, total)


def _tracked[T](items: Iterable[T], description: str, total: int | None, console: Console) -> Iterator[T]:
    """Yield *items* behind a live Rich progress bar."""
    columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]
    with Progress(*columns, console=console, transient=True) as progress:
        task = progress.add_task(description, total=total)
        for item in items:
            yield item
            progress.advance(task)


def _logged[T](items: Iterable[T], description: str, total: int | None) -> Iterator[T]:
    """Yield *items*, logging a summary line every :data:`_LOG_EVERY` items."""
    logger = logging.getLogger(__name__)
    suffix = f"/{total}" if total is not None else ""
    count = 0
    for item in items:
        yield item
        count += 1
        if count % _LOG_EVERY == 0:
            logger.info("%s: %d%s", description, count, suffix)
    logger.info("%s: %d%s done", description, count, suffix)
