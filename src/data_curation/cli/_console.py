"""Rich consoles for the CLI, and the interactivity check they share.

Interactivity is decided from ``isatty()`` rather than from Rich's
``Console.is_terminal``. Rich treats the mere *presence* of ``FORCE_COLOR`` in the
environment as a request to behave like a terminal, whatever its value, so
``FORCE_COLOR=0`` makes ``is_terminal`` return True on a pipe. Relying on that
would let a live progress bar redraw itself thousands of times into a ``dvc
repro`` log.

Both consoles widen themselves when not interactive. Rich assumes 80 columns for
a pipe, which hard-wraps the stage table and the aligned coverage tables
``annotate_drugs`` logs - and a pipe is exactly where most of this output is read.

Logs and error panels go to stderr, so anything a command writes to stdout stays
machine-readable.
"""

import sys

from rich.console import Console

# Wide enough for the widest output the pipeline produces: the stage table and
# the per-dataset coverage lines in annotate_drugs.
NON_TTY_WIDTH = 160

# Rich only honours an explicit width if a height is given too: with width alone,
# Console.size falls through to its dumb-terminal branch and reports 80x25. The
# height itself is irrelevant to us, since nothing here paginates.
_NON_TTY_HEIGHT = 50


def is_interactive(*, stderr: bool = True) -> bool:
    """Report whether the given stream is attached to a real terminal."""
    stream = sys.stderr if stderr else sys.stdout
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        # A closed or replaced stream (pytest's capture does both) is not a tty.
        return False


def _console(*, stderr: bool) -> Console:
    """Build a console, widened when its stream is not a terminal."""
    if is_interactive(stderr=stderr):
        return Console(stderr=stderr)
    return Console(stderr=stderr, width=NON_TTY_WIDTH, height=_NON_TTY_HEIGHT)


def out_console() -> Console:
    """Return the stdout console, used for command output such as tables."""
    return _console(stderr=False)


def error_console() -> Console:
    """Return the stderr console, used for logs and error panels."""
    return _console(stderr=True)
