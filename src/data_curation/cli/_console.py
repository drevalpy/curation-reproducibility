"""The Rich console for the CLI, and the interactivity check behind it.

Interactivity is decided from ``isatty()`` rather than from Rich's
``Console.is_terminal``. Rich treats the mere *presence* of ``FORCE_COLOR`` in the
environment as a request to behave like a terminal, whatever its value, so
``FORCE_COLOR=0`` makes ``is_terminal`` return True on a pipe. Relying on that
would let a live progress bar redraw itself thousands of times into a ``dvc
repro`` log.

The console widens itself when not interactive. Rich assumes 80 columns for
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


def is_interactive() -> bool:
    """Report whether stderr is attached to a real terminal."""
    try:
        return bool(sys.stderr.isatty())
    except (AttributeError, ValueError):
        # A closed or replaced stream (pytest's capture does both) is not a tty.
        return False


def error_console() -> Console:
    """Return the stderr console, used for logs and error panels."""
    if is_interactive():
        return Console(stderr=True)
    return Console(stderr=True, width=NON_TTY_WIDTH, height=_NON_TTY_HEIGHT)
