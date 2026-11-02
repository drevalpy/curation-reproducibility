"""Exceptions for failures the operator can fix.

The CLI renders these (and :class:`FileNotFoundError`) as a one-line message.
Every other exception is treated as a bug and keeps its traceback, so raise
:class:`CurationError` only for bad or missing inputs and configuration, never
to wrap an unexpected internal failure.
"""


class CurationError(Exception):
    """An expected, user-facing failure: missing input, bad name, bad config."""
