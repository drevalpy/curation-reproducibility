"""Render the summary notebook to static HTML with ``marimo export html``."""

import logging
import subprocess  # noqa: S404 - runs the marimo CLI with fixed arguments
import sys
from pathlib import Path

from upath import UPath

from data_curation.core.errors import CurationError
from data_curation.core.layout import REPORT_HTML

logger = logging.getLogger(__name__)

NOTEBOOK = Path(__file__).with_name("notebook.py")


def main(output: UPath = REPORT_HTML) -> None:
    """Execute the notebook and write its rendered HTML to *output*.

    The notebook runs in a subprocess of the current interpreter, so it sees the
    same environment as the pipeline and resolves ``data/`` relative to the
    repository root, where DVC runs every stage.

    Raises:
        CurationError: If the export fails; marimo's output is logged.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "marimo",
        "export",
        "html",
        str(NOTEBOOK),
        "--output",
        str(output),
        "--force",
        "--no-include-code",
    ]
    logger.info("Rendering %s to %s", NOTEBOOK.name, output)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    if result.returncode != 0 or not output.exists():
        logger.error("marimo export output:\n%s%s", result.stdout, result.stderr)
        msg = f"marimo export failed with exit code {result.returncode}"
        raise CurationError(msg)
    logger.info("Wrote %s", output)
