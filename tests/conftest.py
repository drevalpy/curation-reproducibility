"""Shared fixtures for the data_curation test suite.

Every fixture builds its inputs from literals into ``tmp_dir``. The suite is
deliberately data-free: it must not read anything under ``data/`` (48 GiB of
DVC-tracked artifacts) and must not touch the network, so it can run as a
pre-commit hook in well under a second.
"""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pytest
from upath import UPath

MODEL_LIST_COLUMNS = ["model_id", "model_name", "RRID", "tissue", "synonyms"]


@pytest.fixture
def tmp_dir(tmp_path: Path) -> UPath:
    """``tmp_path`` as a ``UPath``, which is what the stage modules accept.

    The stages annotate paths as ``UPath`` so that remote filesystems work
    transparently; ``pathlib.Path`` is not a ``UPath``, only the other way
    round, so tests hand over the converted value rather than the raw fixture.
    """
    return UPath(tmp_path)


@pytest.fixture
def write_model_list(tmp_dir: UPath):
    """Return a factory writing a Cell Model Passports ``model_list.csv``.

    The factory accepts row dicts with any subset of the real columns and fills
    the rest with blanks, so a test only states the fields it cares about.
    """

    def _write(rows: Sequence[dict], *, name: str = "model_list.csv", columns: Sequence[str] | None = None) -> UPath:
        frame = pd.DataFrame(list(rows))
        for column in columns or MODEL_LIST_COLUMNS:
            if column not in frame.columns:
                frame[column] = pd.NA
        path = tmp_dir / name
        frame.to_csv(str(path), index=False)
        return path

    return _write
