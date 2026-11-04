"""The ``standard`` input format: a file contract for datasets reshaped elsewhere.

A dataset whose raw data cannot live in this repository (in-house or partner
screens) is reshaped by its own code into five Parquet files and then runs
through the public stages unchanged. Its config block selects the route with
``response.format: standard``, ``transcriptomics.format: standard`` and
``sample_metadata: standard``. The files are:

* ``wells.parquet`` (response raw dir): one row per well in the columns
  ``drevalpy/curate`` reads. ``sample`` and ``drug`` are the dataset's own
  identifiers, ``concentration`` is in µM and ``intensity`` is viability relative
  to the untreated control. ``experiment`` (biological replicate), ``replicate``
  (technical replicate) and ``position`` are optional; any other column is
  per-series metadata.
* ``published.parquet`` (response raw dir, optional): fits the dataset ships
  itself, keyed by ``sample`` and ``drug`` as in the wells. Every other column is
  a numeric metric and becomes the response layer ``<column>_published``.
* ``drugs.parquet`` (response raw dir): one row per ``drug`` of the wells with
  its ``drug_name`` for the PubChem lookup, and optionally a known
  ``pubchem_id`` and ``smiles``.
* ``samples.parquet`` (response raw dir): one row per ``sample_id`` with its
  ``tissue`` and any further metadata columns.
* ``expression.parquet`` (transcriptomics raw dir): long-format expression, one
  row per ``sample_id`` and ``gene_symbol`` with ``tpm`` and optionally ``counts``.

:func:`validate` is the single definition of the contract. The stages call it
through :func:`read`; the code that writes the files should call it before
writing, so a broken export fails at once rather than in a later stage.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from upath import UPath

from data_curation.core.errors import CurationError

#: Format name selecting the standard route in ``config/sources.yaml``.
STANDARD = "standard"

Kind = Literal["wells", "published", "drugs", "samples", "expression"]


@dataclass(frozen=True)
class _Schema:
    """Columns and checks of one contract file.

    Attributes:
        file: File name inside the raw directory.
        strings: Required string columns; no missing values.
        numbers: Required float columns; finite.
        optional_strings: String columns that may be absent; no missing values when present.
        optional_numbers: Float columns that may be absent; NaN allowed.
        unique: Columns whose combination must not repeat.
        positive: Numeric columns that must be strictly positive.
        non_negative: Numeric columns that must not be negative.
        extra: How the remaining columns are treated: kept as they are, required
            to be numeric metrics, or cast to string.
    """

    file: str
    strings: tuple[str, ...] = ()
    numbers: tuple[str, ...] = ()
    optional_strings: tuple[str, ...] = ()
    optional_numbers: tuple[str, ...] = ()
    unique: tuple[str, ...] = ()
    positive: tuple[str, ...] = ()
    non_negative: tuple[str, ...] = ()
    extra: Literal["keep", "metrics", "strings"] = "keep"


SCHEMAS: dict[Kind, _Schema] = {
    "wells": _Schema(
        file="wells.parquet",
        strings=("sample", "drug"),
        numbers=("concentration", "intensity"),
        optional_strings=("experiment", "replicate", "position"),
        positive=("concentration",),
    ),
    "published": _Schema(file="published.parquet", strings=("sample", "drug"), extra="metrics"),
    "drugs": _Schema(
        file="drugs.parquet",
        strings=("drug", "drug_name"),
        optional_strings=("smiles",),
        unique=("drug",),
    ),
    "samples": _Schema(file="samples.parquet", strings=("sample_id", "tissue"), unique=("sample_id",), extra="strings"),
    "expression": _Schema(
        file="expression.parquet",
        strings=("sample_id", "gene_symbol"),
        numbers=("tpm",),
        optional_numbers=("counts",),
        non_negative=("tpm", "counts"),
    ),
}

#: Contract files a standard dataset may leave out.
OPTIONAL_KINDS: frozenset[Kind] = frozenset({"published"})


def path(kind: Kind, raw_dir: UPath) -> UPath:
    """Return where a contract file lives inside a raw directory.

    Args:
        kind: The contract file.
        raw_dir: The dataset's raw directory for the file's modality.

    Returns:
        ``raw_dir / <file name>``.
    """
    return raw_dir / SCHEMAS[kind].file


def read(kind: Kind, raw_dir: UPath) -> pd.DataFrame | None:
    """Read and validate one contract file.

    Args:
        kind: The contract file.
        raw_dir: The dataset's raw directory for the file's modality.

    Returns:
        The validated table, or ``None`` for an optional file that is absent.

    Raises:
        CurationError: If a required file is missing or the table breaks the contract.
    """
    file = path(kind, raw_dir)
    if not file.is_file():
        if kind in OPTIONAL_KINDS:
            return None
        msg = f"Standard dataset file {file} is missing"
        raise CurationError(msg)
    return validate(kind, pd.read_parquet(str(file)), source=str(file))


def validate(kind: Kind, frame: pd.DataFrame, *, source: str | None = None) -> pd.DataFrame:
    """Check a table against the contract and normalise its dtypes.

    Args:
        kind: The contract file the table is meant for.
        frame: The table.
        source: Name used in error messages; defaults to the file name.

    Returns:
        A copy with string columns as ``str`` and numeric columns as ``float64``.

    Raises:
        CurationError: If a required column is missing, a value is missing or out
            of range, a key repeats, or a metric column is not numeric.
    """
    schema = SCHEMAS[kind]
    label = source or schema.file
    _require_columns(frame, (*schema.strings, *schema.numbers), label)

    out = frame.copy()
    for column in schema.strings + _present(out, schema.optional_strings):
        out[column] = _as_strings(out[column], column, label)
    for column in schema.numbers:
        out[column] = _as_numbers(out[column], column, label, finite=True)
    for column in _present(out, schema.optional_numbers):
        out[column] = _as_numbers(out[column], column, label, finite=False)

    _check_ranges(out, schema, label)
    _check_unique(out, schema.unique, label)
    return _normalise_extra(out, schema, label)


def _present(frame: pd.DataFrame, columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(column for column in columns if column in frame.columns)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        msg = f"{label} lacks the required column(s) {missing}"
        raise CurationError(msg)


def _as_strings(series: pd.Series, column: str, label: str) -> pd.Series:
    if series.isna().any():
        msg = f"{label}: column '{column}' has {int(series.isna().sum())} missing value(s)"
        raise CurationError(msg)
    values = series.map(_render)
    if (values == "").any():
        msg = f"{label}: column '{column}' has {int((values == '').sum())} empty value(s)"
        raise CurationError(msg)
    return values


def _as_numbers(series: pd.Series, column: str, label: str, *, finite: bool) -> pd.Series:
    try:
        values = pd.to_numeric(series, errors="raise").astype("float64")
    except (TypeError, ValueError) as exc:
        msg = f"{label}: column '{column}' is not numeric"
        raise CurationError(msg) from exc
    if finite and not np.isfinite(values).all():
        msg = f"{label}: column '{column}' has {int((~np.isfinite(values)).sum())} missing or infinite value(s)"
        raise CurationError(msg)
    return values


def _check_ranges(frame: pd.DataFrame, schema: _Schema, label: str) -> None:
    for column in _present(frame, schema.positive):
        bad = int((frame[column] <= 0).sum())
        if bad:
            msg = f"{label}: column '{column}' has {bad} value(s) that are not positive"
            raise CurationError(msg)
    for column in _present(frame, schema.non_negative):
        bad = int((frame[column] < 0).sum())
        if bad:
            msg = f"{label}: column '{column}' has {bad} negative value(s)"
            raise CurationError(msg)


def _check_unique(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    if not columns:
        return
    duplicated = frame.duplicated(subset=list(columns), keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, list(columns)].drop_duplicates().head(3).to_dict("records")
        msg = f"{label}: {list(columns)} repeat, e.g. {examples}"
        raise CurationError(msg)


def _normalise_extra(frame: pd.DataFrame, schema: _Schema, label: str) -> pd.DataFrame:
    known = {
        *schema.strings,
        *schema.numbers,
        *schema.optional_strings,
        *schema.optional_numbers,
    }
    extra = [column for column in frame.columns if column not in known]
    if schema.extra == "metrics":
        if not extra:
            msg = f"{label} has no metric column besides {sorted(known)}"
            raise CurationError(msg)
        for column in extra:
            frame[column] = _as_numbers(frame[column], column, label, finite=False)
    elif schema.extra == "strings":
        for column in extra:
            frame[column] = frame[column].map(_render)
    return frame


def _render(value: object) -> str:
    """Render one metadata value as a string: missing as empty, a float without its ``.0``."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()
