"""Parse the Cellosaurus flat file into a structured table.

Cellosaurus is not a fallback for Cell Model Passports, it covers a different
gap. CMP is the better source for everything it carries (tissue, cancer type,
patient, ploidy), but it only knows the cell lines Sanger models, and it says
almost nothing about whether a line is what its label claims.

Two measured reasons this stage exists:

- **Name resolution.** CTRPv2 screens 52 cell lines CMP does not carry, mostly
  the ``BT*`` glioma stem-cell panel, and Cellosaurus resolves 50 of them - 5.7%
  of that dataset's measurements. Its synonym table is far denser: over the cell
  lines both databases share, CMP offers a median of 0 synonyms against
  Cellosaurus's 2.
- **Data quality.** 66 of the cell lines these datasets screen are flagged
  ``Problematic cell line``, and CMP's free-text fields mention only 7 of them.
  Some are contaminated or misidentified outright (``U-87MG ATCC``,
  ``KPL-1``); others are misclassified, which matters because tissue and cancer
  type are what leave-tissue-out splitting keys on - ``Hep-G2`` is labelled
  hepatocellular carcinoma but is a hepatoblastoma. For a benchmark, that
  belongs in ``obs`` as a flag users can filter on.
"""

import logging
import tempfile
from collections import defaultdict

import pandas as pd
import pooch
from upath import UPath

from data_curation.core.config import source_config

logger = logging.getLogger(__name__)
# Single-value fields, kept as-is.
_SCALAR_FIELDS = {
    "ID": "name",
    "AC": "cellosaurus_id",
    "SX": "sex",
    "AG": "age_at_sampling",
    "CA": "category",
}

# Multi-value fields, joined with "; ".
_LIST_FIELDS = {
    "SY": "synonyms",
    "OX": "species",
    "DR": "cross_references",
    "AS": "secondary_accessions",
    "HI": "derived_from",
    "OI": "same_individual",
    "DI": "disease",
}

# ``CC`` packs many comment types into one field code, keyed by the text before
# the first colon. Only the quality-relevant ones are extracted.
_COMMENT_TOPICS = {
    "Problematic cell line": "problem_description",
    "Discontinued": "discontinued",
    "Misspelling": "misspelling",
}

_ALL_COLUMNS = [
    *_SCALAR_FIELDS.values(),
    *_LIST_FIELDS.values(),
    *_COMMENT_TOPICS.values(),
    "problematic",
]

_RECORD_SEPARATOR = "//"
_FIELD_VALUE_START = 5


def _finalize_record(field_values: dict[str, list[str]]) -> dict[str, object] | None:
    """Turn one record's accumulated field values into an output row.

    Args:
        field_values: Raw values per two-letter field code.

    Returns:
        The row, or None when the record carried no accession.
    """
    if not field_values.get("AC"):
        return None

    row: dict[str, object] = {}
    for code, column in _SCALAR_FIELDS.items():
        values = field_values.get(code) or [""]
        row[column] = values[0]
    for code, column in _LIST_FIELDS.items():
        row[column] = "; ".join(field_values.get(code) or [])

    comments = field_values.get("CC") or []
    for topic, column in _COMMENT_TOPICS.items():
        prefix = f"{topic}:"
        matching = [c[len(prefix) :].strip() for c in comments if c.startswith(prefix)]
        row[column] = " | ".join(matching)

    row["problematic"] = bool(row["problem_description"])
    return row


def _parse_cellosaurus(filepath: UPath) -> pd.DataFrame:
    """Parse a Cellosaurus flat file into one row per cell line.

    The format is line-oriented with two-letter field codes and ``//`` record
    separators. Field codes may repeat within a record; the semantics differ per
    code, so single-value fields keep the first occurrence and multi-value fields
    are joined.

    Args:
        filepath: Location of ``cellosaurus.txt``.

    Returns:
        DataFrame with one row per accession.
    """
    wanted = set(_SCALAR_FIELDS) | set(_LIST_FIELDS) | {"CC"}
    records: list[dict[str, object]] = []
    field_values: dict[str, list[str]] = defaultdict(list)

    with filepath.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(_RECORD_SEPARATOR):
                row = _finalize_record(field_values)
                if row is not None:
                    records.append(row)
                field_values = defaultdict(list)
                continue
            code = line[:2]
            if code in wanted:
                field_values[code].append(line[_FIELD_VALUE_START:].rstrip())

    row = _finalize_record(field_values)
    if row is not None:
        records.append(row)

    frame = pd.DataFrame(records, columns=pd.Index(_ALL_COLUMNS))
    logger.info(
        "Parsed %d Cellosaurus records (%d problematic, %d discontinued, %d with synonyms)",
        len(frame),
        int(frame["problematic"].sum()),
        int((frame["discontinued"] != "").sum()),
        int((frame["synonyms"] != "").sum()),
    )
    return frame


def _strip_reference(value: str) -> str:
    """Drop the ``! comment`` half of a cross-reference accession list."""
    return "; ".join(part.split("!")[0].strip() for part in value.split(";") if part.strip())


def download_cellosaurus(output_path: UPath) -> None:
    """Download the pinned Cellosaurus release and write it to *output_path*.

    Fetches the flat file into a temporary directory via pooch (using the URL
    and hash from ``config/sources.yaml``), parses it, and writes one row per
    accession to *output_path* in Parquet format.

    Args:
        output_path: Destination for the parsed Parquet file. The parent
            directory must already exist.
    """
    cello_params = source_config("cellosaurus")
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_path = pooch.retrieve(
            url=cello_params["url"],
            known_hash=cello_params.get("known_hash"),
            fname="cellosaurus.txt",
            path=tmp_dir,
        )
        frame = _parse_cellosaurus(UPath(raw_path))

    for column in ("derived_from", "same_individual"):
        frame[column] = frame[column].map(_strip_reference)
    frame.to_parquet(str(output_path), index=False)
    logger.info("Wrote %s", output_path)
