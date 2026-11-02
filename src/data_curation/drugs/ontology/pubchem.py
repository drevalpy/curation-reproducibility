"""PubChem REST helpers for drug ontology annotation.

Two kinds of lookup are needed:

- Per-compound cross references (InChIKey, ChEBI ID) for the drugs in our
  datasets. These are batched, since PubChem accepts many CIDs per request.
- The complete PubChem -> NCI Thesaurus link table, which is only available by
  paging through the PUG-View annotations endpoint.

Requests go through ``urllib`` rather than ``pubchempy`` because these are bulk
endpoints that ``pubchempy`` does not wrap, and because some PubChem payloads
contain bytes that are not valid UTF-8 and must be decoded permissively.
"""

import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from upath import UPath

from data_curation.cli._progress import track
from data_curation.core.errors import CurationError

logger = logging.getLogger(__name__)

_PUG_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_PUG_VIEW = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"

# PubChem asks for no more than 5 requests/s; these batch sizes keep the number
# of requests low while staying inside the URL length limit for GET requests.
_SYNONYM_BATCH = 100
_PROPERTY_BATCH = 150
_REQUEST_DELAY = 0.25
_MAX_ATTEMPTS = 4
_RETRY_BACKOFF = 2.0

_NCIT_CODE = re.compile(r"C\d+")


class PubChemUnavailable(RuntimeError):  # noqa: N818 - name is part of the PubChem interface
    """PubChem could not be reached or kept failing after every retry.

    This is distinct from "PubChem has no such record", which is a valid empty
    answer and never raises.
    """


def _is_retryable(exc: Exception) -> bool:
    """Whether a failed request may succeed if repeated (5xx, throttling, network)."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code >= 500 or exc.code == 429
    return True


def _request(url: str, data: dict[str, str] | None = None) -> dict:
    """Fetch a JSON payload, retrying transient failures with linear backoff.

    Args:
        url: Endpoint to call.
        data: Optional form fields; when given the request is sent as a POST.

    Returns:
        The decoded JSON object, or an empty dict if PubChem answered 404
        ("not found"), which is a valid answer rather than a failure.

    Raises:
        PubChemUnavailable: If the request failed (network error, 5xx, throttling,
            unparsable body) on every attempt, or failed with a non-retryable
            client error.
    """
    body = urllib.parse.urlencode(data).encode() if data else None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(url, body, timeout=120) as response:  # noqa: S310
                # PubChem occasionally emits non-UTF-8 bytes inside synonym
                # lists; replacing them is preferable to losing a whole batch.
                return json.loads(response.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                logger.debug("PubChem has no record for %s", url)
                return {}
            failure: Exception = exc
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            failure = exc
        if attempt == _MAX_ATTEMPTS or not _is_retryable(failure):
            msg = f"PubChem request failed after {attempt} attempt(s): {url}: {failure}"
            raise PubChemUnavailable(msg) from failure
        time.sleep(_RETRY_BACKOFF * attempt)
    msg = f"PubChem request failed: {url}"  # pragma: no cover
    raise PubChemUnavailable(msg)  # pragma: no cover


def enforce_failure_budget(
    failures: Sequence[tuple[str, str]],
    total: int,
    max_fraction: float,
    report_path: UPath | Path,
) -> None:
    """Write the PubChem failure report and fail if too many lookups failed.

    The report is written on every call, empty when nothing failed, so that it is
    a stable stage output.

    Args:
        failures: ``(query, reason)`` pairs for lookups that failed (not for
            lookups that legitimately found nothing).
        total: Number of lookups attempted.
        max_fraction: Largest tolerated ``len(failures) / total``.
        report_path: TSV to write, with columns ``query`` and ``reason``.

    Raises:
        CurationError: If ``total > 0`` and the failure fraction exceeds
            ``max_fraction``.
    """
    path = UPath(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["query\treason"]
    lines.extend(f"{_tsv_field(query)}\t{_tsv_field(reason)}" for query, reason in failures)
    path.write_text("\n".join(lines) + "\n")

    if failures:
        logger.warning("%d of %d PubChem lookups failed; see %s", len(failures), total, path)
    if total > 0 and len(failures) / total > max_fraction:
        msg = (
            f"{len(failures)} of {total} PubChem lookups failed "
            f"({len(failures) / total:.1%}), above the allowed {max_fraction:.1%}; see {path}"
        )
        raise CurationError(msg)


def _tsv_field(value: str) -> str:
    """Flatten tabs and newlines so one failure stays on one TSV line."""
    return " ".join(str(value).split())


def _batched(items: list[str], size: int) -> list[list[str]]:
    """Split a list into consecutive chunks of at most ``size`` elements."""
    return [items[i : i + size] for i in range(0, len(items), size)]


def fetch_cross_references(cids: list[str], failures: list[tuple[str, str]] | None = None) -> dict[str, dict[str, str]]:
    """Look up the preferred name and ChEBI ID for each CID.

    PubChem does not expose a dedicated cross-reference endpoint for ChEBI, but
    ChEBI IDs are present in the synonym list, so they are recovered from there.

    Args:
        cids: Numeric PubChem CIDs as strings.
        failures: If given, batches that fail after all retries are recorded here
            as ``(cid, reason)`` pairs, one per CID, instead of raising.

    Returns:
        Mapping of CID to a dict with ``name`` and ``chebi_id`` keys.

    Raises:
        PubChemUnavailable: If a batch fails and ``failures`` is None.
    """
    result: dict[str, dict[str, str]] = {}
    batches = _batched(cids, _SYNONYM_BATCH)
    for batch in track(batches, "Fetching PubChem cross references"):
        try:
            payload = _request(f"{_PUG_REST}/compound/cid/synonyms/JSON", {"cid": ",".join(batch)})
        except PubChemUnavailable as exc:
            if failures is None:
                raise
            failures.extend((f"cross_reference:{cid}", str(exc)) for cid in batch)
            continue
        result.update(_parse_cross_references(payload))
        logger.debug("Resolved cross references for %d/%d CIDs", len(result), len(cids))
        time.sleep(_REQUEST_DELAY)
    return result


def _parse_cross_references(payload: dict) -> dict[str, dict[str, str]]:
    """Extract ``{cid: {name, chebi_id}}`` from one synonyms response."""
    parsed: dict[str, dict[str, str]] = {}
    for record in payload.get("InformationList", {}).get("Information", []):
        synonyms = record.get("Synonym", [])
        chebi = next((s for s in synonyms if s.upper().startswith("CHEBI")), "")
        parsed[str(record["CID"])] = {
            "name": synonyms[0] if synonyms else "",
            "chebi_id": chebi,
        }
    return parsed


def fetch_inchikeys(cids: list[str], failures: list[tuple[str, str]] | None = None) -> dict[str, str]:
    """Look up the InChIKey for each CID.

    Args:
        cids: Numeric PubChem CIDs as strings.
        failures: If given, batches that fail after all retries are recorded here
            as ``(cid, reason)`` pairs, one per CID, instead of raising.

    Returns:
        Mapping of CID to InChIKey.

    Raises:
        PubChemUnavailable: If a batch fails and ``failures`` is None.
    """
    result: dict[str, str] = {}
    for batch in track(_batched(cids, _PROPERTY_BATCH), "Fetching PubChem InChIKeys"):
        try:
            payload = _request(f"{_PUG_REST}/compound/cid/{','.join(batch)}/property/InChIKey/JSON")
        except PubChemUnavailable as exc:
            if failures is None:
                raise
            failures.extend((f"inchikey:{cid}", str(exc)) for cid in batch)
            continue
        for record in payload.get("PropertyTable", {}).get("Properties", []):
            if record.get("InChIKey"):
                result[str(record["CID"])] = record["InChIKey"]
        logger.debug("Resolved InChIKeys for %d/%d CIDs", len(result), len(cids))
        time.sleep(_REQUEST_DELAY)
    return result


def _ncit_codes(record: dict) -> list[str]:
    """Extract the NCIt codes annotated to one PUG-View record."""
    return [
        markup["String"]
        for item in record.get("Data", [])
        for markup in item.get("Value", {}).get("StringWithMarkup", [])
        if _NCIT_CODE.fullmatch(markup.get("String", ""))
    ]


def fetch_cid_to_ncit(heading: str) -> dict[str, list[str]]:
    """Download the complete PubChem CID -> NCI Thesaurus code table.

    The annotations endpoint is paged and reports its own total page count, so
    the loop follows that rather than assuming a fixed size.

    Args:
        heading: PUG-View compound heading to page through.

    Returns:
        Mapping of CID to the NCIt codes annotated to it.

    Raises:
        PubChemUnavailable: If a page cannot be fetched; a partial table would
            silently drop NCIt classes.
    """
    encoded = urllib.parse.quote_plus(heading)
    mapping: dict[str, list[str]] = {}
    page, total_pages = 1, None

    while total_pages is None or page <= total_pages:
        payload = _request(f"{_PUG_VIEW}/annotations/heading/JSON?heading_type=Compound&heading={encoded}&page={page}")
        annotations = payload.get("Annotations", {})
        if total_pages is None:
            total_pages = annotations.get("TotalPages")
            if not total_pages:
                logger.warning("PubChem returned no pages for heading '%s'", heading)
                return mapping

        for record in annotations.get("Annotation", []):
            codes = _ncit_codes(record)
            if not codes:
                continue
            for cid in record.get("LinkedRecords", {}).get("CID", []):
                mapping.setdefault(str(cid), []).extend(codes)

        logger.info("NCIt page %d/%s (%d CIDs so far)", page, total_pages, len(mapping))
        page += 1
        time.sleep(_REQUEST_DELAY)

    return mapping
