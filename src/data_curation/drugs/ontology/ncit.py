"""Resolve NCI Thesaurus drug classes for compounds.

NCIt is the oncology-specific classification in the stack: it places a drug
under branches such as ``EGFR-targeting Agent`` -> ``Targeted Therapy Agent`` ->
``Antineoplastic Agent``, which the broader chemical ontologies do not express.

Concepts reach us as codes (via PubChem's CID -> NCIt links). Those codes are
resolved to labels and walked upwards through the ``parents`` column of the NCIt
flat file, so a very specific concept still yields a useful coarse grouping.
"""

import csv
import logging
from collections import defaultdict, deque

from upath import UPath

logger = logging.getLogger(__name__)

csv.field_size_limit(10**9)

_MIN_COLUMNS = 8
_CODE, _PARENTS, _SYNONYMS, _PREFERRED = 0, 2, 3, 5

# Broad NCIt branches worth surfacing as a grouping label, most specific first.
# Codes verified against the 26.07e release; labels are the NCIt preferred names.
_BRANCH_PRIORITY = [
    ("C163758", "Targeted Therapy Agent"),
    ("C2189", "Signal Transduction Inhibitor"),
    ("C274", "Antineoplastic Agent"),
    ("C1909", "Pharmacologic Substance"),
    ("C1908", "Drug, Food, Chemical or Biomedical Material"),
]
_MAX_DEPTH = 12


class NcitClasses:
    """Lookup of NCIt concept labels and their ancestor branches."""

    def __init__(self, flat_path: UPath) -> None:
        """Load the NCIt flat-file release.

        Args:
            flat_path: ``Thesaurus.txt`` as extracted from the NCI EVS release.
        """
        self._labels: dict[str, str] = {}
        self._parents: dict[str, list[str]] = defaultdict(list)

        with flat_path.open(encoding="utf-8", errors="replace") as fh:
            for row in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
                if len(row) < _MIN_COLUMNS:
                    continue
                code = row[_CODE]
                self._labels[code] = row[_PREFERRED] or row[_SYNONYMS].split("|")[0]
                self._parents[code] = [p for p in row[_PARENTS].split("|") if p]

        logger.info("Loaded NCI Thesaurus: %d concepts", len(self._labels))

    def _ancestors(self, code: str) -> set[str]:
        """Return the ancestor codes of a concept, breadth-first and depth-capped."""
        seen: set[str] = set()
        queue: deque[tuple[str, int]] = deque((p, 1) for p in self._parents.get(code, []))
        while queue:
            current, depth = queue.popleft()
            if current in seen or depth > _MAX_DEPTH:
                continue
            seen.add(current)
            queue.extend((p, depth + 1) for p in self._parents.get(current, []))
        return seen

    def annotate(self, codes: list[str]) -> dict[str, str]:
        """Resolve NCIt codes into grouping fields.

        Args:
            codes: NCIt codes linked to one compound.

        Returns:
            Dict with ``ncit_code`` (the first resolvable code), ``ncit_class``
            (its immediate classifying parent) and ``ncit_branch`` (the broad
            branch it falls under). Values are empty strings when unresolved.
        """
        empty = {"ncit_code": "", "ncit_class": "", "ncit_branch": ""}
        known = [code for code in codes if code in self._labels]
        if not known:
            return empty

        code = known[0]
        ancestors = set()
        for candidate in known:
            ancestors |= self._ancestors(candidate)

        parents = [self._labels[p] for p in self._parents.get(code, []) if p in self._labels]
        branch = next((label for branch_code, label in _BRANCH_PRIORITY if branch_code in ancestors), "")

        return {
            "ncit_code": code,
            "ncit_class": parents[0] if parents else "",
            "ncit_branch": branch,
        }
