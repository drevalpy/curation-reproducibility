"""Resolve target-based drug classification from the Probes & Drugs export.

Probes & Drugs redistributes the ChEMBL protein family classification already
flattened into five levels per target, which is what makes multi-resolution
grouping possible without walking a tree ourselves.

The important subtlety is that the export lists *every measured* target of a
compound, including weak screening hits. Picking the most potent row outright
produces badly wrong annotations (camptothecin resolves to LMNA, and
3-nitropropionic acid to a helicase). Rows are therefore gated to those that
P&D has curated a mechanism of action for, or that the compound is a designated
chemical probe against. Compounds with no such row are left unannotated rather
than being given a plausible-looking guess.
"""

import csv
import logging
from collections import defaultdict

from upath import UPath

logger = logging.getLogger(__name__)

# Some P&D target rows carry very long semicolon-joined gene lists.
csv.field_size_limit(10**9)

CHEMBL_LEVELS = [f"ChEMBL [LEVEL {i}]" for i in range(1, 6)]
_N_LEVELS = len(CHEMBL_LEVELS)


def _potency(row: dict[str, str]) -> float:
    """Return the strongest reported activity for a target row.

    P&D reports pActivity values, so larger is more potent. Rows without any
    measurement sort last but remain eligible.
    """
    values = []
    for key in ("activity_biochemical", "activity_cell"):
        try:
            values.append(float(row.get(key) or ""))
        except ValueError:
            continue
    return max(values, default=-1.0)


class DrugTargets:
    """Lookup of Probes & Drugs target classification, keyed by InChIKey."""

    def __init__(self, compounds_path: UPath, targets_path: UPath) -> None:
        """Load the Probes & Drugs compound and target exports.

        Args:
            compounds_path: Compound export with ``inchikey`` and ``pdid``.
            targets_path: Target export with the classification levels.
        """
        self._by_inchikey: dict[str, dict[str, str]] = {}
        self._by_skeleton: dict[str, dict[str, str]] = {}
        self._targets: dict[str, list[dict[str, str]]] = defaultdict(list)

        with compounds_path.open(encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                inchikey = row.get("inchikey")
                if not inchikey:
                    continue
                self._by_inchikey[inchikey] = row
                self._by_skeleton.setdefault(inchikey.split("-")[0], row)

        with targets_path.open(encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                self._targets[row["pdid"]].append(row)

        logger.info(
            "Loaded Probes & Drugs: %d compounds, %d with target rows",
            len(self._by_inchikey),
            len(self._targets),
        )

    def _lookup(self, inchikey: str) -> dict[str, str] | None:
        """Find a compound by InChIKey, falling back to its skeleton.

        The first InChIKey block encodes the connectivity only, so the fallback
        matches across salt forms and stereochemistry differences between our
        source datasets and P&D.
        """
        exact = self._by_inchikey.get(inchikey)
        if exact is not None:
            return exact
        return self._by_skeleton.get(inchikey.split("-")[0])

    def _primary(self, rows: list[dict[str, str]]) -> dict[str, str] | None:
        """Select the target that best represents the compound's mechanism."""
        candidates = [
            row
            for row in rows
            if row.get("data_for_human_target") == "1" and (row.get("moa") or row.get("experimental_probe") == "1")
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda row: (row.get("experimental_probe") == "1", _potency(row)))

    def annotate(self, inchikey: str) -> dict[str, str]:
        """Resolve the target classification of one compound.

        Args:
            inchikey: InChIKey of the compound.

        Returns:
            Dict with ``target_gene``, ``target_gene_all``, ``target_moa`` and
            ``target_class_l1``..``target_class_l5``. Values are empty strings
            where nothing could be resolved.
        """
        empty = {
            "target_gene": "",
            "target_gene_all": "",
            "target_moa": "",
            **{f"target_class_l{i}": "" for i in range(1, _N_LEVELS + 1)},
        }
        if not inchikey:
            return empty

        compound = self._lookup(inchikey)
        if compound is None:
            return empty

        rows = self._targets.get(compound["pdid"], [])
        primary = self._primary(rows)
        if primary is None:
            return empty

        all_genes = sorted({row["gene_name"] for row in rows if row.get("gene_name")})
        annotation = {
            "target_gene": primary.get("gene_name", ""),
            "target_gene_all": "|".join(all_genes),
            "target_moa": primary.get("moa", ""),
        }
        for index, column in enumerate(CHEMBL_LEVELS, start=1):
            annotation[f"target_class_l{index}"] = primary.get(column) or ""
        return annotation
