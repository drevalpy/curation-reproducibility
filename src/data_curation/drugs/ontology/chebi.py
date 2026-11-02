"""Resolve ChEBI therapeutic roles for compounds.

ChEBI models drug classification with two relations that matter here:

- ``is_a`` builds the chemical hierarchy (erlotinib is a secondary amino
  compound) and also the role hierarchy (EGFR antagonist is a ... antineoplastic
  agent).
- ``has_role`` links a compound to a role.

A compound's roles are therefore not just its direct ``has_role`` edges: roles
are also inherited from its chemical ancestors, and each role implies all of its
own ``is_a`` ancestors. Both closures are applied so a specific annotation such
as "EGFR antagonist" still yields the broad "antineoplastic agent" grouping.
"""

import csv
import gzip
import logging
from collections import defaultdict

from upath import UPath

logger = logging.getLogger(__name__)

_IS_A = "5"
_HAS_ROLE = "4"

# ChEBI IDs of the role subtrees used for the broad therapeutic grouping.
ANTINEOPLASTIC = "35610"
_ROLE_PRIORITY = [
    (ANTINEOPLASTIC, "antineoplastic agent"),
    ("35222", "inhibitor"),
    ("52217", "pharmaceutical"),
]

# Roles that describe chemical behaviour or provenance rather than therapeutic
# intent. They dominate the raw counts but are useless for grouping drugs.
_UNINFORMATIVE = {
    "Brønsted acid",
    "Brønsted base",
    "metabolite",
    "xenobiotic",
    "environmental contaminant",
    "fundamental metabolite",
    "plant metabolite",
    "human metabolite",
    "mouse metabolite",
    "Escherichia coli metabolite",
    "Saccharomyces cerevisiae metabolite",
    "algal metabolite",
    "fungal metabolite",
    "bacterial metabolite",
    "marine metabolite",
    "food component",
}


class ChebiRoles:
    """Lookup of ChEBI compound roles, with both hierarchies closed over."""

    def __init__(self, compounds_path: UPath, relation_path: UPath) -> None:
        """Load the ChEBI flat files.

        Args:
            compounds_path: ``compounds.tsv.gz`` giving ID -> name.
            relation_path: ``relation.tsv.gz`` giving the typed relations.
        """
        self._names: dict[str, str] = {}
        self._parents: dict[str, set[str]] = defaultdict(set)
        self._roles: dict[str, set[str]] = defaultdict(set)
        self._ancestor_cache: dict[str, set[str]] = {}

        with gzip.open(compounds_path.open("rb"), mode="rt", encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                self._names[row["id"]] = row["name"]

        with gzip.open(relation_path.open("rb"), mode="rt", encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                relation, init, final = row["relation_type_id"], row["init_id"], row["final_id"]
                if relation == _IS_A:
                    self._parents[init].add(final)
                elif relation == _HAS_ROLE:
                    self._roles[init].add(final)

        logger.info(
            "Loaded ChEBI: %d compounds, %d is_a edges, %d has_role edges",
            len(self._names),
            sum(len(v) for v in self._parents.values()),
            sum(len(v) for v in self._roles.values()),
        )

    def _ancestors(self, node: str) -> set[str]:
        """Return all ``is_a`` ancestors of a node, memoised."""
        cached = self._ancestor_cache.get(node)
        if cached is not None:
            return cached

        seen: set[str] = set()
        stack = [node]
        while stack:
            for parent in self._parents.get(stack.pop(), ()):
                if parent not in seen:
                    seen.add(parent)
                    stack.append(parent)

        self._ancestor_cache[node] = seen
        return seen

    def _role_ids(self, chebi_id: str) -> tuple[set[str], set[str]]:
        """Return (direct roles, roles closed over the role hierarchy)."""
        direct: set[str] = set()
        for node in {chebi_id} | self._ancestors(chebi_id):
            direct |= self._roles.get(node, set())

        closed = set(direct)
        for role in direct:
            closed |= self._ancestors(role)
        return direct, closed

    def annotate(self, chebi_id: str) -> dict[str, str]:
        """Resolve the roles of one compound into grouping fields.

        Args:
            chebi_id: A ChEBI accession such as ``CHEBI:114785`` or ``114785``.

        Returns:
            Dict with ``drug_role`` (the broadest informative role) and
            ``drug_role_all`` (all informative direct roles, pipe-separated).
            Values are empty strings when nothing could be resolved.
        """
        numeric = chebi_id.split(":")[-1]
        if not numeric.isdigit():
            return {"drug_role": "", "drug_role_all": ""}

        direct, closed = self._role_ids(numeric)
        if not direct:
            return {"drug_role": "", "drug_role_all": ""}

        labels = sorted(
            name for name in (self._names.get(role, "") for role in direct) if name and name not in _UNINFORMATIVE
        )

        primary = next((label for role_id, label in _ROLE_PRIORITY if role_id in closed), "")
        if not primary and labels:
            primary = labels[0]

        return {"drug_role": primary, "drug_role_all": "|".join(labels)}
