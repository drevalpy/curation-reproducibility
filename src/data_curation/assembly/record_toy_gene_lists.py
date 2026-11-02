"""Record which drevalpy gene lists the toy dataset must hold, for DVC.

``subsample_toy`` keeps every gene of the lists drevalpy ships, but those CSVs live in the
installed package, not in this repository. This stage writes a fingerprint of them to
``data/07_assembled/toy_gene_lists.json`` and ``subsample_toy`` depends on that file, so a
drevalpy upgrade that changes a list re-runs the subsample while one that does not leaves it
up to date. The stage is ``always_changed``; DVC only treats the output as new when the content differs.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence

from data_curation.core import layout

# Like the toy dataset itself, the fingerprint is a pipeline artefact next to the assembled files.
FINGERPRINT_PATH = layout.ASSEMBLED / "toy_gene_lists.json"


def shipped_gene_lists() -> dict[str, dict[str, tuple[str, ...]]]:
    """Return the gene lists drevalpy ships, as ``{view: {list name: genes}}``."""
    from drevalpy.components.featurizers.sample.gene_lists import GENE_LISTS_BY_VIEW, resolve_gene_list

    return {view: {name: resolve_gene_list(name) for name in names} for view, names in GENE_LISTS_BY_VIEW.items()}


def fingerprint(gene_lists: Mapping[str, Mapping[str, Sequence[str]]]) -> dict[str, dict[str, dict[str, object]]]:
    """Summarise each list as its gene count and the SHA-256 of its ordered genes.

    Args:
        gene_lists: Genes per list name per view, as :func:`shipped_gene_lists` returns.

    Returns:
        ``{view: {list name: {"n_genes": ..., "sha256": ...}}}``.
    """
    return {
        view: {
            name: {"n_genes": len(genes), "sha256": hashlib.sha256("\n".join(genes).encode()).hexdigest()}
            for name, genes in lists.items()
        }
        for view, lists in gene_lists.items()
    }


def main() -> None:
    """Write the fingerprint of the installed drevalpy gene lists."""
    FINGERPRINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINGERPRINT_PATH.write_text(json.dumps(fingerprint(shipped_gene_lists()), indent=2, sort_keys=True) + "\n")
