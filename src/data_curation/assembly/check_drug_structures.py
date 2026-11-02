"""Check that every drug has one SMILES and one ID across all assembled datasets.

``resolve_drug_structures`` promises, and the harmonize and assemble stages are meant to
preserve, that a drug ID carries the same ``canonical_smiles`` in every dataset and that
no InChIKey sits under two IDs. A downstream consumer that concatenates datasets (drevalpy
``concat``) fails on a violation, so this stage fails first, on the files themselves.

Only ``mod/response/var`` of each ``.h5mu`` is read; the files are hundreds of MB.
"""

import json
import logging
from typing import cast

import h5py
import pandas as pd
from anndata.io import read_elem
from upath import UPath

from data_curation.core import layout
from data_curation.core.errors import CurationError

logger = logging.getLogger(__name__)

#: Defined here rather than in ``core/layout.py``: every stage, the fits included, depends on that file.
REPORT_PATH = layout.REPORTS / "drug_structures.json"

#: ``var`` columns that identify the structure of a drug.
SMILES_COLUMN = "canonical_smiles"
INCHIKEY_COLUMN = "inchikey"

#: Placeholder of an InChIKey the ontology could not provide.
NO_INCHIKEY = {"", "unknown"}

#: Longest conflict list written to the report; the counts stay exact.
_MAX_LISTED = 50


def read_response_var(path: UPath) -> pd.DataFrame:
    """Read ``mod/response/var`` of an assembled file without loading the data matrices."""
    with h5py.File(str(path), "r") as handle:
        return cast(pd.DataFrame, read_elem(handle["mod/response/var"]))


def collect_drugs(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack the per-dataset ``var`` tables into one row per (dataset, drug ID)."""
    rows = []
    for dataset, var in frames.items():
        inchikeys = var[INCHIKEY_COLUMN].astype(str) if INCHIKEY_COLUMN in var else pd.Series("", index=var.index)
        rows.append(
            pd.DataFrame(
                {
                    "dataset": dataset,
                    "pubchem_id": var.index.astype(str),
                    "canonical_smiles": var[SMILES_COLUMN].fillna("").astype(str).to_numpy(),
                    "inchikey": inchikeys.to_numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _conflicts(drugs: pd.DataFrame, *, key: str, value: str) -> dict[str, list[str]]:
    """Return ``{key: sorted distinct values}`` for every *key* that has more than one *value*."""
    distinct = drugs.groupby(key)[value].agg(lambda values: sorted(set(values)))
    return {str(k): v for k, v in distinct.items() if len(v) > 1}


def check_structures(frames: dict[str, pd.DataFrame]) -> dict:
    """Check one SMILES per drug ID and one drug ID per InChIKey, and summarise.

    Args:
        frames: ``{dataset: response var table}``, indexed by drug ID.

    Returns:
        A JSON-serialisable report: counts per dataset, the conflicts, and ``passed``.
    """
    drugs = collect_drugs(frames)
    smiles_conflicts = _conflicts(drugs, key="pubchem_id", value="canonical_smiles")
    keyed = drugs[~drugs["inchikey"].isin(NO_INCHIKEY)]
    inchikey_conflicts = _conflicts(keyed, key="inchikey", value="pubchem_id")

    return {
        "passed": not smiles_conflicts and not inchikey_conflicts,
        "datasets": {dataset: len(var) for dataset, var in frames.items()},
        "n_distinct_drug_ids": int(drugs["pubchem_id"].nunique()),
        "n_distinct_inchikeys": int(keyed["inchikey"].nunique()),
        "n_smiles_conflicts": len(smiles_conflicts),
        "n_inchikey_conflicts": len(inchikey_conflicts),
        "smiles_conflicts": dict(list(smiles_conflicts.items())[:_MAX_LISTED]),
        "inchikey_conflicts": dict(list(inchikey_conflicts.items())[:_MAX_LISTED]),
    }


def main() -> None:
    """Check every assembled file and write the report, failing on a conflict.

    Raises:
        CurationError: If there are no assembled files, or a drug ID has several SMILES or
            an InChIKey several drug IDs.
    """
    paths = sorted(layout.ASSEMBLED.glob("*.h5mu"))
    if not paths:
        msg = f"No assembled files found in {layout.ASSEMBLED}"
        raise CurationError(msg)

    frames = {path.stem: read_response_var(path) for path in paths}
    report = check_structures(frames)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    logger.info(
        "Checked %d datasets, %d drug IDs, %d InChIKeys",
        len(frames),
        report["n_distinct_drug_ids"],
        report["n_distinct_inchikeys"],
    )

    if not report["passed"]:
        msg = (
            f"{report['n_smiles_conflicts']} drug ID(s) with several SMILES and "
            f"{report['n_inchikey_conflicts']} InChIKey(s) under several drug IDs; see {REPORT_PATH}"
        )
        raise CurationError(msg)
