"""The ontology keys on canonical drug IDs."""

import pandas as pd
from upath import UPath

from data_curation.drugs.annotate_drugs import _canonical_combinations, collect_drugs
from data_curation.drugs.mapping.names import load_source_drug_names

STRUCTURES = pd.DataFrame(
    {
        "pubchem_id": ["5212", "135398744", "GDSC_9"],
        "canonical_id": ["5212", "5212", "GDSC_9"],
        "canonical_smiles": ["S", "S", ""],
        "inchikey": ["K", "K", ""],
    }
)


def _write(directory: UPath, dataset: str, ids: list[str], names: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"DRUG_ID": names, "DRUG_NAME": names, "pubchem_id": ids, "canonical_smiles": [""] * len(ids)}
    ).to_parquet(str(directory / f"{dataset}.parquet"))


def test_two_cids_of_one_molecule_are_one_drug_per_dataset(tmp_dir: UPath, monkeypatch) -> None:
    _write(tmp_dir, "A", ["5212", "GDSC_9"], ["sildenafil", "x"])
    _write(tmp_dir, "B", ["135398744"], ["Sildenafil citrate"])
    monkeypatch.setattr("data_curation.drugs.annotate_drugs.DRUG_MAPPING_DIR", tmp_dir)
    assert collect_drugs(STRUCTURES) == {"A": ["5212", "GDSC_9"], "B": ["5212"]}


def test_source_names_are_collected_under_the_canonical_id(tmp_dir: UPath) -> None:
    _write(tmp_dir, "A", ["135398744"], ["sildenafil"])
    assert load_source_drug_names(tmp_dir, STRUCTURES).to_dict() == {"5212": "sildenafil"}


def test_combinations_move_onto_canonical_ids_and_a_combination_beats_a_single_agent() -> None:
    combinations = pd.DataFrame(
        {
            "is_combination": [False, True, False],
            "n_components": [0, 2, 0],
            "component_pubchem_ids": ["", "135398744|unknown", ""],
        },
        index=pd.Index(["5212", "GDSC_9", "135398744"], name="pubchem_id"),
    )
    result = _canonical_combinations(combinations, STRUCTURES)
    assert result.index.tolist() == ["GDSC_9", "5212"]
    assert result.loc["GDSC_9", "component_pubchem_ids"] == "5212|unknown"
