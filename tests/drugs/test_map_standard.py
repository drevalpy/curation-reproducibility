"""map_drugs for the ``standard`` input format, with every PubChem lookup faked."""

from pathlib import Path

import pandas as pd
import pytest
import yaml
from upath import UPath

from data_curation.core.errors import CurationError
from data_curation.drugs import map_drugs

DATASET = "ORAKL"


@pytest.fixture(autouse=True)
def _fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(map_drugs.time, "sleep", lambda _s: None)
    map_drugs._LOOKUPS.reset()


@pytest.fixture
def queries(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Fake PubChem: names in ``_NAMES`` resolve, CIDs return a SMILES; every query is recorded."""
    calls: list[tuple[str, str]] = []

    def lookup(identifier: str, namespace: str = "name") -> tuple[int | None, str | None]:
        calls.append((namespace, identifier))
        if namespace == "cid":
            return int(identifier), f"smiles-{identifier}"
        return _NAMES.get(identifier, (None, None))

    monkeypatch.setattr(map_drugs, "_lookup_pubchem", lookup)
    return calls


_NAMES: dict[str, tuple[int | None, str | None]] = {"aspirin": (2244, "CC(=O)O"), "x": (7, "X")}


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drugs: pd.DataFrame, drug_mapping: dict | None) -> None:
    monkeypatch.chdir(tmp_path)
    block: dict = {"response": {"format": "standard", "normalize": False}}
    if drug_mapping is not None:
        block["drug_mapping"] = drug_mapping
    config = tmp_path / "config"
    config.mkdir()
    (config / "sources.yaml").write_text(yaml.safe_dump({"datasets": {DATASET: block}}))
    (config / "processing.yaml").write_text(yaml.safe_dump({"pubchem": {"max_failure_fraction": 0.5}}))
    raw = tmp_path / "data" / "01_raw" / "response" / DATASET
    raw.mkdir(parents=True)
    drugs.to_parquet(raw / "drugs.parquet", index=False)


def _map(tmp_path: Path) -> pd.DataFrame:
    return map_drugs._map_standard(UPath(tmp_path) / "data" / "01_raw" / "response" / DATASET, dataset=DATASET)


def _by_drug(frame: pd.DataFrame) -> dict[str, dict]:
    return frame.set_index("DRUG_ID").to_dict("index")


def test_given_cid_wins_without_name_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, queries: list[tuple[str, str]]
) -> None:
    drugs = pd.DataFrame({"drug": ["d1", "d2"], "drug_name": ["aspirin", "other"], "pubchem_id": [5090, 99.0]})
    drugs["smiles"] = ["", "S"]
    drugs.loc[0, "smiles"] = "GIVEN"
    _setup(tmp_path, monkeypatch, drugs, None)

    rows = _by_drug(_map(tmp_path))

    assert rows["d1"] == {"DRUG_NAME": "aspirin", "pubchem_id": "5090", "canonical_smiles": "GIVEN"}
    assert rows["d2"]["pubchem_id"] == "99"
    assert queries == []


def test_given_cid_without_smiles_fetches_structure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, queries: list[tuple[str, str]]
) -> None:
    drugs = pd.DataFrame({"drug": ["d1", "d2"], "drug_name": ["aspirin", "nothing"], "pubchem_id": [5090, None]})
    _setup(tmp_path, monkeypatch, drugs, None)

    rows = _by_drug(_map(tmp_path))

    assert rows["d1"]["canonical_smiles"] == "smiles-5090"
    assert rows["d2"]["pubchem_id"] == f"{DATASET}_d2"
    assert queries == [("cid", "5090"), ("name", "nothing")]


def test_name_lookup_then_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, queries: list[tuple[str, str]]
) -> None:
    drugs = pd.DataFrame({"drug": ["d1", "d2", "d3"], "drug_name": ["aspirin", "code-9", "unknown"]})
    _setup(tmp_path, monkeypatch, drugs, {"cid_overrides": {"code-9": 123, "aspirin": 1}})

    rows = _by_drug(_map(tmp_path))

    assert rows["d1"] == {"DRUG_NAME": "aspirin", "pubchem_id": "2244", "canonical_smiles": "CC(=O)O"}
    assert rows["d2"] == {"DRUG_NAME": "code-9", "pubchem_id": "123", "canonical_smiles": "smiles-123"}
    assert rows["d3"]["pubchem_id"] == f"{DATASET}_d3"
    assert ("cid", "1") not in queries


def test_drug_id_is_the_native_identifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, queries: list[tuple[str, str]]
) -> None:
    drugs = pd.DataFrame({"drug": ["101", "102"], "drug_name": ["aspirin", "unknown"]})
    _setup(tmp_path, monkeypatch, drugs, None)

    frame = _map(tmp_path)

    assert list(frame.columns) == ["DRUG_ID", "DRUG_NAME", "pubchem_id", "canonical_smiles"]
    assert frame["DRUG_ID"].tolist() == ["101", "102"]
    assert frame["DRUG_NAME"].tolist() == ["aspirin", "unknown"]
    assert frame["pubchem_id"].tolist() == ["2244", f"{DATASET}_102"]


def test_main_never_sends_no_lookup_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, queries: list[tuple[str, str]]
) -> None:
    components: list[str] = []

    def component(name: str) -> int | None:
        components.append(name)
        return {"y": 8}.get(name)

    monkeypatch.setattr(map_drugs, "_lookup_component", component)
    drugs = pd.DataFrame(
        {
            "drug": ["d1", "d2", "d3", "d4", "d5"],
            "drug_name": [" Secret ", "x + secret", "aspirin + y", "aspirin", "Hidden + x"],
            "smiles": ["S1", "S2", "S3", "S4", "S5"],
        }
    )
    _setup(tmp_path, monkeypatch, drugs, {"no_lookup": ["SECRET", "hidden"], "cid_overrides": {"secret": 5}})

    map_drugs.main(DATASET)

    mapping = pd.read_parquet(tmp_path / "data" / "03_drugs" / "mapping" / f"{DATASET}.parquet")
    rows = _by_drug(mapping)
    sent = [identifier.lower() for _, identifier in queries] + [name.lower() for name in components]
    assert not [identifier for identifier in sent if "secret" in identifier or "hidden" in identifier]
    assert components == ["x", "y"]
    assert rows["d1"]["pubchem_id"] == f"{DATASET}_d1"
    assert rows["d1"]["canonical_smiles"] == "S1"
    assert not rows["d1"]["is_combination"]
    assert rows["d2"]["pubchem_id"] == f"{DATASET}_d2"
    assert rows["d2"]["is_combination"]
    assert rows["d2"]["component_pubchem_ids"] == f"unknown|{DATASET}_d1"
    assert rows["d3"]["is_combination"]
    assert rows["d3"]["n_components"] == 2
    assert rows["d3"]["component_pubchem_ids"] == "2244|8"
    assert rows["d4"]["pubchem_id"] == "2244"
    assert rows["d5"]["component_pubchem_ids"] == "unknown|unknown"


def test_non_integer_override_is_a_curation_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    drugs = pd.DataFrame({"drug": ["d1"], "drug_name": ["aspirin"]})
    _setup(tmp_path, monkeypatch, drugs, {"cid_overrides": {"aspirin": "abc"}})

    with pytest.raises(CurationError, match="not an integer CID"):
        map_drugs.main(DATASET)
