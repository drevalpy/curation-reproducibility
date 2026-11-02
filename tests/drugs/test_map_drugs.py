"""map_drugs fallback and failure accounting, with pubchempy stubbed out."""

import pandas as pd
import pubchempy as pcp
import pytest
from upath import UPath

from data_curation.core.errors import CurationError
from data_curation.drugs import map_drugs


class _Compound:
    def __init__(self, cid: int, smiles: str = "C") -> None:
        self.cid = cid
        self.canonical_smiles = smiles


@pytest.fixture(autouse=True)
def _fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(map_drugs.time, "sleep", lambda _s: None)
    map_drugs._LOOKUPS.reset()


def test_not_found_is_no_match_not_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(map_drugs.pcp, "get_compounds", lambda *_a: [])
    assert map_drugs._lookup_pubchem("nothing") == (None, None)

    def raise_not_found(*_a: object) -> None:
        raise pcp.NotFoundError(404, "not found", [])

    monkeypatch.setattr(map_drugs.pcp, "get_compounds", raise_not_found)
    assert map_drugs._lookup_pubchem("nothing") == (None, None)
    assert map_drugs._LOOKUPS.failures == []
    assert map_drugs._LOOKUPS.total == 2


def test_rejected_identifier_is_no_match_not_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def reject(*_a: object) -> None:
        calls.append(1)
        raise pcp.BadRequestError(400, "Unable to standardize the given structure", [])

    monkeypatch.setattr(map_drugs.pcp, "get_compounds", reject)
    assert map_drugs._lookup_pubchem("C1CC1, C1CC1", namespace="smiles") == (None, None)
    assert calls == [1]
    assert map_drugs._LOOKUPS.failures == []


def test_failed_lookup_is_recorded_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def busy(*_a: object) -> None:
        calls.append(1)
        raise pcp.ServerBusyError(503, "busy", [])

    monkeypatch.setattr(map_drugs.pcp, "get_compounds", busy)
    assert map_drugs._lookup_pubchem("aspirin") == (None, None)
    assert len(calls) == map_drugs._MAX_ATTEMPTS
    [(query, reason)] = map_drugs._LOOKUPS.failures
    assert query == "name:aspirin"
    assert "ServerBusyError" in reason


def test_unexpected_exception_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    def bug(*_a: object) -> None:
        raise KeyError("bug")

    monkeypatch.setattr(map_drugs.pcp, "get_compounds", bug)
    with pytest.raises(KeyError):
        map_drugs._lookup_pubchem("aspirin")


def test_main_keeps_pseudo_id_and_enforces_budget(monkeypatch: pytest.MonkeyPatch, tmp_dir: UPath) -> None:
    raw_dir = tmp_dir / "raw"
    raw_dir.mkdir()
    pd.DataFrame({"DRUG_ID": [1, 2], "DRUG_NAME": ["good", "flaky"], "SYNONYMS": [None, None]}).to_csv(
        str(raw_dir / "compounds.csv"), index=False
    )
    output = tmp_dir / "mapping" / "toy.parquet"
    monkeypatch.setattr(map_drugs.layout, "raw_dir", lambda *_a: raw_dir)
    monkeypatch.setattr(map_drugs.layout, "drug_mapping", lambda _d: output)
    monkeypatch.setattr(map_drugs, "dataset_config", lambda *_a: {"format": "gdsc"})

    def get_compounds(identifier: str, _namespace: str) -> list[_Compound]:
        if identifier == "flaky":
            raise pcp.ServerBusyError(503, "busy", [])
        return [_Compound(42)]

    monkeypatch.setattr(map_drugs.pcp, "get_compounds", get_compounds)

    monkeypatch.setattr(map_drugs, "processing_config", lambda _k: {"max_failure_fraction": 0.9})
    map_drugs.main("toy")
    mapping = pd.read_parquet(str(output))
    assert dict(zip(mapping["DRUG_NAME"], mapping["pubchem_id"], strict=True)) == {"good": "42", "flaky": "GDSC_2"}
    report = (tmp_dir / "mapping" / "toy_pubchem_failures.tsv").read_text().splitlines()
    assert report[0] == "query\treason"
    assert report[1].startswith("name:flaky\t")
    assert len(report) == 2

    monkeypatch.setattr(map_drugs, "processing_config", lambda _k: {"max_failure_fraction": 0.05})
    with pytest.raises(CurationError, match="1 of 2 PubChem lookups failed"):
        map_drugs.main("toy")
