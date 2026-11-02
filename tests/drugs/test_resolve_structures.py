"""One canonical ID and one SMILES per drug, with PubChem mocked out."""

import json
import urllib.error

import pandas as pd
import pytest
from upath import UPath

from data_curation.drugs import resolve_structures as rs
from data_curation.drugs.ontology import pubchem

# Sildenafil as PubChem lists it under two CIDs: tautomers with one InChIKey.
SILDENAFIL_KEY = "BNRNXUUZRGQAQC-UHFFFAOYSA-N"
SILDENAFIL_LOW = "CCCC1=NN(C2=C1NC(=NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C"
SILDENAFIL_HIGH = "CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C"

# L-alanine in two writings: the same molecule, different atom order.
L_ALANINE = "C[C@H](N)C(=O)O"
L_ALANINE_REORDERED = "OC(=O)[C@@H](N)C"


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rs.time, "sleep", lambda _s: None)


def _payload(records: dict[str, tuple[str, str] | None]) -> dict:
    """A PUG-REST property table; a CID PubChem does not know comes back with only its CID."""
    properties = []
    for cid, record in records.items():
        row: dict = {"CID": int(cid)}
        if record:
            row["SMILES"], row["InChIKey"] = record
        properties.append(row)
    return {"PropertyTable": {"Properties": properties}}


def _pubchem(monkeypatch: pytest.MonkeyPatch, records: dict[str, tuple[str, str] | None]) -> list[list[str]]:
    """Answer property requests from *records*, returning the CID batches that were asked for."""
    asked: list[list[str]] = []

    def request(url: str, data: dict | None = None) -> dict:
        batch = url.split("/cid/")[1].split("/")[0].split(",")
        asked.append(batch)
        return _payload({cid: records.get(cid) for cid in batch})

    monkeypatch.setattr(rs, "_request", request)
    return asked


def _study(**smiles: str) -> pd.DataFrame:
    return pd.DataFrame({"pubchem_id": list(smiles), "study_smiles": list(smiles.values())})


def _resolve(study: pd.DataFrame, monkeypatch: pytest.MonkeyPatch, tmp_dir: UPath, records: dict) -> pd.DataFrame:
    _pubchem(monkeypatch, records)
    fetched = rs.fetch_structures([d for d in study["pubchem_id"] if rs.is_cid(d)], tmp_dir / "cache.json")
    return rs.resolve_structures(study, fetched).set_index("pubchem_id")


class TestCanonicalize:
    def test_two_writings_of_one_molecule_give_one_string(self) -> None:
        assert rs.canonicalize(L_ALANINE) == rs.canonicalize(L_ALANINE_REORDERED)

    def test_stereochemistry_is_kept(self) -> None:
        assert rs.canonicalize("C[C@H](N)C(=O)O") != rs.canonicalize("C[C@@H](N)C(=O)O")

    def test_an_unparsable_string_is_returned_unchanged(self) -> None:
        multi = "CC(=O)O, c1ccccc1"
        assert rs.canonicalize(multi) == multi

    def test_empty_stays_empty(self) -> None:
        assert rs.canonicalize("") == ""


class TestOneSmilesPerCid:
    def test_a_stereo_and_a_non_stereo_study_smiles_both_become_the_pubchem_isomeric_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_dir: UPath
    ) -> None:
        records = {"5950": (L_ALANINE, "QNAYBMKLOCPYGJ-REOHCLBHSA-N")}
        # The same CID as two datasets carry it: with and without stereo.
        for study_smiles in ("CC(N)C(=O)O", L_ALANINE_REORDERED):
            table = _resolve(_study(**{"5950": study_smiles}), monkeypatch, tmp_dir, records)
            assert table.loc["5950", "canonical_smiles"] == rs.canonicalize(L_ALANINE)
            assert "@" in table.loc["5950", "canonical_smiles"]

    def test_the_study_smiles_is_ignored_when_pubchem_has_the_cid(
        self, monkeypatch: pytest.MonkeyPatch, tmp_dir: UPath
    ) -> None:
        records = {"5950": (L_ALANINE, "QNAYBMKLOCPYGJ-REOHCLBHSA-N")}
        table = _resolve(_study(**{"5950": "C"}), monkeypatch, tmp_dir, records)
        assert table.loc["5950", "canonical_smiles"] == rs.canonicalize(L_ALANINE)


class TestOneCidPerInchikey:
    @pytest.fixture
    def table(self, monkeypatch: pytest.MonkeyPatch, tmp_dir: UPath) -> pd.DataFrame:
        records = {"5212": (SILDENAFIL_LOW, SILDENAFIL_KEY), "135398744": (SILDENAFIL_HIGH, SILDENAFIL_KEY)}
        return _resolve(_study(**{"135398744": "", "5212": ""}), monkeypatch, tmp_dir, records)

    def test_both_cids_map_to_the_lowest(self, table: pd.DataFrame) -> None:
        assert table["canonical_id"].tolist() == ["5212", "5212"]

    def test_the_lowest_is_numeric_not_lexicographic(self, monkeypatch: pytest.MonkeyPatch, tmp_dir: UPath) -> None:
        records = {"9": (SILDENAFIL_LOW, SILDENAFIL_KEY), "10": (SILDENAFIL_HIGH, SILDENAFIL_KEY)}
        table = _resolve(_study(**{"10": "", "9": ""}), monkeypatch, tmp_dir, records)
        assert table["canonical_id"].tolist() == ["9", "9"]

    def test_the_group_shares_the_canonical_cids_smiles(self, table: pd.DataFrame) -> None:
        assert table["canonical_smiles"].nunique() == 1
        assert table.loc["135398744", "canonical_smiles"] == rs.canonicalize(SILDENAFIL_LOW)

    def test_a_cid_without_a_partner_maps_to_itself(self, monkeypatch: pytest.MonkeyPatch, tmp_dir: UPath) -> None:
        records = {"5950": (L_ALANINE, "QNAYBMKLOCPYGJ-REOHCLBHSA-N"), "5212": (SILDENAFIL_LOW, SILDENAFIL_KEY)}
        table = _resolve(_study(**{"5950": "", "5212": ""}), monkeypatch, tmp_dir, records)
        assert table["canonical_id"].to_dict() == {"5212": "5212", "5950": "5950"}


class TestWithoutPubchemRecord:
    def test_a_pseudo_id_keeps_its_canonicalized_study_smiles(
        self, monkeypatch: pytest.MonkeyPatch, tmp_dir: UPath
    ) -> None:
        table = _resolve(_study(GDSC_1003=L_ALANINE_REORDERED), monkeypatch, tmp_dir, {})
        row = table.loc["GDSC_1003"]
        assert (row["canonical_id"], row["canonical_smiles"], row["inchikey"]) == (
            "GDSC_1003",
            rs.canonicalize(L_ALANINE),
            "",
        )

    def test_a_pseudo_id_with_an_unparsable_smiles_keeps_it_unchanged(
        self, monkeypatch: pytest.MonkeyPatch, tmp_dir: UPath
    ) -> None:
        table = _resolve(_study(PRISM_X="CC(=O)O, c1ccccc1"), monkeypatch, tmp_dir, {})
        assert table.loc["PRISM_X", "canonical_smiles"] == "CC(=O)O, c1ccccc1"

    def test_a_pseudo_id_is_never_sent_to_pubchem(self, monkeypatch: pytest.MonkeyPatch, tmp_dir: UPath) -> None:
        asked = _pubchem(monkeypatch, {})
        study = _study(GDSC_1003="C", **{"5950": ""})
        rs.fetch_structures([d for d in study["pubchem_id"] if rs.is_cid(d)], tmp_dir / "cache.json")
        assert asked == [["5950"]]

    def test_a_cid_pubchem_misses_falls_back_to_the_study_smiles_and_is_logged(
        self, monkeypatch: pytest.MonkeyPatch, tmp_dir: UPath, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            table = _resolve(_study(**{"999999999": L_ALANINE_REORDERED}), monkeypatch, tmp_dir, {"999999999": None})
        row = table.loc["999999999"]
        assert (row["canonical_id"], row["canonical_smiles"]) == ("999999999", rs.canonicalize(L_ALANINE))
        assert "999999999" in caplog.text

    def test_a_missed_cid_never_joins_an_inchikey_group(self, monkeypatch: pytest.MonkeyPatch, tmp_dir: UPath) -> None:
        records = {"5212": (SILDENAFIL_LOW, SILDENAFIL_KEY), "7": None}
        table = _resolve(_study(**{"5212": "", "7": "C"}), monkeypatch, tmp_dir, records)
        assert table["canonical_id"].to_dict() == {"5212": "5212", "7": "7"}


class TestCache:
    def test_a_second_run_does_not_ask_pubchem_again(self, monkeypatch: pytest.MonkeyPatch, tmp_dir: UPath) -> None:
        asked = _pubchem(monkeypatch, {"5950": (L_ALANINE, "K"), "7": None})
        first = rs.fetch_structures(["5950", "7"], tmp_dir / "cache.json")
        second = rs.fetch_structures(["5950", "7"], tmp_dir / "cache.json")
        assert first == second
        assert len(asked) == 1

    def test_only_new_cids_are_fetched(self, monkeypatch: pytest.MonkeyPatch, tmp_dir: UPath) -> None:
        asked = _pubchem(monkeypatch, {"5950": (L_ALANINE, "K"), "5212": (SILDENAFIL_LOW, SILDENAFIL_KEY)})
        rs.fetch_structures(["5950"], tmp_dir / "cache.json")
        rs.fetch_structures(["5950", "5212"], tmp_dir / "cache.json")
        assert asked == [["5950"], ["5212"]]

    def test_requests_are_batched(self, monkeypatch: pytest.MonkeyPatch, tmp_dir: UPath) -> None:
        monkeypatch.setattr(rs, "_BATCH", 2)
        asked = _pubchem(monkeypatch, {})
        rs.fetch_structures(["1", "2", "3"], tmp_dir / "cache.json")
        assert asked == [["1", "2"], ["3"]]

    def test_the_cache_is_a_plain_json_file_of_the_raw_answers(
        self, monkeypatch: pytest.MonkeyPatch, tmp_dir: UPath
    ) -> None:
        _pubchem(monkeypatch, {"5950": (L_ALANINE, "K"), "7": None})
        rs.fetch_structures(["5950", "7"], tmp_dir / "cache.json")
        assert json.loads((tmp_dir / "cache.json").read_text()) == {
            "5950": {"smiles": L_ALANINE, "inchikey": "K"},
            "7": None,
        }

    def test_a_failed_request_aborts_and_keeps_the_batches_already_cached(
        self, monkeypatch: pytest.MonkeyPatch, tmp_dir: UPath
    ) -> None:
        monkeypatch.setattr(rs, "_BATCH", 1)
        calls = []

        def request(url: str, data: dict | None = None) -> dict:
            calls.append(url)
            if len(calls) == 2:
                raise pubchem.PubChemUnavailable("down")
            return _payload({"1": (L_ALANINE, "K")})

        monkeypatch.setattr(rs, "_request", request)
        with pytest.raises(pubchem.PubChemUnavailable):
            rs.fetch_structures(["1", "2"], tmp_dir / "cache.json")
        assert json.loads((tmp_dir / "cache.json").read_text()) == {"1": {"smiles": L_ALANINE, "inchikey": "K"}}


class TestRequest:
    def test_the_url_asks_for_the_isomeric_smiles_and_the_inchikey(
        self, monkeypatch: pytest.MonkeyPatch, tmp_dir: UPath
    ) -> None:
        urls: list[str] = []
        monkeypatch.setattr(rs, "_request", lambda url, data=None: urls.append(url) or {})
        rs.fetch_structures(["5950", "5212"], tmp_dir / "cache.json")
        assert urls == [f"{pubchem._PUG_REST}/compound/cid/5950,5212/property/SMILES,InChIKey/JSON"]

    def test_a_404_is_a_miss_not_an_error(self, monkeypatch: pytest.MonkeyPatch, tmp_dir: UPath) -> None:
        def urlopen(*_args: object, **_kwargs: object) -> None:
            raise urllib.error.HTTPError("http://x", 404, "not found", {}, None)  # ty: ignore[invalid-argument-type]

        monkeypatch.setattr(pubchem.urllib.request, "urlopen", urlopen)
        assert rs.fetch_structures(["999999999"], tmp_dir / "cache.json") == {"999999999": None}


class TestLoadStudyStructures:
    def _write(self, directory: UPath, dataset: str, ids: list[str], smiles: list[str]) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"pubchem_id": ids, "canonical_smiles": smiles}).to_parquet(str(directory / f"{dataset}.parquet"))

    def test_the_first_non_empty_smiles_in_dataset_order_wins(self, tmp_dir: UPath) -> None:
        self._write(tmp_dir, "A", ["1", "2"], ["", "CC"])
        self._write(tmp_dir, "B", ["1", "2"], ["CCC", "CCCC"])
        study = rs.load_study_structures(tmp_dir).set_index("pubchem_id")["study_smiles"]
        assert study.to_dict() == {"1": "CCC", "2": "CC"}

    def test_no_mapping_files_is_an_error(self, tmp_dir: UPath) -> None:
        with pytest.raises(FileNotFoundError):
            rs.load_study_structures(tmp_dir)
