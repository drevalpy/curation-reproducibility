"""The cross-dataset structure check, on small var tables."""

import json

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
from upath import UPath

from data_curation.assembly import check_drug_structures as check
from data_curation.core import layout
from data_curation.core.errors import CurationError


def _var(ids: list[str], smiles: list[str], inchikeys: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"canonical_smiles": smiles, "inchikey": inchikeys}, index=pd.Index(ids, name="pubchem_id"))


def test_consistent_datasets_pass() -> None:
    frames = {
        "A": _var(["1", "2"], ["C", "CC"], ["k1", "k2"]),
        "B": _var(["2", "3"], ["CC", "CCC"], ["k2", "k3"]),
    }
    report = check.check_structures(frames)
    assert report["passed"]
    assert report["datasets"] == {"A": 2, "B": 2}
    assert report["n_distinct_drug_ids"] == 3


def test_one_id_with_two_smiles_fails() -> None:
    frames = {"A": _var(["1"], ["C"], ["k1"]), "B": _var(["1"], ["C(C)"], ["k1"])}
    report = check.check_structures(frames)
    assert not report["passed"]
    assert report["smiles_conflicts"] == {"1": ["C", "C(C)"]}


def test_one_inchikey_under_two_ids_fails() -> None:
    frames = {"A": _var(["1"], ["C"], ["k1"]), "B": _var(["2"], ["C"], ["k1"])}
    report = check.check_structures(frames)
    assert not report["passed"]
    assert report["inchikey_conflicts"] == {"k1": ["1", "2"]}


@pytest.mark.parametrize("missing", ["unknown", ""])
def test_ids_without_an_inchikey_may_share_the_placeholder(missing: str) -> None:
    frames = {"A": _var(["GDSC_1", "GDSC_2"], ["", ""], [missing, missing])}
    assert check.check_structures(frames)["passed"]


def _write_h5mu(path: UPath, var: pd.DataFrame) -> None:
    response = ad.AnnData(X=np.zeros((1, len(var)), dtype=np.float32), var=var)
    md.MuData({"response": response}).write(str(path))


def test_main_reads_only_the_response_var_and_writes_the_report(
    tmp_dir: UPath, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(str(tmp_dir))
    layout.ASSEMBLED.mkdir(parents=True)
    _write_h5mu(layout.ASSEMBLED / "A.h5mu", _var(["1"], ["C"], ["k1"]))
    _write_h5mu(layout.ASSEMBLED / "B.h5mu", _var(["1", "2"], ["C", "CC"], ["k1", "k2"]))

    check.main()

    report = json.loads(check.REPORT_PATH.read_text())
    assert report["passed"]
    assert report["datasets"] == {"A": 1, "B": 2}


def test_main_writes_the_report_and_raises_on_a_conflict(tmp_dir: UPath, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(str(tmp_dir))
    layout.ASSEMBLED.mkdir(parents=True)
    _write_h5mu(layout.ASSEMBLED / "A.h5mu", _var(["1"], ["C"], ["k1"]))
    _write_h5mu(layout.ASSEMBLED / "B.h5mu", _var(["1"], ["CC"], ["k1"]))

    with pytest.raises(CurationError, match="several SMILES"):
        check.main()

    assert not json.loads(check.REPORT_PATH.read_text())["passed"]


def test_no_assembled_files_is_an_error(tmp_dir: UPath, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(str(tmp_dir))
    with pytest.raises(CurationError, match="No assembled files"):
        check.main()
