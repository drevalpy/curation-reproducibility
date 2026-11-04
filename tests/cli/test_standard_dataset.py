"""A dataset in the standard format runs through every public stage after the export.

Builds the five standard files from literals, then runs the stages a private
pipeline calls, in order, with the two network steps (PubChem lookups) and the
curve fit (Nextflow) replaced by fakes. The assembled file is checked for what
drevalpy reads: the response, the omics, the per-sample tissue and the published layers.
"""

from pathlib import Path

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
import yaml

from data_curation.assembly import assemble_mudata, check_drug_structures
from data_curation.cell_lines import process_sample_meta
from data_curation.core import layout, standard
from data_curation.drugs import map_drugs, resolve_structures
from data_curation.omics import process_transcriptomics
from data_curation.report.reading import read_snapshot
from data_curation.response import harmonize_response, prepare_viability

DATASET = "Inhouse"

SOURCES = {
    "datasets": {
        DATASET: {
            "sample_metadata": "standard",
            "response": {"format": "standard", "normalize": False},
            "drug_mapping": {"cid_overrides": {}, "no_lookup": ["secret-1"]},
            "transcriptomics": {"format": "standard", "counts_unit": "salmon_numreads"},
        }
    }
}
PROCESSING = {"pubchem": {"max_failure_fraction": 0.5}, "toy": {"name": "TOYv1"}}

#: Name -> CID the fake PubChem knows.
PUBCHEM = {"gemcitabine": 60750, "5fu": 3385, "oxaliplatin": 9887053}


def _wells() -> pd.DataFrame:
    rows = []
    for sample in ("S1", "S2"):
        for drug in ("gemcitabine", "5fu + oxaliplatin", "secret-1"):
            for passage in (3, 5):
                for dose, viability in zip((0.01, 0.1, 1.0, 10.0), (1.0, 0.9, 0.5, 0.1), strict=True):
                    rows.append(
                        {
                            "sample": sample,
                            "drug": drug,
                            "concentration": dose,
                            "intensity": viability,
                            "experiment": f"{sample}:P{passage}",
                            "replicate": f"1:plate{passage}:0",
                            "position": "A1",
                            "plate_id": f"plate{passage}",
                        }
                    )
    return pd.DataFrame(rows)


def _write_standard_files() -> None:
    response_dir = layout.raw_dir("response", DATASET)
    transcriptomics_dir = layout.raw_dir("transcriptomics", DATASET)
    response_dir.mkdir(parents=True)
    transcriptomics_dir.mkdir(parents=True)
    drugs = ["gemcitabine", "5fu + oxaliplatin", "secret-1"]
    tables: dict[standard.Kind, pd.DataFrame] = {
        "wells": _wells(),
        "published": pd.DataFrame({"sample": ["S1", "S1"], "drug": ["gemcitabine"] * 2, "AUC": [0.4, 0.6]}),
        "drugs": pd.DataFrame({"drug": drugs, "drug_name": drugs}),
        "samples": pd.DataFrame(
            {"sample_id": ["S1", "S2"], "tissue": ["Pancreas", "Lung"], "cancer_type": ["PDAC", "Lung"]}
        ),
    }
    for kind, frame in tables.items():
        standard.validate(kind, frame).to_parquet(str(standard.path(kind, response_dir)), index=False)
    expression = pd.DataFrame(
        {
            "sample_id": ["S1", "S1", "S2", "S2"],
            "gene_symbol": ["TP53", "KRAS", "TP53", "KRAS"],
            "tpm": [1.0, 3.0, 7.0, 0.0],
            "counts": [10.0, 30.0, 70.0, 0.0],
        }
    )
    standard.validate("expression", expression).to_parquet(str(standard.path("expression", transcriptomics_dir)))


def _fake_fit() -> None:
    """Write the curves file the Nextflow fit would, keyed by the native identifiers."""
    drugs = ["gemcitabine", "5fu + oxaliplatin", "secret-1"]
    curves = ad.AnnData(
        X=np.array([[6.0, 5.5, 5.0], [6.5, 6.0, 4.5]], dtype=np.float32),
        obs=pd.DataFrame(index=pd.Index(["S1", "S2"])),
        var=pd.DataFrame(index=pd.Index(drugs)),
        layers={"signal_quality": np.ones((2, 3), dtype=np.float32), "RMSE": np.zeros((2, 3), dtype=np.float32)},
    )
    layout.curves(DATASET).parent.mkdir(parents=True, exist_ok=True)
    curves.write_h5ad(str(layout.curves(DATASET)))


@pytest.fixture
def pubchem_calls() -> list[str]:
    """Every identifier the fake PubChem was asked about."""
    return []


@pytest.fixture
def workdir(monkeypatch, tmp_path: Path, pubchem_calls: list[str]) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "sources.yaml").write_text(yaml.safe_dump(SOURCES))
    (tmp_path / "config" / "processing.yaml").write_text(yaml.safe_dump(PROCESSING))

    def fake_lookup(identifier: str, namespace: str = "name") -> tuple[int | None, str | None]:
        pubchem_calls.append(identifier)
        if namespace == "cid":
            return int(identifier), "C"
        cid = PUBCHEM.get(identifier.lower())
        return cid, ("C" * (cid % 5 + 1) if cid else None)

    def fake_fetch(cids: list[str], _cache_path) -> dict:
        return {cid: {"smiles": "C" * (int(cid) % 5 + 1), "inchikey": f"KEY{cid}"} for cid in cids}

    monkeypatch.setattr(map_drugs, "_lookup_pubchem", fake_lookup)
    monkeypatch.setattr(map_drugs, "_REQUEST_DELAY", 0)
    monkeypatch.setattr(resolve_structures, "fetch_structures", fake_fetch)
    _write_standard_files()
    return tmp_path


def _run_pipeline() -> md.MuData:
    prepare_viability.main(DATASET)
    _fake_fit()
    map_drugs.main(DATASET)
    resolve_structures.main()
    harmonize_response.main(DATASET)
    process_transcriptomics.main(DATASET)
    process_sample_meta.main(DATASET)
    assemble_mudata.main(DATASET)
    check_drug_structures.main()
    return md.read_h5mu(str(layout.assembled(DATASET)))


class TestStandardDataset:
    def test_the_well_table_keeps_the_replicate_hierarchy(self, workdir: Path) -> None:
        prepare_viability.main(DATASET)
        wells = pd.read_parquet(str(layout.wells(DATASET)))
        assert set(wells["experiment"]) == {"S1:P3", "S1:P5", "S2:P3", "S2:P5"}
        assert "plate_id" in wells.columns

    def test_the_assembled_file_has_response_omics_and_metadata(self, workdir: Path) -> None:
        mdata = _run_pipeline()
        assert set(mdata.mod) == {"response", "gene_expression"}
        assert sorted(mdata.obs_names) == ["S1", "S2"]
        assert mdata.obs.loc["S1", "tissue"] == "Pancreas"
        assert mdata.obs.loc["S2", "cancer_type"] == "Lung"
        assert mdata.uns["dataset_name"] == DATASET

    def test_drugs_are_canonical_and_the_secret_is_never_sent(self, workdir: Path, pubchem_calls: list[str]) -> None:
        mdata = _run_pipeline()
        assert {"60750", f"{DATASET}_secret-1"} <= set(mdata.mod["response"].var_names)
        assert pubchem_calls
        assert not any("secret" in name for name in pubchem_calls)

    def test_published_layers_are_averaged(self, workdir: Path) -> None:
        response = _run_pipeline().mod["response"]
        assert isinstance(response, ad.AnnData)
        auc = response.to_df(layer="AUC_published")
        assert auc.loc["S1", "60750"] == pytest.approx(0.5)
        assert np.isnan(auc.loc["S2", "60750"])

    def test_expression_is_log2_tpm_plus_one(self, workdir: Path) -> None:
        expression = _run_pipeline().mod["gene_expression"]
        frame = pd.DataFrame(expression.X, index=expression.obs_names, columns=expression.var_names)
        assert frame.loc["S2", "TP53"] == pytest.approx(3.0)

    def test_the_report_reads_the_file(self, workdir: Path) -> None:
        _run_pipeline()
        snapshot = read_snapshot(layout.assembled(DATASET))
        assert set(snapshot.tissues) == {"Pancreas", "Lung"}
