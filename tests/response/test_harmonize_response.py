"""Tests for the harmonize_response stage's axis rename and annotation.

These exercise the seam Phase 2b created: curves fitted on native identifiers
being renamed onto the sample IDs (Cellosaurus accessions for cell lines) and PubChem
CIDs drevalpy expects,
with the duplicates that renaming can produce resolved by fit quality.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from upath import UPath

from data_curation.core import layout
from data_curation.core.errors import CurationError
from data_curation.response.harmonize.identity import resolve_samples
from data_curation.response.harmonize.structures import apply_structures
from data_curation.response.harmonize_response import (
    RESPONSE_X_COLUMN,
    VAR_INDEX_NAME,
    harmonize,
)

DRUG_MAPPING = pd.DataFrame(
    {
        "DRUG_ID": ["1001", "1002", "1003"],
        "DRUG_NAME": ["Doxorubicin", "Doxorubicin", "Unresolvable"],
        "pubchem_id": ["31703", "31703", "GDSC_1003"],
        "canonical_smiles": ["CC1", "CC1", ""],
    }
)


def _curves(
    obs_names: list[str],
    var_names: list[str],
    x: list[list[float]],
    *,
    signal_quality: list[list[float]] | None = None,
) -> ad.AnnData:
    layers = {}
    if signal_quality is not None:
        layers["signal_quality"] = np.array(signal_quality, dtype=np.float32)
    return ad.AnnData(
        X=np.array(x, dtype=np.float32),
        obs=pd.DataFrame(index=pd.Index(obs_names)),
        var=pd.DataFrame(index=pd.Index(var_names)),
        layers=layers,
    )


def _values(matrix) -> list[float]:
    """Flatten an AnnData matrix, whose static type is a wide union."""
    return np.asarray(matrix).ravel().tolist()


@pytest.fixture
def raw_dir(write_model_list) -> UPath:
    """A raw response directory whose model list carries two GDSC cell lines."""
    return write_model_list(
        [
            {"model_id": "SIDM00001", "model_name": "MEC-1", "RRID": "CVCL_1870"},
            {"model_id": "SIDM00002", "model_name": "HT-29", "RRID": "CVCL_0320"},
        ]
    ).parent


class TestAxisRename:
    def test_both_axes_become_canonical(self, raw_dir: UPath) -> None:
        curves = _curves(["SIDM00001"], ["1001"], [[6.5]])
        result = harmonize(curves, fmt="gdsc", model_list_path=raw_dir / "model_list.csv", drug_mapping=DRUG_MAPPING)
        assert list(result.obs_names) == ["CVCL_1870"]
        assert list(result.var_names) == ["31703"]

    def test_the_values_travel_with_their_labels(self, raw_dir: UPath) -> None:
        curves = _curves(["SIDM00002", "SIDM00001"], ["1001"], [[7.0], [6.0]])
        result = harmonize(curves, fmt="gdsc", model_list_path=raw_dir / "model_list.csv", drug_mapping=DRUG_MAPPING)
        assert _values(result[["CVCL_1870"], :].X) == [6.0]
        assert _values(result[["CVCL_0320"], :].X) == [7.0]

    def test_the_axes_are_named_for_their_identifier_space(self, raw_dir: UPath) -> None:
        curves = _curves(["SIDM00001"], ["1001"], [[6.5]])
        result = harmonize(curves, fmt="gdsc", model_list_path=raw_dir / "model_list.csv", drug_mapping=DRUG_MAPPING)
        assert result.obs_names.name == layout.SAMPLE_INDEX_NAME
        assert result.var_names.name == VAR_INDEX_NAME

    def test_the_meaning_of_x_is_recorded(self, raw_dir: UPath) -> None:
        curves = _curves(["SIDM00001"], ["1001"], [[6.5]])
        result = harmonize(curves, fmt="gdsc", model_list_path=raw_dir / "model_list.csv", drug_mapping=DRUG_MAPPING)
        assert result.uns["x_column"] == RESPONSE_X_COLUMN


class TestAnnotations:
    def test_the_cell_line_name_is_attached(self, raw_dir: UPath) -> None:
        curves = _curves(["SIDM00001"], ["1001"], [[6.5]])
        result = harmonize(curves, fmt="gdsc", model_list_path=raw_dir / "model_list.csv", drug_mapping=DRUG_MAPPING)
        assert list(result.obs["cell_line_name"]) == ["MEC-1"]

    def test_the_smiles_is_attached(self, raw_dir: UPath) -> None:
        curves = _curves(["SIDM00001"], ["1001"], [[6.5]])
        result = harmonize(curves, fmt="gdsc", model_list_path=raw_dir / "model_list.csv", drug_mapping=DRUG_MAPPING)
        assert list(result.var["canonical_smiles"]) == ["CC1"]

    def test_a_drug_without_a_smiles_gets_an_empty_string(self, raw_dir: UPath) -> None:
        curves = _curves(["SIDM00001"], ["1003"], [[6.5]])
        result = harmonize(curves, fmt="gdsc", model_list_path=raw_dir / "model_list.csv", drug_mapping=DRUG_MAPPING)
        assert list(result.var["canonical_smiles"]) == [""]


class TestUnresolved:
    def test_an_unmappable_sample_is_dropped(self, write_model_list) -> None:
        raw_dir = write_model_list(
            [{"model_id": f"SIDM{i:05d}", "model_name": f"L{i}", "RRID": f"CVCL_{i:04d}"} for i in range(20)]
        ).parent
        natives = [f"SIDM{i:05d}" for i in range(20)] + ["SIDM99999"]
        curves = _curves(natives, ["1001"], [[6.5]] * len(natives))
        result = harmonize(curves, fmt="gdsc", model_list_path=raw_dir / "model_list.csv", drug_mapping=DRUG_MAPPING)
        assert len(result.obs_names) == 20
        assert "SIDM99999" not in set(result.obs_names)

    def test_too_much_loss_is_an_error(self, raw_dir: UPath) -> None:
        curves = _curves(["SIDM00001", "SIDM99999"], ["1001"], [[6.5], [6.5]])
        with pytest.raises(CurationError, match="unmappable"):
            harmonize(
                curves,
                fmt="gdsc",
                model_list_path=raw_dir / "model_list.csv",
                drug_mapping=DRUG_MAPPING,
            )

    def test_a_pseudo_id_drug_survives(self, raw_dir: UPath) -> None:
        """An unresolvable drug keeps its pseudo-ID rather than losing its data."""
        curves = _curves(["SIDM00001"], ["1003"], [[6.5]])
        result = harmonize(curves, fmt="gdsc", model_list_path=raw_dir / "model_list.csv", drug_mapping=DRUG_MAPPING)
        assert list(result.var_names) == ["GDSC_1003"]


class TestDrugCollapse:
    """Repeat screens of one compound share a CID: 24 such pairs in GDSC1."""

    def test_two_native_drug_ids_collapse_onto_one_cid(self, raw_dir: UPath) -> None:
        curves = _curves(["SIDM00001"], ["1001", "1002"], [[6.0, 7.0]], signal_quality=[[0.1, 0.9]])
        result = harmonize(curves, fmt="gdsc", model_list_path=raw_dir / "model_list.csv", drug_mapping=DRUG_MAPPING)
        assert list(result.var_names) == ["31703"]

    def test_the_better_fit_wins_the_collapse(self, raw_dir: UPath) -> None:
        curves = _curves(["SIDM00001"], ["1001", "1002"], [[6.0, 7.0]], signal_quality=[[0.1, 0.9]])
        result = harmonize(curves, fmt="gdsc", model_list_path=raw_dir / "model_list.csv", drug_mapping=DRUG_MAPPING)
        assert _values(result.X) == [7.0]

    def test_the_collapse_is_logged(self, raw_dir: UPath, caplog) -> None:
        curves = _curves(["SIDM00001"], ["1001", "1002"], [[6.0, 7.0]], signal_quality=[[0.1, 0.9]])
        with caplog.at_level("INFO"):
            harmonize(
                curves,
                fmt="gdsc",
                model_list_path=raw_dir / "model_list.csv",
                drug_mapping=DRUG_MAPPING,
            )
        assert "31703" in caplog.text

    def test_an_injective_dataset_reports_no_collapse(self, raw_dir: UPath, caplog) -> None:
        curves = _curves(["SIDM00001"], ["1001"], [[6.5]])
        with caplog.at_level("INFO"):
            harmonize(
                curves,
                fmt="gdsc",
                model_list_path=raw_dir / "model_list.csv",
                drug_mapping=DRUG_MAPPING,
            )
        assert "injective" in caplog.text


class TestIdentityKeyedFormats:
    @pytest.mark.parametrize("fmt", ["bruna", "beataml", "standard"])
    def test_specimen_identifiers_pass_through_unchanged(self, raw_dir: UPath, fmt: str) -> None:
        mapping = pd.DataFrame(
            {
                "DRUG_ID": ["Venetoclax"],
                "DRUG_NAME": ["Venetoclax"],
                "pubchem_id": ["49846579"],
                "canonical_smiles": ["CC"],
            }
        )
        curves = _curves(["2100"], ["Venetoclax"], [[6.5]])
        result = harmonize(curves, fmt=fmt, model_list_path=raw_dir / "model_list.csv", drug_mapping=mapping)
        assert list(result.obs_names) == ["2100"]
        assert list(result.var_names) == ["49846579"]

    def test_specimens_get_no_cell_line_name(self, raw_dir: UPath) -> None:
        mapping = pd.DataFrame({"DRUG_ID": ["X"], "DRUG_NAME": ["X"], "pubchem_id": ["1"], "canonical_smiles": [""]})
        result = harmonize(
            _curves(["2100"], ["X"], [[6.5]]),
            fmt="beataml",
            model_list_path=raw_dir / "model_list.csv",
            drug_mapping=mapping,
        )
        assert "cell_line_name" not in result.obs.columns

    def test_standard_samples_keep_their_native_ids(self) -> None:
        """A standard dataset carries its own sample IDs, so no model list is needed."""
        identity = resolve_samples(pd.Index(["PDO-7", "PDO-8"]), "standard")
        assert identity.canonical.to_dict() == {"PDO-7": "PDO-7", "PDO-8": "PDO-8"}
        assert identity.cell_line_names == {}


class TestRemappedCids:
    """Two native drugs whose CIDs share an InChIKey, after ``apply_structures``."""

    MAPPING = pd.DataFrame(
        {
            "DRUG_ID": ["1001", "1002"],
            "DRUG_NAME": ["Sildenafil", "Sildenafil"],
            "pubchem_id": ["5212", "135398744"],
            "canonical_smiles": ["TAUTOMER_A", "TAUTOMER_B"],
        }
    )
    STRUCTURES = pd.DataFrame(
        {
            "pubchem_id": ["5212", "135398744"],
            "canonical_id": ["5212", "5212"],
            "canonical_smiles": ["TAUTOMER_A", "TAUTOMER_A"],
            "inchikey": ["K", "K"],
        }
    )

    def _harmonize(self, raw_dir: UPath, curves: ad.AnnData) -> ad.AnnData:
        mapping = apply_structures(self.MAPPING, self.STRUCTURES)
        return harmonize(curves, fmt="gdsc", model_list_path=raw_dir / "model_list.csv", drug_mapping=mapping)

    def test_the_natives_collapse_onto_the_lowest_cid_with_one_smiles(self, raw_dir: UPath) -> None:
        curves = _curves(["SIDM00001"], ["1001", "1002"], [[6.0, 7.0]], signal_quality=[[0.1, 0.9]])
        result = self._harmonize(raw_dir, curves)
        assert list(result.var_names) == ["5212"]
        assert list(result.var["canonical_smiles"]) == ["TAUTOMER_A"]
        assert list(result.var["native_ids"]) == ["1001;1002"]

    def test_the_collapse_rule_still_picks_the_better_fit(self, raw_dir: UPath) -> None:
        curves = _curves(["SIDM00001"], ["1001", "1002"], [[6.0, 7.0]], signal_quality=[[0.1, 0.9]])
        assert _values(self._harmonize(raw_dir, curves).X) == [7.0]


def _replicate_table(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """A native-keyed replicate-fits table as ``curate()`` writes it, one row per pair."""
    samples = [sample for sample, _ in pairs]
    drugs = [drug for _, drug in pairs]
    table = pd.DataFrame(
        {
            "sample_id": samples,
            "pubchem_id": drugs,
            "native_sample": samples,
            "native_drug": drugs,
            "status": ["fitted"] * len(pairs),
            "pEC50": np.arange(len(pairs), dtype=np.float32),
        }
    )
    return table.astype({column: "category" for column in ("sample_id", "pubchem_id", "native_sample", "native_drug")})


class TestReplicateFits:
    def _harmonize(self, raw_dir: UPath, curves: ad.AnnData) -> ad.AnnData:
        return harmonize(curves, fmt="gdsc", model_list_path=raw_dir / "model_list.csv", drug_mapping=DRUG_MAPPING)

    def test_the_table_is_rekeyed_onto_the_canonical_axes(self, raw_dir: UPath) -> None:
        curves = _curves(["SIDM00001", "SIDM00002"], ["1001"], [[6.0], [7.0]])
        curves.uns["replicate_fits"] = _replicate_table([("SIDM00001", "1001"), ("SIDM00002", "1001")])

        table = self._harmonize(raw_dir, curves).uns["replicate_fits"]

        assert table[["sample_id", "pubchem_id"]].astype(str).values.tolist() == [
            ["CVCL_1870", "31703"],
            ["CVCL_0320", "31703"],
        ]
        assert table["native_sample"].astype(str).tolist() == ["SIDM00001", "SIDM00002"]

    def test_natives_collapsed_in_the_matrix_keep_their_own_rows(self, raw_dir: UPath) -> None:
        curves = _curves(["SIDM00001"], ["1001", "1002"], [[6.0, 7.0]])
        curves.uns["replicate_fits"] = _replicate_table([("SIDM00001", "1001"), ("SIDM00001", "1002")])

        result = self._harmonize(raw_dir, curves)

        assert list(result.var_names) == ["31703"]
        assert result.uns["replicate_fits"]["native_drug"].astype(str).tolist() == ["1001", "1002"]

    def test_a_table_sample_missing_from_the_matrix_is_still_resolved(self, raw_dir: UPath) -> None:
        curves = _curves(["SIDM00001"], ["1001"], [[6.0]])
        curves.uns["replicate_fits"] = _replicate_table([("SIDM00001", "1001"), ("SIDM00002", "1001")])

        table = self._harmonize(raw_dir, curves).uns["replicate_fits"]

        assert table["sample_id"].astype(str).tolist() == ["CVCL_1870", "CVCL_0320"]

    def test_without_a_table_none_is_added(self, raw_dir: UPath) -> None:
        result = self._harmonize(raw_dir, _curves(["SIDM00001"], ["1001"], [[6.0]]))

        assert "replicate_fits" not in result.uns
