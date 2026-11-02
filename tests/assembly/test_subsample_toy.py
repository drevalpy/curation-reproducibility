"""The toy subsample keeps every gene of the drevalpy gene lists, on small literal sources."""

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
from drevalpy.curation._replicate_fits import ID_COLUMNS, METRICS, to_schema
from drevalpy.data._replicate_fits import KEY

from data_curation.assembly import subsample_toy
from data_curation.assembly.toy_selection import required_genes, select_features
from data_curation.core.errors import CurationError

TOY_PARAMS = {
    "name": "TOYtest",
    "n_tissues": 2,
    "samples_per_tissue": 2,
    "n_drugs": 2,
    "n_features_per_modality": 3,
}
SAMPLES = [f"s{i}" for i in range(6)]
TISSUES = ["lung", "lung", "lung", "skin", "skin", "skin"]
DRUGS = ["111", "222", "333"]


def _omics(genes: list[str], seed: int) -> ad.AnnData:
    """Return a dense modality over ``SAMPLES`` whose genes have strictly rising variance."""
    rng = np.random.default_rng(seed)
    scales = np.arange(1, len(genes) + 1, dtype=np.float64)
    values = rng.normal(size=(len(SAMPLES), len(genes))) * scales
    return ad.AnnData(
        X=values,
        obs=pd.DataFrame(index=pd.Index(SAMPLES, name="sample_id")),
        var=pd.DataFrame(index=pd.Index(genes, name="gene_symbol")),
    )


def _response() -> ad.AnnData:
    values = np.arange(len(SAMPLES) * len(DRUGS), dtype=np.float64).reshape(len(SAMPLES), len(DRUGS))
    var = pd.DataFrame({"canonical_smiles": ["C", "CC", "CCC"]}, index=pd.Index(DRUGS))
    return ad.AnnData(X=values, obs=pd.DataFrame(index=pd.Index(SAMPLES, name="sample_id")), var=var)


def _replicate_fits() -> pd.DataFrame:
    """One fitted series per (sample, drug) pair of the source response."""
    pairs = [(sample, drug) for sample in SAMPLES for drug in DRUGS]
    table = pd.DataFrame({column: pd.Series(["x"] * len(pairs), dtype=object) for column in ID_COLUMNS})
    table["sample_id"] = [sample for sample, _ in pairs]
    table["pubchem_id"] = [drug for _, drug in pairs]
    table["replicate"] = 0
    table["biological_replicate"] = 0
    table["technical_replicate"] = 0
    table["n_doses"] = 8
    for metric in METRICS:
        table[metric] = 1.0
    return to_schema(table)


def _source(expression_genes: list[str]) -> md.MuData:
    response = _response()
    response.uns[KEY] = _replicate_fits()
    source = md.MuData({"gene_expression": _omics(expression_genes, 0), "response": response})
    source.obs = pd.DataFrame({"tissue": TISSUES}, index=pd.Index(SAMPLES, name="sample_id"))
    return source


# Ten genes with rising variance: the top three are G8-G10, so the list genes G1 and G2 are
# only there because the list asks for them.
GENES = [f"G{i}" for i in range(1, 11)]
LISTS = {"gene_expression": {"list_a": ["G2", "G1"], "list_b": ["G1", "G5"]}}


class TestRequiredGenes:
    def test_genes_of_all_lists_come_back_once_in_order(self) -> None:
        genes = required_genes("gene_expression", LISTS["gene_expression"], pd.Index(GENES))

        assert genes == ["G2", "G1", "G5"]

    def test_a_missing_gene_names_the_list_and_the_view(self) -> None:
        lists = {"list_a": ["G1", "NOPE"], "list_b": ["G2"]}

        with pytest.raises(CurationError, match=r"'gene_expression'.*'list_a' lacks 1 of 2 genes \(NOPE\)") as caught:
            required_genes("gene_expression", lists, pd.Index(GENES))

        assert "list_b" not in str(caught.value)

    def test_the_error_previews_only_the_first_missing_genes(self) -> None:
        lists = {"big": [f"X{i}" for i in range(9)]}

        with pytest.raises(CurationError, match=r"lacks 9 of 9 genes \(X0, X1, X2, X3, X4, \.\.\.\)"):
            required_genes("gene_expression", lists, pd.Index(GENES))

    def test_no_lists_require_nothing(self) -> None:
        assert required_genes("response", {}, pd.Index(GENES)) == []


class TestSelectFeatures:
    def test_required_features_join_the_top_variance_ones_in_source_order(self) -> None:
        adata = _omics(GENES, 0)

        chosen = select_features(adata, 3, ["G2", "G1"])

        assert chosen == ["G1", "G2", "G8", "G9", "G10"]

    def test_a_required_feature_that_is_also_top_variance_is_kept_once(self) -> None:
        chosen = select_features(_omics(GENES, 0), 3, ["G10"])

        assert chosen == ["G8", "G9", "G10"]


class TestBuildToy:
    def test_every_list_gene_of_the_source_ends_up_in_the_toy(self) -> None:
        toy = subsample_toy.build_toy(_source(GENES), TOY_PARAMS, LISTS)

        kept = set(toy.mod["gene_expression"].var_names)
        assert {"G1", "G2", "G5"} <= kept
        assert {"G8", "G9", "G10"} <= kept
        assert kept == {"G1", "G2", "G5", "G8", "G9", "G10"}

    def test_a_modality_without_lists_keeps_only_top_variance_features(self) -> None:
        toy = subsample_toy.build_toy(_source(GENES), TOY_PARAMS, {})

        assert list(toy.mod["gene_expression"].var_names) == ["G8", "G9", "G10"]

    def test_a_gene_the_source_lacks_fails_naming_the_list(self) -> None:
        source = _source([gene for gene in GENES if gene != "G5"])

        with pytest.raises(CurationError, match=r"'list_b' lacks 1 of 2 genes \(G5\)"):
            subsample_toy.build_toy(source, TOY_PARAMS, LISTS)

    def test_the_response_modality_is_not_touched_by_the_lists(self) -> None:
        toy = subsample_toy.build_toy(_source(GENES), TOY_PARAMS, LISTS)

        assert toy.mod["response"].n_vars == TOY_PARAMS["n_drugs"]

    def test_the_replicate_fits_cover_only_the_toy_samples_and_drugs(self) -> None:
        response = subsample_toy.build_toy(_source(GENES), TOY_PARAMS, {}).mod["response"]

        fits = response.uns[KEY]
        pairs = {(sample, drug) for sample in response.obs_names for drug in response.var_names}
        assert set(zip(fits["sample_id"], fits["pubchem_id"], strict=True)) == pairs
        assert len(fits) == len(pairs)
        assert set(fits["sample_id"].cat.categories) == set(response.obs_names)


class TestShippedGeneLists:
    @pytest.fixture
    def shipped(self) -> dict[str, dict[str, tuple[str, ...]]]:
        return subsample_toy.shipped_gene_lists()

    def test_the_shipped_lists_cover_the_views_the_assembled_datasets_have(self, shipped) -> None:
        assert {"gene_expression", "mutations", "copy_number_variation_gistic", "proteomics"} <= set(shipped)
        assert "landmark_genes" in shipped["gene_expression"]

    def test_the_toy_holds_all_shipped_expression_genes(self, shipped) -> None:
        union = list(dict.fromkeys(gene for genes in shipped["gene_expression"].values() for gene in genes))
        noise = [f"NOISE{i}" for i in range(20)]
        source = _source(noise + union)

        toy = subsample_toy.build_toy(source, TOY_PARAMS, {"gene_expression": shipped["gene_expression"]})

        assert set(union) <= set(toy.mod["gene_expression"].var_names)
