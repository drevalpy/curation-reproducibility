"""Attach published (author-reported) curve metrics as extra AnnData layers.

Each dataset family ships its own official fits. These are stored alongside our
CurveCurator fits as ``*_published`` layers so the two can be compared without
re-deriving anything downstream.
"""

import logging
from collections.abc import Callable

import anndata as ad
import numpy as np
import pandas as pd
from upath import UPath

from data_curation.cell_lines.sidm_mapping import build_name_to_cellosaurus_map, build_sidm_to_cellosaurus_map
from data_curation.core.errors import CurationError
from data_curation.core.paths import find_file

logger = logging.getLogger(__name__)


def _drug_map(drug_mapping: pd.DataFrame) -> dict[str, str]:
    """Map the native ``DRUG_ID`` (as a string) to the canonical PubChem ID."""
    return dict(zip(drug_mapping["DRUG_ID"].astype(str), drug_mapping["pubchem_id"], strict=False))


def _fill_published_layer(adata: ad.AnnData, fitted: pd.DataFrame, metric: str, col: str) -> None:
    """Fill a single published metric as a layer aligned to the AnnData grid."""
    agg = fitted.groupby(["sample", "drug"])[col].mean().reset_index()
    layer = np.full(adata.shape, np.nan, dtype=np.float32)
    obs_idx = {name: i for i, name in enumerate(adata.obs_names)}
    var_idx = {name: i for i, name in enumerate(adata.var_names)}
    for _, row in agg.iterrows():
        oi = obs_idx.get(str(row["sample"]))
        vi = var_idx.get(str(row["drug"]))
        if oi is not None and vi is not None:
            layer[oi, vi] = row[col]
    adata.layers[metric] = layer


def _add_layers(adata: ad.AnnData, fitted: pd.DataFrame, metrics: list[tuple[str, str]]) -> None:
    """Add every requested (layer name, source column) pair that is present."""
    fitted = fitted.dropna(subset=["sample", "drug"])
    for metric, col in metrics:
        if col in fitted.columns:
            _fill_published_layer(adata, fitted, metric, col)
        else:
            logger.warning("Published metric column '%s' not found; skipping layer '%s'", col, metric)


def _add_gdsc(adata: ad.AnnData, raw_dir: UPath, model_list_path: UPath, drug_mapping: pd.DataFrame) -> None:
    """Add published LN_IC50 and AUC from the official GDSC fits."""
    fitted_file = raw_dir / "fitted_dose_response.xlsx"
    if not fitted_file.exists():
        return
    logger.info("Adding published LN_IC50/AUC layers from %s", fitted_file)
    fitted = pd.read_excel(str(fitted_file))

    sidm_to_cvcl = build_sidm_to_cellosaurus_map(model_list_path)
    if "SANGER_MODEL_ID" not in fitted.columns:
        msg = (
            f"{fitted_file} has no SANGER_MODEL_ID column, so its rows cannot be joined to the "
            f"Cellosaurus-keyed sample axis (columns: {sorted(fitted.columns)})"
        )
        raise CurationError(msg)
    fitted["sample"] = fitted["SANGER_MODEL_ID"].map(lambda s: sidm_to_cvcl[s][0] if s in sidm_to_cvcl else None)

    fitted["drug"] = fitted["DRUG_ID"].astype(str).map(_drug_map(drug_mapping))

    _add_layers(adata, fitted, [("LN_IC50_published", "LN_IC50"), ("AUC_published", "AUC")])


def _add_bruna(adata: ad.AnnData, raw_dir: UPath, _model_list_path: UPath, drug_mapping: pd.DataFrame) -> None:
    """Add the published AUC from the Bruna et al. supplementary fits.

    The paper's AUC is the area under the dose-*response* curve while ours is
    under the dose-*viability* curve, hence the ``1 - AUC`` inversion.

    ``iC50`` is deliberately not exported: the published values (up to ~3.7e4)
    are inconsistent with a screen dosed in the 1e-4 to 1 uM range and the unit
    is undocumented, so they are not comparable to our IC50.
    """
    try:
        fitted_file = find_file(raw_dir, "DrugResponsesAUCSamples*")
    except FileNotFoundError:
        logger.warning("No published Bruna fits found in %s; skipping published layers", raw_dir)
        return

    logger.info("Adding published AUC layer from %s", fitted_file)
    fitted = pd.read_csv(str(fitted_file), sep="\t")

    fitted["sample"] = fitted["ID"].astype(str)
    drug_map = _drug_map(drug_mapping)
    fitted["drug"] = fitted["Drug"].astype(str).map(drug_map)
    fitted["AUC_viability"] = 1 - fitted["AUC"]

    _add_layers(adata, fitted, [("AUC_published", "AUC_viability")])


def _add_beataml(adata: ad.AnnData, raw_dir: UPath, _model_list_path: UPath, drug_mapping: pd.DataFrame) -> None:
    """Add the published AUC and IC50 from the BeatAML2 probit fits.

    Values are stored as published; no rescaling is applied.
    """
    try:
        fitted_file = find_file(raw_dir, "*probit_curve_fits*")
    except FileNotFoundError:
        logger.warning("No published BeatAML2 fits found in %s; skipping published layers", raw_dir)
        return

    logger.info("Adding published AUC/IC50 layers from %s", fitted_file)
    fitted = pd.read_csv(str(fitted_file), sep="\t", low_memory=False)

    fitted["sample"] = fitted["dbgap_subject_id"].astype(str)
    drug_map = _drug_map(drug_mapping)
    fitted["drug"] = fitted["inhibitor"].astype(str).map(drug_map)

    _add_layers(adata, fitted, [("AUC_published", "auc"), ("IC50_published", "ic50")])


def _add_ccle(adata: ad.AnnData, raw_dir: UPath, model_list_path: UPath, drug_mapping: pd.DataFrame) -> None:
    """Add the published IC50 and AUC from the Barretina supplement's own fits.

    The supplement's ``ActArea`` is the *activity* area summed over its doses, so
    a fully resistant line scores 0 and a fully sensitive one scores the dose
    count. Dividing by the number of doses and inverting turns it into the area
    under the dose-*viability* curve on ``[0, 1]``, which is what our ``AUC``
    measures. ``EC50 (uM)`` is deliberately not exported: our ``EC50`` layer is
    derived from ``pEC50``, so exporting theirs under a ``*_published`` name next
    to it invites comparing two differently parameterised fits.
    """
    from data_curation.response.prepare.ccle import (
        DRUG_COLUMN,
        SAMPLE_COLUMN,
        SUPPLEMENT_PATTERN,
        read_activity_table,
    )

    try:
        raw_file = find_file(raw_dir, SUPPLEMENT_PATTERN)
    except FileNotFoundError:
        logger.warning("No CCLE supplement found in %s; skipping published layers", raw_dir)
        return

    logger.info("Adding published IC50/AUC layers from %s", raw_file)
    fitted = read_activity_table(raw_file)

    fitted["sample"] = _resolve_names(fitted[SAMPLE_COLUMN], model_list_path)
    drug_map = _drug_map(drug_mapping)
    fitted["drug"] = fitted[DRUG_COLUMN].astype(str).str.strip().map(drug_map)

    doses = fitted["Doses (uM)"].astype(str).str.split(",").str.len()
    fitted["AUC_viability"] = 1 - fitted["ActArea"] / doses

    _add_layers(adata, fitted, [("IC50_published", "IC50 (µM)"), ("AUC_published", "AUC_viability")])


def _resolve_names(names: pd.Series, model_list_path: UPath) -> pd.Series:
    """Resolve cell-line names the way ``harmonize_response`` does.

    The published layers are filled by ``(sample, drug)`` lookup against the
    already-harmonized axes, so the names have to travel the same route: Cell
    Model Passports first, Cellosaurus as the fallback.
    """
    from data_curation.cell_lines.cellosaurus import build_fallback_name_map, resolve_names

    primary = build_name_to_cellosaurus_map(model_list_path)
    resolved, _ = resolve_names(names.astype(str), primary, build_fallback_name_map())
    return resolved


# Every adder shares one signature so they can sit in this table; the ones that
# need no model list name it with a leading underscore.
PUBLISHED_LAYER_ADDERS: dict[str, Callable[[ad.AnnData, UPath, UPath, pd.DataFrame], None]] = {
    "gdsc": _add_gdsc,
    "ccle": _add_ccle,
    "bruna": _add_bruna,
    "beataml": _add_beataml,
}


def add_published_layers(
    adata: ad.AnnData, fmt: str, raw_dir: UPath, model_list_path: UPath, drug_mapping: pd.DataFrame
) -> None:
    """Add published metric layers for the given response format, if any exist."""
    adder = PUBLISHED_LAYER_ADDERS.get(fmt)
    if adder is not None:
        adder(adata, raw_dir, model_list_path, drug_mapping)
