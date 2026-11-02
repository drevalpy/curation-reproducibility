"""Collapse several native identifiers onto one canonical identifier.

Fitting on native identifiers means the rename to canonical ones can produce
duplicate ``obs``/``var`` entries: several native IDs legitimately denote the same
sample or compound. Measured over the curated datasets, samples are nearly
injective (CTRPv2 has exactly one collision, ``RH30``/``SJRH30`` -> ``CVCL_0041``;
GDSC is injective) but drugs are not - **GDSC1 collapses 402 native ``DRUG_ID``s
onto 378 PubChem CIDs and GDSC2 294 onto 285**, all repeat screens of the same
compound (Doxorubicin, Docetaxel, Selumetinib, ...).

The split does not create this. Those repeat screens use different dose ranges,
and CurveCurator groups by dose range, so they were already separate fitted rows
sharing one ``Name``; ``_pivot_metric``'s ``aggfunc="first"`` in drevalpy then
picked one arbitrarily. What changes is that the choice is now explicit.

**The rule.** For every canonical (sample, drug) cell, among the native
candidates take:

1. a fitted value over a missing one,
2. then the higher ``signal_quality``,
3. then the lower ``RMSE``,
4. then the lexicographically smaller native identifier.

Step 4 exists only so the result cannot depend on row order; steps 1-3 are the
fit-quality preference. Every collapse is logged with the per-native cell counts,
and the surviving axis records its sources in ``native_ids``.
"""

import logging
from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Layers the rule ranks by. An absent layer degrades to the next criterion.
QUALITY_LAYER = "signal_quality"
ERROR_LAYER = "RMSE"

#: Column recording which native identifiers a canonical entry came from.
NATIVE_IDS_COLUMN = "native_ids"

_NATIVE_ID_SEPARATOR = ";"


@dataclass(frozen=True)
class Collapse:
    """One canonical identifier that several native identifiers mapped onto."""

    canonical: str
    natives: tuple[str, ...]
    cells_won: tuple[int, ...]


def collapse_axis(adata: ad.AnnData, canonical: pd.Series, *, axis: int) -> tuple[ad.AnnData, list[Collapse]]:
    """Rename one axis to canonical labels, resolving duplicates by fit quality.

    Args:
        adata: Curve metrics keyed by native identifiers.
        canonical: Canonical label per axis entry, indexed by native identifier
            and covering every entry of the axis.
        axis: ``0`` for ``obs``, ``1`` for ``var``.

    Returns:
        The AnnData with a deduplicated, sorted canonical axis, and one
        :class:`Collapse` per canonical label that absorbed several natives.
    """
    if axis == 1:
        transposed, collapses = collapse_axis(adata.T, canonical, axis=0)
        return transposed.T, collapses
    return _reduce_groups(adata, _grouped_natives(canonical))


def _grouped_natives(canonical: pd.Series) -> dict[str, list[str]]:
    """Map each canonical label to its native identifiers, in sorted order.

    Sorting is what makes criterion 4 of the rule reproducible.
    """
    groups: dict[str, list[str]] = {}
    for native, target in canonical.items():
        groups.setdefault(str(target), []).append(str(native))
    return {target: sorted(natives) for target, natives in groups.items()}


def _reduce_groups(adata: ad.AnnData, groups: dict[str, list[str]]) -> tuple[ad.AnnData, list[Collapse]]:
    """Build one row per canonical label by reducing its native rows pairwise."""
    matrices = _as_matrices(adata)
    positions = {str(name): i for i, name in enumerate(adata.obs_names)}
    labels = sorted(groups)

    reduced = {name: np.empty((len(labels), adata.n_vars)) for name in matrices}
    collapses: list[Collapse] = []
    columns = np.arange(adata.n_vars)

    for row, label in enumerate(labels):
        natives = groups[label]
        rows = np.array([positions[native] for native in natives])
        winners = _select_winners(matrices, rows)
        for name, matrix in matrices.items():
            reduced[name][row] = matrix[rows[winners], columns]
        collapse = _describe_collapse(label, natives, winners)
        if collapse is not None:
            collapses.append(collapse)

    return _rebuild(adata, labels=labels, groups=groups, reduced=reduced), collapses


def _as_matrices(adata: ad.AnnData) -> dict[str, np.ndarray]:
    """Collect ``X`` and every real layer as float64 arrays.

    anndata 0.13 lists a ``None`` key in ``layers`` as an alias for ``X``.
    Stringifying the keys while copying turned that alias into a real layer
    literally named ``"None"`` - a second copy of ``X`` in the written file.
    """
    matrices: dict[str, np.ndarray] = {"X": np.asarray(adata.X, dtype=np.float64)}
    for name, layer in adata.layers.items():
        if name is not None:
            matrices[str(name)] = np.asarray(layer, dtype=np.float64)
    return matrices


def _describe_collapse(label: str, natives: list[str], winners: np.ndarray) -> Collapse | None:
    """Describe a genuine collapse, or return ``None`` for a plain rename."""
    if len(natives) == 1:
        return None
    counts = tuple(int((winners == i).sum()) for i in range(len(natives)))
    return Collapse(canonical=label, natives=tuple(natives), cells_won=counts)


def _rebuild(
    adata: ad.AnnData,
    *,
    labels: list[str],
    groups: dict[str, list[str]],
    reduced: dict[str, np.ndarray],
) -> ad.AnnData:
    """Assemble the reduced matrices into an AnnData on the canonical axis."""
    obs = pd.DataFrame(
        {NATIVE_IDS_COLUMN: [_NATIVE_ID_SEPARATOR.join(groups[label]) for label in labels]},
        index=pd.Index(labels),
    )
    layers = {name: matrix.astype(np.float32) for name, matrix in reduced.items() if name != "X"}
    return ad.AnnData(
        X=reduced["X"].astype(np.float32),
        obs=obs,
        var=pd.DataFrame(adata.var).copy(),
        uns=dict(adata.uns),
        layers=layers,
    )


def _select_winners(matrices: dict[str, np.ndarray], rows: np.ndarray) -> np.ndarray:
    """Return, per column, the index into *rows* that the rule selects."""
    x = matrices["X"]
    columns = np.arange(x.shape[1])
    winners = np.zeros(x.shape[1], dtype=np.intp)

    for candidate in range(1, len(rows)):
        incumbent = rows[winners]
        wins = _candidate_wins(
            valid=(np.isfinite(x[rows[candidate]]), np.isfinite(x[incumbent, columns])),
            quality=_pair(matrices.get(QUALITY_LAYER), rows[candidate], incumbent),
            error=_pair(matrices.get(ERROR_LAYER), rows[candidate], incumbent),
        )
        winners = np.where(wins, candidate, winners)
    return winners


def _pair(
    matrix: np.ndarray | None, candidate_row: int, incumbent_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract the candidate and incumbent vectors of a ranking layer."""
    if matrix is None:
        return None
    return matrix[candidate_row], matrix[incumbent_rows, np.arange(matrix.shape[1])]


def _candidate_wins(
    *,
    valid: tuple[np.ndarray, np.ndarray],
    quality: tuple[np.ndarray, np.ndarray] | None,
    error: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    """Apply the ranking rule elementwise, returning where the candidate wins.

    Ties fall through to the incumbent, which is the lexicographically smaller
    native identifier because :func:`_grouped_natives` sorts them.
    """
    candidate_valid, incumbent_valid = valid
    wins = candidate_valid & ~incumbent_valid
    undecided = candidate_valid & incumbent_valid

    if quality is not None:
        candidate_quality = np.nan_to_num(quality[0], nan=-np.inf)
        incumbent_quality = np.nan_to_num(quality[1], nan=-np.inf)
        wins |= undecided & (candidate_quality > incumbent_quality)
        undecided = undecided & (candidate_quality == incumbent_quality)

    if error is not None:
        candidate_error = np.nan_to_num(error[0], nan=np.inf)
        incumbent_error = np.nan_to_num(error[1], nan=np.inf)
        wins |= undecided & (candidate_error < incumbent_error)

    return wins


def log_collapses(collapses: list[Collapse], *, axis_name: str) -> None:
    """Report every collapse, so a silent merge cannot happen unnoticed."""
    if not collapses:
        logger.info("%s identifiers are injective; no duplicates to collapse", axis_name)
        return

    absorbed = sum(len(collapse.natives) for collapse in collapses)
    logger.info(
        "Collapsed %d native %s identifiers onto %d canonical ones by fit quality",
        absorbed,
        axis_name,
        len(collapses),
    )
    for collapse in collapses:
        detail = ", ".join(
            f"{native} won {count}" for native, count in zip(collapse.natives, collapse.cells_won, strict=True)
        )
        logger.info("  %s <- %s (%s)", collapse.canonical, list(collapse.natives), detail)
