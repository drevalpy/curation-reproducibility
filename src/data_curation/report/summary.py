"""Tidy summary tables built from :class:`~data_curation.report.reading.DatasetSnapshot` objects.

Everything here is pure pandas over the small snapshots, so the notebook only
lays the tables out and draws them.
"""

import pandas as pd

from data_curation.report.reading import RESPONSE_KEY, DatasetSnapshot


def modality_membership(snapshots: list[DatasetSnapshot]) -> pd.DataFrame:
    """Return one row per (dataset, sample, modality) present.

    The response modality is listed for samples with at least one measured
    value; omics modalities are restricted to those same samples, so every
    row describes a sample that can actually be used for training.
    """
    frames = [
        pd.DataFrame({"dataset": s.name, "sample": s.response_samples, "modality": RESPONSE_KEY}) for s in snapshots
    ]
    for s in snapshots:
        for modality, lines in s.omics_samples.items():
            kept = lines.intersection(s.response_samples)
            frames.append(pd.DataFrame({"dataset": s.name, "sample": kept, "modality": modality}))
    if not frames:
        return pd.DataFrame(columns=pd.Index(["dataset", "sample", "modality"]))
    return pd.concat(frames, ignore_index=True)


def dataset_overview(snapshots: list[DatasetSnapshot]) -> pd.DataFrame:
    """Return one row per dataset with response size, density and replicate counts."""
    rows = []
    for s in snapshots:
        n_lines = len(s.response_samples)
        cells = n_lines * s.n_drugs
        rows.append(
            {
                "dataset": s.name,
                "samples": n_lines,
                "drugs": s.n_drugs,
                "measurements": s.n_measured,
                "density": s.n_measured / cells if cells else 0.0,
                "tissues": s.tissues.nunique(),
                "omics_modalities": len(s.omics_samples),
                "replicate_samples": len(s.replicates.samples),
                "replicate_pairs": s.replicates.n_pairs,
            }
        )
    return pd.DataFrame(rows).set_index("dataset")


def replicate_structure(snapshots: list[DatasetSnapshot]) -> pd.DataFrame:
    """Return per dataset how its replicated pairs split into biological and technical replicates.

    ``biological_pairs`` have fitted series in two or more biological replicates,
    ``technical_only_pairs`` in only one. The medians are NaN for a dataset without
    replicated pairs.
    """
    rows = [
        {
            "dataset": s.name,
            "replicate_pairs": s.replicates.n_pairs,
            "biological_pairs": s.replicates.n_biological_pairs,
            "technical_only_pairs": s.replicates.n_technical_only_pairs,
            "median_biological": s.replicates.median_biological,
            "median_technical": s.replicates.median_technical,
        }
        for s in snapshots
    ]
    columns = pd.Index(
        ["replicate_pairs", "biological_pairs", "technical_only_pairs", "median_biological", "median_technical"]
    )
    return pd.DataFrame(rows, columns=pd.Index(["dataset", *columns])).set_index("dataset")


def tissue_counts(snapshots: list[DatasetSnapshot]) -> pd.DataFrame:
    """Return the number of response samples per (dataset, tissue)."""
    frames = [s.tissues.value_counts().rename("samples").rename_axis("tissue").reset_index() for s in snapshots]
    for frame, s in zip(frames, snapshots, strict=True):
        frame.insert(0, "dataset", s.name)
    if not frames:
        return pd.DataFrame(columns=pd.Index(["dataset", "tissue", "samples"]))
    return pd.concat(frames, ignore_index=True)


def omics_coverage(snapshots: list[DatasetSnapshot]) -> pd.DataFrame:
    """Return one row per (dataset, omics modality) with covered samples and features.

    ``fraction`` is relative to the dataset's response samples. Modalities a
    dataset does not carry at all appear with zero coverage, so the table is a
    full dataset x modality grid.
    """
    modalities = sorted({m for s in snapshots for m in s.omics_samples})
    rows = []
    for s in snapshots:
        n_lines = len(s.response_samples)
        for modality in modalities:
            lines = s.omics_samples.get(modality, pd.Index([]))
            covered = len(lines.intersection(s.response_samples))
            rows.append(
                {
                    "dataset": s.name,
                    "modality": modality,
                    "samples": covered,
                    "fraction": covered / n_lines if n_lines else 0.0,
                    "features": s.omics_features.get(modality, 0),
                }
            )
    return pd.DataFrame(rows, columns=pd.Index(["dataset", "modality", "samples", "fraction", "features"]))


def union_coverage(membership: pd.DataFrame) -> pd.DataFrame:
    """Return, per modality, the unique samples covered across all datasets."""
    grouped = membership.groupby("modality")
    table = pd.DataFrame(
        {
            "unique_samples": grouped["sample"].nunique(),
            "datasets": grouped["dataset"].nunique(),
        }
    )
    return table.sort_values("unique_samples", ascending=False)


def dataset_indicators(snapshots: list[DatasetSnapshot], axis: str = "samples") -> pd.DataFrame:
    """Return a boolean entity x dataset table of response membership.

    This is the input format of :func:`upsetplot.from_indicators`.

    Args:
        snapshots: One snapshot per dataset.
        axis: ``"samples"`` or ``"drugs"``. Drugs are keyed by PubChem CID
            after harmonization, so the same compound matches across datasets.
    """
    members = {
        "samples": lambda s: s.response_samples,
        "drugs": lambda s: s.response_drugs,
    }[axis]
    names = [s.name for s in snapshots]
    index = pd.Index(sorted({x for s in snapshots for x in members(s)}), name=axis.rstrip("s"))
    return pd.DataFrame({s.name: index.isin(members(s)) for s in snapshots}, index=index)[names]


def dataset_multiplicity(snapshots: list[DatasetSnapshot]) -> pd.Series:
    """Return how many samples appear in exactly 1, 2, ... datasets."""
    counts = pd.Series([c for s in snapshots for c in s.response_samples]).value_counts()
    return counts.value_counts().sort_index().rename_axis("datasets").rename("samples")
