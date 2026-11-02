"""Data summary of the assembled datasets, rendered to HTML by the ``report`` stage.

Run interactively with ``uv run marimo edit src/data_curation/report/notebook.py``
from the repository root; ``data-curation report`` exports it to static HTML.
All data logic lives in :mod:`data_curation.report.summary`, so the cells here
only lay tables and figures out.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Data curation summary")


@app.cell
def _():
    from datetime import datetime, timezone

    import marimo as mo
    import plotly.express as px

    from data_curation.report import summary
    from data_curation.report.reading import discover_datasets, read_snapshots

    return datetime, discover_datasets, mo, px, read_snapshots, summary, timezone


@app.cell
def _(discover_datasets, read_snapshots, summary):
    snapshots = read_snapshots(discover_datasets())
    overview = summary.dataset_overview(snapshots)
    membership = summary.modality_membership(snapshots)
    coverage = summary.omics_coverage(snapshots)
    return coverage, membership, overview, snapshots


@app.cell
def _(datetime, mo, snapshots, timezone):
    _names = ", ".join(f"`{s.name}`" for s in snapshots)
    mo.md(f"""
    # Data curation summary

    Built {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC from the {len(snapshots)} assembled
    datasets in `data/07_assembled/`: {_names}.

    A sample counts towards a dataset when it has at least one measured
    drug response; omics coverage is relative to those samples.
    """)
    return


@app.cell
def _(mo, overview):
    _table = overview.style.format({"density": "{:.1%}", "measurements": "{:,}", "replicate_pairs": "{:,}"})
    mo.vstack([mo.md("## Samples per dataset"), mo.Html(_table.to_html())])
    return


@app.cell
def _(px, snapshots, summary):
    _tissues = summary.tissue_counts(snapshots)
    px.bar(
        _tissues,
        x="dataset",
        y="samples",
        color="tissue",
        title="Response samples per dataset, by tissue",
        labels={"samples": "samples"},
        height=550,
    )
    return


@app.cell
def _(membership, mo, px, snapshots, summary):
    _multiplicity = summary.dataset_multiplicity(snapshots).reset_index()
    _total = membership.loc[membership["modality"] == "response", "sample"].nunique()
    _single = int(_multiplicity.loc[_multiplicity["datasets"] == 1, "samples"].sum())
    _drugs = summary.dataset_indicators(snapshots, "drugs")
    _drugs_single = int((_drugs.sum(axis=1) == 1).sum())
    mo.vstack(
        [
            mo.md(f"""
    ## Total

    **{_total:,}** unique samples have drug response data across all datasets;
    {_single:,} of them appear in only one dataset. They were screened against
    **{len(_drugs):,}** unique drugs, {_drugs_single:,} of which appear in only one dataset.
    """),
            px.bar(
                _multiplicity,
                x="datasets",
                y="samples",
                title="Number of datasets each sample appears in",
                labels={"datasets": "datasets", "samples": "samples"},
            ),
        ]
    )
    return


@app.cell
def _(coverage, membership, mo, px, summary):
    _grid = coverage.pivot(index="dataset", columns="modality", values="fraction")
    _counts = coverage.pivot(index="dataset", columns="modality", values="samples")
    _heatmap = px.imshow(
        _grid,
        zmin=0,
        zmax=1,
        color_continuous_scale="Blues",
        aspect="auto",
        title="Share of response samples with each omics modality",
    )
    _heatmap.update_traces(text=_counts.to_numpy(), texttemplate="%{text}<br>(%{z:.0%})")
    _union = summary.union_coverage(membership)
    _features = coverage.pivot(index="dataset", columns="modality", values="features")
    mo.vstack(
        [
            mo.md("## Omics coverage"),
            _heatmap,
            mo.md("### Unique samples per modality across all datasets"),
            mo.Html(_union.to_html()),
            mo.md("### Features per modality"),
            mo.Html(_features.to_html()),
        ]
    )
    return


@app.cell
def _(mo, snapshots, summary):
    from data_curation.report.plots import upset_figure

    _min_size = 10
    mo.vstack(
        [
            mo.md(f"""
    ## Dataset overlap

    Each column is one exact combination of datasets; the bar is the number of
    samples (or drugs) measured in exactly those datasets, and the bars on the
    left are each dataset's total. Intersections smaller than {_min_size} are hidden.

    ### Samples
    """),
            upset_figure(
                summary.dataset_indicators(snapshots, "samples"),
                "Response samples shared between datasets",
                _min_size,
            ),
            mo.md("""
    ### Drugs

    Drugs are matched on their PubChem CID, the identifier every dataset is
    harmonized to, so the same compound counts once however each screen named it.
    """),
            upset_figure(
                summary.dataset_indicators(snapshots, "drugs"),
                "Response drugs shared between datasets",
                _min_size,
            ),
        ]
    )
    return


@app.cell
def _(mo, overview, snapshots, summary):
    _replicates = overview[["samples", "replicate_samples", "replicate_pairs", "measurements"]].copy()
    _replicates["pair_share"] = _replicates["replicate_pairs"] / _replicates["measurements"].where(
        _replicates["measurements"] > 0
    )
    _missing = _replicates.index[_replicates["replicate_pairs"] == 0].tolist()
    _note = (
        f"No replicate pairs at all in: {', '.join(f'`{n}`' for n in _missing)}."
        if _missing
        else "Every dataset has replicate pairs."
    )
    mo.vstack(
        [
            mo.md(f"""
    ## Replicate fits

    `response.uns["replicate_fits"]` holds one curve fit per replicate dose series.
    A (sample, drug) pair counts as replicated when at least two of its series were
    fitted; `pair_share` is relative to the measured pairs. {_note}
    """),
            mo.Html(
                _replicates.style.format(
                    {"pair_share": "{:.1%}", "replicate_pairs": "{:,}", "measurements": "{:,}"}, na_rep="-"
                ).to_html()
            ),
            mo.md("""
    Each replicated pair's fitted series are grouped into biological replicates
    (`biological_replicate`: one independently seeded experiment each) and the
    technical replicates within them (`technical_replicate`: parallel plates or wells
    of one experiment). `biological_pairs` have two or more biological replicates,
    `technical_only_pairs` only one; the medians are per replicated pair and per
    biological replicate.
    """),
            mo.Html(
                summary.replicate_structure(snapshots)
                .style.format(
                    {
                        "replicate_pairs": "{:,}",
                        "biological_pairs": "{:,}",
                        "technical_only_pairs": "{:,}",
                        "median_biological": "{:g}",
                        "median_technical": "{:g}",
                    },
                    na_rep="-",
                )
                .to_html()
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
