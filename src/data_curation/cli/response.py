"""Commands for the drug response curation stages.

The command order matches the pipeline order: download, prepare-viability,
fit-curves, harmonize.
"""

from typing import Annotated

import typer

from data_curation.cli._datasets import DATASET_ARGUMENT, validate

app = typer.Typer(help="Drug response curation, from raw plates to fitted curves.", no_args_is_help=True)


@app.command("download")
def download(dataset: Annotated[str, DATASET_ARGUMENT]) -> None:
    """Download a dataset's raw response files."""
    validate(dataset, "response")

    from data_curation.core import layout
    from data_curation.omics.download import download_omics_data

    output_dir = layout.raw_dir("response", dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    download_omics_data(dataset, "response", output_dir)


@app.command("prepare-viability")
def prepare_viability(dataset: Annotated[str, DATASET_ARGUMENT]) -> None:
    """Normalise raw measurements into a well table (minutes)."""
    validate(dataset, "response")

    from data_curation.response.prepare_viability import main

    main(dataset)


@app.command("fit-curves")
def fit_curves(dataset: Annotated[str, DATASET_ARGUMENT]) -> None:
    """Fit pooled and per-replicate dose-response curves with CurveCurator (expensive: hours)."""
    validate(dataset, "response")

    from data_curation.response.fit_curves import main

    main(dataset)


@app.command("harmonize")
def harmonize(dataset: Annotated[str, DATASET_ARGUMENT]) -> None:
    """Rename fitted curves onto canonical identifiers and collapse duplicates."""
    validate(dataset, "response")

    from data_curation.response.harmonize_response import main

    main(dataset)
