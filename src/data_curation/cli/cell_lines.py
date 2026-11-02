"""Commands for the cell-line identity and sample metadata stages."""

from typing import Annotated

import typer

from data_curation.cli._datasets import DATASET_ARGUMENT, validate

app = typer.Typer(help="Cell-line identity and sample metadata.", no_args_is_help=True)


@app.command("download-cellosaurus")
def download_cellosaurus() -> None:
    """Download the pinned Cellosaurus release and write it to Parquet."""
    from data_curation.cell_lines.download_cellosaurus import download_cellosaurus as fetch
    from data_curation.core import layout

    output_path = layout.CELLOSAURUS
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fetch(output_path)


@app.command("process-cmp-meta")
def process_cmp_meta() -> None:
    """Build the sample metadata table shared by all Cell Model Passports datasets (cell lines)."""
    from data_curation.cell_lines.process_sample_meta import main_cmp

    main_cmp()


@app.command("process-meta")
def process_meta(dataset: Annotated[str, DATASET_ARGUMENT]) -> None:
    """Build the sample metadata table of a dataset with a constant tissue (no CMP entry)."""
    validate(dataset)

    from data_curation.cell_lines.process_sample_meta import main

    main(dataset)
