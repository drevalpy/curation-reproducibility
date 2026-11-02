"""Commands for the omics download and processing stages.

Download and process are each one command taking the modality as an argument,
rather than the eight near-identical modules this replaces. The four download
modules were 6-line shims whose whole body was binding one string; that string is
now the ``MODALITY`` argument.

The Cell Model Passports files are identical for every dataset that uses them, so
they have their own dataset-free ``download-cmp`` command.
"""

from typing import Annotated

import typer

from data_curation.cli._datasets import DATASET_ARGUMENT, Modality, validate

app = typer.Typer(help="Transcriptomics, mutations, CNV and proteomics.", no_args_is_help=True)

MODALITY_ARGUMENT = typer.Argument(metavar="MODALITY", help="Omics modality to act on.")


@app.command("download")
def download(
    modality: Annotated[Modality, MODALITY_ARGUMENT],
    dataset: Annotated[str, DATASET_ARGUMENT],
) -> None:
    """Download a dataset's raw files for one modality."""
    validate(dataset, modality.value)

    from data_curation.core import layout
    from data_curation.omics.download import download_omics_data

    output_dir = layout.raw_dir(modality.value, dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    download_omics_data(dataset, modality.value, output_dir)


@app.command("download-cmp")
def download_cmp() -> None:
    """Download the Cell Model Passports files shared by all CMP datasets, once."""
    from data_curation.omics.download import download_cmp as fetch

    fetch()


@app.command("process")
def process(
    modality: Annotated[Modality, MODALITY_ARGUMENT],
    dataset: Annotated[str, DATASET_ARGUMENT],
) -> None:
    """Turn a dataset's raw files for one modality into an AnnData file."""
    validate(dataset, modality.value)

    # Spelled out rather than resolved with importlib so the type checker can see
    # the call, and so grepping a modality still finds the module that runs it.
    match modality:
        case Modality.TRANSCRIPTOMICS:
            from data_curation.omics.process_transcriptomics import main
        case Modality.MUTATIONS:
            from data_curation.omics.process_mutations import main
        case Modality.CNV:
            from data_curation.omics.process_cnv import main
        case Modality.PROTEOMICS:
            from data_curation.omics.process_proteomics import main

    main(dataset)
