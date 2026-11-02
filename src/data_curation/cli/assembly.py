"""Commands for the MuData assembly stages."""

from typing import Annotated

import typer

from data_curation.cli._datasets import DATASET_ARGUMENT, validate

app = typer.Typer(help="Assemble curated modalities into MuData files.", no_args_is_help=True)


@app.command("mudata")
def mudata(dataset: Annotated[str, DATASET_ARGUMENT]) -> None:
    """Assemble one dataset's modalities into a single .h5mu file."""
    validate(dataset)

    from data_curation.assembly.assemble_mudata import main

    main(dataset)


@app.command("check-drug-structures")
def check_drug_structures() -> None:
    """Check one SMILES per drug ID and one drug ID per InChIKey across the assembled files."""
    from data_curation.assembly.check_drug_structures import main

    main()


@app.command("subsample-toy")
def subsample_toy() -> None:
    """Carve the tiny toy dataset out of an assembled one."""
    from data_curation.assembly.subsample_toy import main

    main()


@app.command("record-toy-gene-lists")
def record_toy_gene_lists() -> None:
    """Record the drevalpy gene lists the toy dataset must hold, for DVC."""
    from data_curation.assembly.record_toy_gene_lists import main

    main()
