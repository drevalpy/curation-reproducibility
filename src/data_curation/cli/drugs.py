"""Commands for the drug mapping and ontology annotation stages."""

from typing import Annotated

import typer

from data_curation.cli._datasets import DATASET_ARGUMENT, validate

app = typer.Typer(help="Drug identity and ontology annotation.", no_args_is_help=True)


@app.command("download-ontology")
def download_ontology() -> None:
    """Download the pinned drug classification source releases."""
    from data_curation.drugs.download_drug_ontology import main

    main()


@app.command("map")
def map_drugs(dataset: Annotated[str, DATASET_ARGUMENT]) -> None:
    """Resolve a dataset's drug identifiers to PubChem CIDs (rate-limited)."""
    validate(dataset, "response")

    from data_curation.drugs.map_drugs import main

    main(dataset)


@app.command("map-prism")
def map_prism(dataset: Annotated[str, DATASET_ARGUMENT]) -> None:
    """Resolve a PRISM release's compounds to PubChem CIDs, checking names against structures."""
    validate(dataset, "response")

    from data_curation.drugs.map_prism import main

    main(dataset)


@app.command("resolve-structures")
def resolve_structures() -> None:
    """Give every drug one canonical ID and one SMILES across all datasets (rate-limited)."""
    from data_curation.drugs.resolve_structures import main

    main()


@app.command("annotate")
def annotate() -> None:
    """Build the central drug ontology dictionary across all datasets."""
    from data_curation.drugs.annotate_drugs import main

    main()
