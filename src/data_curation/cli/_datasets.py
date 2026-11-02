"""Dataset arguments: validation, shell completion and the modality enum.

Dataset names are checked against ``config/sources.yaml`` before a command
runs, so a typo fails in milliseconds with the valid names listed rather than
several minutes into a stage with a ``KeyError``. Validation is scoped to the
modality where one applies: ``PDX_Bruna`` has no CNV block, so ``omics process
cnv PDX_Bruna`` is rejected outright.

Everything here tolerates a missing or malformed config file by falling back
to accepting the value, so ``--help`` and completion still work outside a
checkout. The stage itself raises the definitive error in that case.
"""

from enum import StrEnum

import typer

from data_curation.core.config import dataset_names


class Modality(StrEnum):
    """An omics modality, which is also its sources.yaml key and raw subdirectory."""

    TRANSCRIPTOMICS = "transcriptomics"
    MUTATIONS = "mutations"
    CNV = "cnv"
    PROTEOMICS = "proteomics"


def complete_dataset(incomplete: str) -> list[str]:
    """Complete a dataset name against every configured dataset."""
    return [name for name in dataset_names() if name.startswith(incomplete)]


def validate(dataset: str, modality: str | None = None) -> str:
    """Check a dataset name against ``config/sources.yaml``.

    Args:
        dataset: Dataset name as given on the command line.
        modality: Restrict to the datasets configured for this modality, e.g.
            ``"response"``; None to allow any configured dataset.

    Returns:
        *dataset*, unchanged.

    Raises:
        typer.BadParameter: If the name is not among the configured datasets.
    """
    known = dataset_names(modality)
    if not known or dataset in known:
        return dataset

    scope = f" for {modality}" if modality else ""
    msg = f"Unknown dataset '{dataset}'{scope}. Configured datasets{scope}: {', '.join(known)}"
    raise typer.BadParameter(msg)


DATASET_ARGUMENT = typer.Argument(
    metavar="DATASET",
    help="Dataset name as configured in config/sources.yaml.",
    autocompletion=complete_dataset,
)
