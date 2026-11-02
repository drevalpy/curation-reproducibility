"""Command for the final ``report`` stage.

A single top-level command rather than a sub-app: the stage has no datasets or
options to choose between. It is registered in :func:`data_curation.cli._mount`.
"""


def report() -> None:
    """Render the marimo summary notebook to data/08_reports/data_summary.html."""
    from data_curation.report.render import main

    main()
