"""GDSC plate-tag conventions shared by the drug mapping and response stages.

Every well in a GDSC raw file carries a ``TAG`` naming its role on the plate:
``L<n>-D<m>-S`` for the drug library arm, ``NC-*`` / ``B`` / ``DMSO`` for
negative and blank controls, ``PC*`` for positive-control compounds, and
``R*`` / ``T*`` for anchor and test arms.

Only the library arm becomes a dose-response measurement, so both stages have to
agree on which drugs that leaves. A drug screened solely as a plate control has
no curve to fit: GDSC2 runs Staurosporine (DRUG_ID 1034) as ``PC2-*`` on every
plate, and mapping it while the response stage drops it made the drug mapping
report one more drug than the response matrix holds.
"""

import pandas as pd

#: Prefix of the ``TAG`` values marking the screened drug library arm.
LIBRARY_TAG_PREFIX = "L"


def library_wells(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the rows measuring the drug library arm, excluding control wells."""
    return frame[frame["TAG"].astype(str).str.startswith(LIBRARY_TAG_PREFIX, na=False)]


def library_drug_ids(frame: pd.DataFrame) -> set:
    """Collect the DRUG_IDs appearing at least once in the drug library arm.

    A compound used both as a plate control and as a genuine library drug is
    kept, since it does contribute dose-response measurements.

    Args:
        frame: Raw GDSC well-level data with TAG and DRUG_ID columns.

    Returns:
        The DRUG_IDs that are genuinely screened.
    """
    return set(library_wells(frame)["DRUG_ID"].dropna().unique())
