"""Number a preparer's wells into series the way ``fit_curves`` does, for the preparer tests."""

from drevalpy.curation._series import Series, number_series

from data_curation.response.prepare.wells import Wells, metadata_columns, to_wells


def number(wells: Wells, *, min_doses: int = 1) -> Series:
    """Run a preparer's output through ``to_wells`` and curate's series numbering."""
    frame = to_wells(wells)
    return number_series(frame, metadata=metadata_columns(frame), min_doses=min_doses)
