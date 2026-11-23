"""Per-dataset readers turning raw response files into well-level measurements.

Every module here exposes one ``prepare_*`` function with the same contract: read a
dataset's raw directory (GDSC also takes its two control tags) and return a
:class:`~data_curation.response.prepare.wells.Wells` - the normalized
``concentration`` and ``intensity`` per well, keyed by the dataset's own
identifiers, together with the module's ``SeriesSpec`` naming the columns that
group those wells into dose series (one replicate each).
:func:`~data_curation.response.prepare.wells.to_wells` renames that into the well
table ``drevalpy.curation.curate()`` reads. Identity resolution happens later, in
``response.harmonize``.
"""
