"""Drug response: raw measurements to fitted, harmonized curve metrics.

Curation runs in three stages because curve fitting is by far the most expensive
thing the pipeline does (~9 h for all datasets) and identity resolution is among
the most frequently changed:

- ``prepare_viability`` normalises each dataset's raw measurements into one long
  table, keyed by the dataset's *own* identifiers (``prepare`` subpackage).
- ``fit_curves`` fits those with CurveCurator, which treats the identifiers as
  opaque labels.
- ``harmonize_response`` renames both axes to Cellosaurus accessions and PubChem
  CIDs, collapsing any duplicates the rename creates (``harmonize`` subpackage).

Nothing in ``prepare`` resolves identity. That omission is the point of the
split: it keeps the fit's dependency surface to the viability table and the
fitting params, so a change to name resolution cannot invalidate a 9 h fit it
cannot affect.
"""
