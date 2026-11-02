"""Renaming fitted curves onto canonical identifiers, and reconciling the result.

Three steps, in the order ``harmonize_response`` applies them:

- ``identity``: map native sample and drug identifiers to Cellosaurus
  accessions and PubChem CIDs.
- ``collapse``: merge the rows that renaming maps together, by a documented
  per-cell rule rather than an arbitrary pick.
- ``published``: attach the upstream published metrics as extra layers.
"""
