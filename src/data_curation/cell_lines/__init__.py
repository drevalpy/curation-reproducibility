"""Cell-line identity (Cellosaurus, Cell Model Passports) and the sample metadata tables.

This package stays named for cell lines because nearly all of it is genuinely
cell-line specific: Cellosaurus and CMP identifier mapping. The one sample-generic
piece, the metadata tables behind ``mdata.obs``, sits here because it builds on
them. Three concerns live here, all of them needed by more than one other domain:

- ``sidm_mapping``: Cell Model Passports ``SIDM`` IDs and raw cell-line names to
  Cellosaurus accessions, plus the name normalisation everything else shares.
- ``cellosaurus``: the parsed Cellosaurus registry, used to rescue names no
  model list resolves and to flag contaminated or problematic lines.
- ``meta``: the sample metadata tables (``sample_id``, ``tissue`` and, for cell
  lines only, ``cellosaurus_id`` and ``cell_line_name``) that end up in
  ``mdata.obs``.

``omics``, ``response`` and ``assembly`` all import from here; this package
imports only from ``core``.
"""
