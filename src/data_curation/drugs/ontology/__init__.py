"""Readers for the public drug classification sources.

One module per source, each exposing an ``annotate``-style lookup over a pinned
release downloaded by ``download_drug_ontology``: ``pubchem`` (cross references
and CID to NCIt links), ``chebi`` (therapeutic roles), ``ncit`` (oncology drug
classes) and ``targets`` (Probes & Drugs target classification).
"""
