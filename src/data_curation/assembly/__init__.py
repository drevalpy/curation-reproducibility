"""Combining the curated modalities into the MuData files drevalpy consumes.

``assemble_mudata`` joins one dataset's response, omics and sample metadata
into a single ``.h5mu``, merging the central drug ontology onto the response
modality's ``var``. ``subsample_toy`` then carves a tiny dataset out of an
assembled one, with the same modality names, obs columns and response layers, so
every drevalpy split mode can be exercised in seconds.

This is the top of the graph: it imports from the other domains and nothing
imports from it.
"""
