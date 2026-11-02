"""Molecular profiles: transcriptomics, mutations, CNV and proteomics.

Four modalities, each with a ``process_*`` module writing one AnnData per
dataset. Most of them ride the shared Cell Model Passports artifacts and so share
``cmp_matrix``; only transcriptomics has dataset-specific sources, in the
``transcriptomics`` subpackage, because PDX_Bruna ships a microarray matrix and
BeatAML2 an RNA-seq table that CMP does not cover.

``download`` handles raw-file retrieval for any modality, including response.
"""

#: Shared name for the raw-counts AnnData layer across all transcriptomics datasets.
#: Absent for PDX_Bruna (microarray).
COUNT_LAYER = "counts"
