"""Molecular profiles: transcriptomics, methylation, mutations, CNV and proteomics.

Five modalities, each with a ``process_*`` module writing one AnnData per
dataset. Mutations, CNV and proteomics ride the shared Cell Model Passports
artifacts and so share ``cmp_matrix``. Transcriptomics and methylation come from
each screen's own release, as in drevalpy v1: GDSC's for GDSC1/2, the Broad's
CCLE release for CCLE and CTRPv1/v2, and per-dataset downloads for the rest; the
readers live in the ``transcriptomics`` and ``methylation`` subpackages, and
``samples`` keys their matrices by Cellosaurus accession.

``download`` handles raw-file retrieval for any modality, including response.
"""

#: Shared name for the raw-counts AnnData layer across all transcriptomics datasets.
#: Absent for PDX_Bruna (microarray).
COUNT_LAYER = "counts"
