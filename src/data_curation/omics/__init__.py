"""Molecular profiles: transcriptomics, methylation, mutations, CNV and proteomics.

Five modalities, each with a ``process_*`` module writing one AnnData per
dataset. Mutations, CNV and proteomics come from the shared Cell Model Passports
downloads. Transcriptomics and methylation come from each screen's own release, as
in drevalpy v1: GDSC's for GDSC1/2, the Broad's CCLE release for CCLE and
CTRPv1/v2, and per-dataset downloads for the rest; the readers live in the
``transcriptomics`` and ``methylation`` subpackages. Every processor keys its
matrix by Cellosaurus accession through ``samples.to_sample_anndata``.

``download`` handles raw-file retrieval for any modality, including response.
"""

#: Shared name for the raw-counts AnnData layer of the transcriptomics datasets that
#: have counts (CCLE and the standard format; not the microarrays or the DepMap release).
COUNT_LAYER = "counts"
