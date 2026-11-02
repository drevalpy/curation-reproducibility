"""Drug identity and classification.

Two concerns, in pipeline order:

- ``mapping``: resolve each dataset's native drug identifiers to PubChem CIDs,
  which is what merges the same compound across datasets and across repeat
  screens. Unresolvable drugs keep a dataset-prefixed pseudo-ID so their
  measurements are never dropped, and fixed-ratio combinations keep one by
  design, since a mixture has no single CID.
- ``ontology``: annotate those CIDs from public, externally maintained sources
  (Probes & Drugs, ChEBI, NCI Thesaurus, PubChem). No vocabulary is invented
  here.

``annotate_drugs`` is a single global stage rather than per dataset: the datasets
overlap heavily, so every distinct drug is annotated once into a central
dictionary that ``assembly`` merges onto each response modality.
"""
