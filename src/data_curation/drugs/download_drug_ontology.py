"""Download the public drug classification sources used to annotate drugs.

Retrieves three kinds of input into ``data/01_raw/drug_ontology/``:

- Probes & Drugs compound and target exports, which carry the ChEMBL protein
  family classification flattened into five levels.
- ChEBI flat files, which provide the ``has_role`` and ``is_a`` relations.
- The PubChem CID -> NCI Thesaurus link table, paged from PubChem and cached as
  JSON alongside its retrieval date, since PubChem offers no versioned URL.
"""

import json
import logging
from datetime import UTC, datetime

from data_curation.core import layout
from data_curation.core.config import source_config
from data_curation.core.download import download_file_list
from data_curation.core.errors import CurationError
from data_curation.drugs.ontology.pubchem import fetch_cid_to_ncit

logger = logging.getLogger(__name__)

RAW_DIR = layout.RAW_DRUG_ONTOLOGY
NCIT_PATH = RAW_DIR / "cid2ncit.json"


def main() -> None:
    """Download all drug ontology source files.

    Raises:
        CurationError: If PubChem returns no CID -> NCIt links.
    """
    onto_params = source_config("drug_ontology")

    download_file_list(onto_params.get("files") or [], RAW_DIR)

    mapping = fetch_cid_to_ncit(onto_params["ncit_heading"])
    if not mapping:
        msg = "PubChem returned no CID -> NCIt links; refusing to write an empty mapping"
        raise CurationError(msg)

    NCIT_PATH.write_text(
        json.dumps(
            {
                "retrieved": datetime.now(UTC).date().isoformat(),
                "heading": onto_params["ncit_heading"],
                "cid_to_ncit": mapping,
            },
            indent=0,
            sort_keys=True,
        )
    )
    logger.info("Wrote %d CID -> NCIt entries to %s", len(mapping), NCIT_PATH)
