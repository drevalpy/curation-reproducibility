"""Test that the packaged Bruna sample map still resolves.

The mapping is data shipped inside the package rather than a DVC artifact, so it
is located relative to ``__file__``. Moving the module that reads it changes how
that path resolves, and nothing else in the suite would notice: the failure only
appears when the stage runs against real data.
"""

import pandas as pd

from data_curation.omics.transcriptomics import bruna


class TestPackagedSampleMap:
    def test_the_file_is_where_the_module_looks_for_it(self) -> None:
        assert bruna._SAMPLE_MAP_PATH.is_file(), f"packaged sample map not found at {bruna._SAMPLE_MAP_PATH}"

    def test_it_lives_inside_the_omics_package(self) -> None:
        assert bruna._SAMPLE_MAP_PATH.parent.parent.name == "omics"

    def test_it_loads_as_a_usable_mapping(self) -> None:
        mapping = bruna._load_sample_map()

        assert mapping
        assert all(isinstance(key, str) and isinstance(value, str) for key, value in mapping.items())

    def test_it_has_the_expected_columns(self) -> None:
        frame = pd.read_csv(str(bruna._SAMPLE_MAP_PATH))

        assert list(frame.columns) == ["expression_sample", "response_sample"]
