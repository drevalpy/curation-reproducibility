"""Tests that dvc.yaml and core/layout.py name the same paths.

``core/layout.py`` is deliberately not a DVC dependency, so nothing in DVC would
notice if the two drifted apart; these tests do instead.
"""

import re
from typing import Any

import pytest
import yaml
from upath import UPath

from data_curation.core import layout

_DATA_PATH = re.compile(r"(?<![\w/])data/[A-Za-z0-9_./${}-]+")
ITEM = "${item}"


def _load_pipeline() -> dict[str, Any]:
    """Parse ``dvc.yaml`` from the repository root."""
    with UPath("dvc.yaml").open() as handle:
        return yaml.safe_load(handle)


def _dvc_paths() -> set[str]:
    """Every ``data/...`` dep and out written in dvc.yaml, templates kept intact."""
    paths: set[str] = set()
    for spec in _load_pipeline()["stages"].values():
        body = spec.get("do") or spec
        for entry in [*(body.get("deps") or []), *(body.get("outs") or [])]:
            paths.update(_DATA_PATH.findall(str(entry)))
    return paths


def _layout_paths() -> dict[str, str]:
    """Every path the layout module produces, formatted with DVC's ``${item}``."""
    return {
        "CELLOSAURUS": str(layout.CELLOSAURUS),
        "RAW_DRUG_ONTOLOGY": f"{layout.RAW_DRUG_ONTOLOGY}/",
        "DRUG_MAPPING_DIR": f"{layout.DRUG_MAPPING_DIR}/",
        "DRUG_ONTOLOGY": str(layout.DRUG_ONTOLOGY),
        "REPORT_HTML": str(layout.REPORT_HTML),
        # Mutations, CNV and proteomics have no per-dataset raw directory: their
        # downloads are the shared Cell Model Passports files.
        **{f"raw_dir:{kind}": f"{layout.raw_dir(kind, ITEM)}/" for kind in ("response", "transcriptomics")},
        **{f"cmp_raw:{kind}": f"{layout.cmp_raw(kind)}/" for kind in layout.CMP_KINDS},
        **{f"omics:{kind}": str(layout.omics(kind, ITEM)) for kind in _OMICS_KINDS},
        "drug_mapping": str(layout.drug_mapping(ITEM)),
        "wells": str(layout.wells(ITEM)),
        "curves": str(layout.curves(ITEM)),
        "harmonized": str(layout.harmonized(ITEM)),
        "sample_meta": str(layout.sample_meta(ITEM)),
        "cmp_sample_meta": str(layout.cmp_sample_meta()),
        "assembled": str(layout.assembled(ITEM)),
    }


_OMICS_KINDS = ("transcriptomics", "mutations", "cnv", "proteomics")


class TestLayoutContract:
    def test_every_dvc_data_path_lives_under_a_layout_root(self) -> None:
        roots = tuple(f"{root}/" for root in layout.ROOTS)
        offenders = sorted(p for p in _dvc_paths() if not p.startswith(roots))
        assert not offenders, f"dvc.yaml paths outside the numbered layout roots: {offenders}"

    @pytest.mark.parametrize(("name", "path"), sorted(_layout_paths().items()))
    def test_every_layout_path_appears_in_dvc_yaml(self, name: str, path: str) -> None:
        assert path in _dvc_paths(), f"layout {name} -> {path} is not a dep or out of any stage in dvc.yaml"

    def test_modality_dirs_are_all_under_layout_roots(self) -> None:
        roots = set(layout.ROOTS)
        assert all(directory.parent in roots for directory in layout.MODALITY_DIRS.values())


class TestCmpLayout:
    def test_each_kind_has_its_own_directory_under_the_cmp_root(self) -> None:
        assert [layout.cmp_raw(kind) for kind in layout.CMP_KINDS] == [
            layout.RAW / "cmp" / kind for kind in layout.CMP_KINDS
        ]

    def test_an_unknown_kind_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown Cell Model Passports kind"):
            layout.cmp_raw("methylation")

    def test_the_shared_meta_is_distinct_from_any_dataset_table(self) -> None:
        assert layout.cmp_sample_meta() == layout.SAMPLE_META_DIR / "cmp.parquet"
        assert layout.cmp_sample_meta() != layout.sample_meta("GDSC2")

    def test_every_cmp_kind_is_declared_once_in_the_sources(self) -> None:
        from data_curation.core.config import source_config

        assert set(source_config("cell_model_passports")) == set(layout.CMP_KINDS)


def test_nextflow_launch_dirs_are_untracked_scratch() -> None:
    """The launch directory is not a DVC out, so it must sit in the git-ignored cache."""
    assert str(layout.nextflow_launch_dir("GDSC2")) == "data/cache/nextflow/GDSC2"
    assert "/data/cache/" in UPath(".gitignore").read_text().splitlines()
