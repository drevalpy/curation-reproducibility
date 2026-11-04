"""Tests that a standard-format sample table flows through MuData assembly's global obs."""

from pathlib import Path

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
from upath import UPath

from data_curation.assembly.assemble_mudata import _build_global_obs
from data_curation.cell_lines.meta import read_sample_meta
from data_curation.core import layout, standard

md.set_options(pull_on_update=False)


def _standard_table(tmp_path: Path) -> UPath:
    frame = standard.validate(
        "samples",
        pd.DataFrame(
            {
                "sample_id": ["S1", "S2"],
                "tissue": ["Blood", "Bone Marrow"],
                "patient_id": [7.0, None],
                "sex": ["F", "M"],
            }
        ),
    )
    path = UPath(tmp_path / "Inhouse.parquet")
    frame.to_parquet(str(path), index=False)
    return path


def _anndata(obs_names: list[str]) -> ad.AnnData:
    return ad.AnnData(
        X=np.zeros((len(obs_names), 1), dtype=np.float32),
        obs=pd.DataFrame(index=pd.Index(obs_names, name=layout.SAMPLE_INDEX_NAME)),
        var=pd.DataFrame(index=pd.Index(["f0"])),
    )


def test_a_standard_table_is_read_without_cell_line_columns(tmp_path: Path) -> None:
    meta = read_sample_meta(_standard_table(tmp_path))

    assert list(meta.columns) == ["tissue", "patient_id", "sex"]
    assert meta.loc["S1", "patient_id"] == "7"


def test_a_standard_table_becomes_the_global_obs(tmp_path: Path) -> None:
    mdata = md.MuData({"response": _anndata(["S1", "S2", "S3"])})
    _build_global_obs(mdata, _standard_table(tmp_path))

    assert list(mdata.obs.columns) == ["tissue", "patient_id", "sex"]
    assert mdata.obs.to_dict("list") == {
        "tissue": ["Blood", "Bone Marrow", ""],
        "patient_id": ["7", "", ""],
        "sex": ["F", "M", ""],
    }

    out = tmp_path / "out.h5mu"
    mdata.write(out)
    reread = md.read_h5mu(out)
    assert reread.obs.loc["S2", "tissue"] == "Bone Marrow"
    assert reread.obs.index.name == layout.SAMPLE_INDEX_NAME
