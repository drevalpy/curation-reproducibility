"""Tests for writing the published MuData files."""

import anndata as ad
import h5py
import mudata as md
import numpy as np
from upath import UPath

from data_curation.assembly.write import write_mudata


def test_every_array_is_gzip_compressed_and_reads_back(tmp_dir: UPath) -> None:
    response = ad.AnnData(np.arange(100, dtype=np.float32).reshape(20, 5))
    path = tmp_dir / "c.h5mu"

    write_mudata(md.MuData({"response": response}), path)

    with h5py.File(str(path)) as f:
        assert f["mod/response/X"].compression == "gzip"
    np.testing.assert_array_equal(md.read_h5mu(str(path))["response"].X, response.X)
