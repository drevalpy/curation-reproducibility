"""Write an assembled MuData file the way every published dataset is written.

Every array is gzip-compressed (level 4): the files are downloaded far more often than
they are written, and gzip roughly halves them for about twice the read time. gzip
rather than ``lzf`` because every HDF5 build, and so every h5py and h5ad reader,
decodes it.
"""

import mudata as md
from upath import UPath

COMPRESSION = "gzip"
COMPRESSION_LEVEL = 4


def write_mudata(mdata: md.MuData, path: UPath) -> None:
    """Write *mdata* to *path* as a gzip-compressed ``.h5mu`` file.

    Args:
        mdata: The assembled dataset.
        path: Output ``.h5mu`` path.
    """
    mdata.write_h5mu(str(path), compression=COMPRESSION, compression_opts=COMPRESSION_LEVEL)
