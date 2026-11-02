"""Shared parsing for Cell Model Passports gene-by-sample matrices.

Cell Model Passports ships its omics matrices with genes as rows and samples as
columns, preceded by a block of header rows. The first header row carries the
``SIDM`` sample identifiers, the last one names the leading gene-annotation
columns, and the rows in between hold further per-sample metadata that the
pipeline does not consume. Layouts differ per file - CNV GISTIC has 3 header
rows and 2 annotation columns (``gene_id``, ``symbol``), RNA-seq has 4 and 3
(``gene_symbol``, ``ensembl_gene_id``, ``gene_id``) - so the annotation column
count is discovered from the header block instead of being passed in.
"""

import re
from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
from upath import UPath

from data_curation.cell_lines.sidm_mapping import build_sidm_to_cellosaurus_map
from data_curation.core import layout
from data_curation.core.errors import CurationError

SAMPLE_ID_PATTERN = re.compile(r"^SIDM\d+$")


@dataclass(frozen=True)
class CmpMatrix:
    """A Cell Model Passports matrix split into gene annotations and values.

    Attributes:
        annotations: Per-gene annotation columns, named by the last header row,
            one row per gene in file order.
        values: Gene x sample values as float32, indexed positionally with SIDM
            sample IDs as columns.
    """

    annotations: pd.DataFrame
    values: pd.DataFrame

    @property
    def sample_ids(self) -> list[str]:
        """SIDM sample identifiers in file order."""
        return [str(column) for column in self.values.columns]


def _count_annotation_columns(sample_id_row: pd.Series, path: UPath) -> int:
    """Return the number of leading annotation columns in the header row.

    Raises:
        CurationError: If the row holds no SIDM identifier, or if annotation and
            sample columns interleave rather than forming two blocks.
    """
    is_sample = [SAMPLE_ID_PATTERN.match(str(value)) is not None for value in sample_id_row]
    if not any(is_sample):
        msg = f"No SIDM sample columns found in the first header row of {path}"
        raise CurationError(msg)
    first_sample = is_sample.index(True)
    if not all(is_sample[first_sample:]):
        msg = f"Annotation and SIDM sample columns are interleaved in {path}"
        raise CurationError(msg)
    return first_sample


def read_cmp_matrix(path: UPath, n_header_rows: int) -> CmpMatrix:
    """Read a Cell Model Passports gene-by-sample matrix.

    Args:
        path: CSV file with genes as rows and samples as columns.
        n_header_rows: Number of leading header rows, including the SIDM row.

    Returns:
        The parsed matrix, annotations and values sharing a positional index.

    Raises:
        CurationError: If sample IDs repeat, or if the data rows do not match the
            header's column layout.
    """
    header = pd.read_csv(str(path), header=None, nrows=n_header_rows, dtype=str, low_memory=False)
    n_annotation_cols = _count_annotation_columns(header.iloc[0], path)
    annotation_names = [str(value) for value in header.iloc[n_header_rows - 1, :n_annotation_cols]]
    sample_ids = [str(value) for value in header.iloc[0, n_annotation_cols:]]
    if len(set(sample_ids)) != len(sample_ids):
        # Values are selected by column label downstream, which silently
        # duplicates data when labels repeat.
        msg = f"{path} repeats SIDM sample identifiers across columns"
        raise CurationError(msg)

    data = pd.read_csv(str(path), header=None, skiprows=n_header_rows, low_memory=False)
    if data.shape[1] != n_annotation_cols + len(sample_ids):
        msg = f"{path} has {data.shape[1]} data columns but {n_annotation_cols + len(sample_ids)} header columns"
        raise CurationError(msg)

    annotations = data.iloc[:, :n_annotation_cols].copy()
    annotations.columns = pd.Index(annotation_names)
    values = data.iloc[:, n_annotation_cols:].astype(np.float32)
    values.columns = pd.Index(sample_ids)
    return CmpMatrix(annotations=annotations, values=values)


def to_cell_line_anndata(
    values: pd.DataFrame,
    gene_names: pd.Series,
    model_list_path: UPath,
    layers: dict[str, pd.DataFrame] | None = None,
) -> ad.AnnData:
    """Transpose a gene-by-sample matrix into a cell line x gene AnnData.

    SIDM sample columns are mapped to Cellosaurus accessions via the model list;
    columns without a mapping are dropped, as are cell lines that map to an
    accession already seen.

    Args:
        values: Gene x sample values with SIDM sample IDs as columns.
        gene_names: Gene name per row of ``values``, used as ``var_names``.
        model_list_path: Cell Model Passports ``model_list.csv``.
        layers: Additional gene x sample matrices, aligned to ``values``, to be
            stored as AnnData layers under their dict keys.

    Returns:
        AnnData with cell lines as obs and genes as var.
    """
    sidm_to_cvcl = build_sidm_to_cellosaurus_map(model_list_path)
    valid_cols = [column for column in values.columns if str(column) in sidm_to_cvcl]

    adata = ad.AnnData(
        X=values[valid_cols].to_numpy(dtype=np.float32).T,
        obs=pd.DataFrame(
            {"cell_line_name": [sidm_to_cvcl[str(column)][1] for column in valid_cols]},
            index=pd.Index([sidm_to_cvcl[str(column)][0] for column in valid_cols]),
        ),
        var=pd.DataFrame(index=pd.Index(gene_names.to_numpy(), name="gene_symbol")),
        layers={name: layer[valid_cols].to_numpy(dtype=np.float32).T for name, layer in (layers or {}).items()},
    )
    adata.obs_names.name = layout.SAMPLE_INDEX_NAME
    adata.var_names_make_unique()
    return adata[~adata.obs_names.duplicated(keep="first")].copy()
