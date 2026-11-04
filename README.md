# data_curation

A DVC pipeline that turns nine public drug response datasets into the MuData
(`.h5mu`) files [drevalpy](https://github.com/daisybio/drevalpy) loads: CCLE,
CTRPv1, CTRPv2, GDSC1, GDSC2, BeatAML2, PDX_Bruna, PRISM_19Q4 and PRISM_20Q2, plus
the TOYv1 subsample. Each file holds the omics of every sample (transcriptomics,
mutations, CNV, proteomics, where the dataset has them) and a `response` modality
of fitted dose-response curves.

## Reproduce the datasets

You need [uv](https://docs.astral.sh/uv/), [Nextflow](https://www.nextflow.io)
25.10.4 or later, and Docker.

```bash
uv sync
uv run dvc repro
```

`dvc repro` downloads every source, fits the curves and writes
`data/07_assembled/<dataset>.h5mu`. The curve fits take about 9 hours on one
machine. To spread them over a cluster, see
[Fit the curves elsewhere](#fit-the-curves-elsewhere).

The repository configures no DVC remote, so a fresh clone rebuilds everything from
the public sources. To keep the outputs, add your own remote with
`dvc remote add --local -d storage s3://<bucket>/<prefix>`.

## Where the data goes

Each step writes to its own numbered directory and reads only from lower-numbered
ones. `src/data_curation/core/layout.py` defines every path.

```text
data/
  01_raw/          downloads
  02_omics/        processed transcriptomics, mutations, CNV and proteomics
  03_drugs/        drug mapping, canonical structures and the drug ontology
  04_response/     wells/ -> curves/ -> harmonized/
  06_samples/      sample metadata
  07_assembled/    the final <dataset>.h5mu files
  08_reports/      data_summary.html and the drug structure check
  cache/           untracked: one Nextflow launch directory per dataset
```

Every DVC stage runs one `data-curation` command. Run `uv run data-curation --help`
to list them.

## Fit the curves

Response curation has three stages per dataset:

| Stage | Output | Runtime |
| --- | --- | --- |
| `prepare_viability@<dataset>` | `data/04_response/wells/<dataset>.parquet` | minutes |
| `fit_curves@<dataset>` | `data/04_response/curves/<dataset>.h5ad` | hours |
| `harmonize_response@<dataset>` | `data/04_response/harmonized/<dataset>.h5ad` | minutes |

`prepare_viability` writes one row per well and keeps the dataset's own sample and
drug identifiers. `fit_curves` fits those wells with CurveCurator, which ignores
what the labels mean. `harmonize_response` then renames both axes to the
identifiers drevalpy expects. Because of this split, a change to identifier mapping
never invalidates a fit.

`fit_curves` runs the [drevalpy/curate](https://github.com/drevalpy/curate)
Nextflow pipeline:

```bash
nextflow run drevalpy/curate -r <execution.revision> -latest \
	--input data/04_response/wells/<dataset>.parquet --replicate_fits ... -resume
```

`execution.revision` in `config/processing.yaml` pins a commit of drevalpy/curate,
and that commit pins the container with drevalpy and CurveCurator. The revision is
a DVC param, so changing it marks every fit stale. To try a pipeline change, push
it to drevalpy/curate and pin its commit.

The pipeline cuts each dataset into chunks of at most
`execution.max_curves_per_job` curves and fits each chunk as a single-core task.
Every run passes `-resume`, so a rerun refits only the chunks whose curves changed.
OpenBLAS is pinned to its Haswell kernels and runs single-threaded. The same chunk
therefore fits the same on any x86-64 CPU with AVX2.

### Fit the curves elsewhere

`execution.backend`, or the environment variable `DATA_CURATION_BACKEND`, picks
the Nextflow config in `config/nextflow/`. The backend decides where tasks run,
never what they compute, so it is not a DVC param.

| Backend | Runs tasks | Container |
| --- | --- | --- |
| `local` (default) | on this machine, 4 at once, in at most 12 GB | Docker |
| `slurm` | as SLURM jobs in the `shared-cpu` partition | Apptainer |
| `aws_batch` | as AWS Batch jobs | Docker, with Fusion added by Wave |

To fit on AWS Batch, set the queues and the S3 work directory:

```bash
export DATA_CURATION_AWS_QUEUE=<queue>               # chunk fits; 2 vCPU and 8 GB suffice
export DATA_CURATION_AWS_LARGE_QUEUE=<queue>         # planning and collecting, 16 GB or more
export DATA_CURATION_AWS_WORKDIR=s3://<bucket>/nextflow/work
DATA_CURATION_BACKEND=aws_batch uv run dvc repro
```

`DATA_CURATION_AWS_LARGE_QUEUE` defaults to `DATA_CURATION_AWS_QUEUE`. The region
comes from your AWS configuration. The Batch instance role needs read and write
access to the work directory.

## Samples

Every `.h5mu` indexes `obs` and every modality by `sample_id`:

| Datasets | `sample_id` | Extra `obs` columns |
| --- | --- | --- |
| CCLE, CTRPv1, CTRPv2, GDSC1, GDSC2, PRISM_19Q4, PRISM_20Q2, TOYv1 | Cellosaurus accession | `tissue`, `cellosaurus_id`, `cell_line_name` |
| PDX_Bruna | PDX model ID | `tissue` |
| BeatAML2 | patient sample ID (dbGaP subject) | `tissue` |

CCLE, CTRPv1, CTRPv2, GDSC1 and GDSC2 all take their omics from the same Cell Model
Passports release.

## Replicates

A replicate is one dose series of a sample and a drug on one plate. The `response`
modality holds one pooled curve per sample and drug, fitted on all of its wells.
It also holds every series fitted on its own, in `response.uns["replicate_fits"]`,
one row per series. Series with fewer than 4 distinct doses are not fitted.

Replicates are nested. An experiment is one independently seeded culture of a
sample. Series within one experiment are technical replicates. Different
experiments of the same pair are biological replicates.

| Dataset | One experiment | Technical replicates in it |
| --- | --- | --- |
| GDSC1, GDSC2 | one cell line stock (`CELL_ID`) seeded on one day | plates, libraries and slots |
| CTRPv2 | `experiment_id` | assay plates |
| PRISM_19Q4, PRISM_20Q2 | `compound_plate` and cell-line pool | replicate detection plates |
| BeatAML2 | run and specimen | panels, plates and native replicates |
| PDX_Bruna | the PDTC | plates |
| CCLE, CTRPv1 | the published, aggregated series | none |

The main columns of `replicate_fits`:

| Column | Meaning |
| --- | --- |
| `sample_id`, `pubchem_id` | the identifiers of the response `obs` and `var` |
| `native_sample`, `native_drug` | the dataset's own identifiers |
| `biological_replicate`, `technical_replicate` | the replicate's position in the hierarchy |
| `n_doses` | the number of distinct doses |
| `source` | `replicate` for its own fit. `pooled` if the pair has a single series, copied from the pooled curve. |
| `status` | `fitted`, `fit_failed` or `too_few_doses` |
| `pEC50` to `back_error` | the 22 curve metrics of the response layers, NaN unless fitted |

## Curation decisions

### Repeat screens of one drug collapse into one entry

GDSC1 maps 402 native drug IDs onto 378 PubChem CIDs, and GDSC2 maps 294 onto 285.
These are repeat screens of the same compound. When several native entries land on
one cell, `harmonize_response` keeps one value by these rules, in order:

1. Prefer a fitted value over a missing one.
2. Prefer the higher `signal_quality`.
3. Prefer the lower `RMSE`.
4. Prefer the lexicographically smaller native identifier.

The surviving entry lists its sources in `native_ids`.

### CCLE activity becomes viability

CCLE's response comes from table S11 of
[Barretina et al. 2012](https://doi.org/10.1038/nature11003). Its activity is the
percent change in growth relative to the untreated control, not viability.
`activity / 100 + 1` converts it to the fractional viability of the other datasets,
clipped at zero. Values above 1 stay as they are.

### Each drug has one ID and one SMILES

Drugs are keyed by PubChem CID. CIDs with the same InChIKey are one molecule and
map onto the lowest CID. Every drug carries its isomeric PubChem SMILES,
canonicalized with RDKit. A drug PubChem can't resolve keeps a dataset-prefixed
pseudo-ID, such as `GDSC_1195`, and the study's SMILES. The
`check_drug_structures` stage fails if a drug ID has two SMILES or an InChIKey has
two IDs.

PRISM lists several registered structures per compound. PRISM drugs are looked up
by name first. The structures confirm the name's CID or, if none matches, replace
it.

Fixed-ratio combinations in CTRPv2 (49) and PDX_Bruna (1) always keep a pseudo-ID.
The columns `is_combination`, `n_components` and `component_pubchem_ids` describe
them.

### Drugs are annotated from public ontologies

The `var` of the `response` modality carries drug classes from these sources:

| Source | Contributes | Version | License |
| --- | --- | --- | --- |
| [Probes & Drugs](https://www.probes-drugs.org/) | ChEMBL target classes, target gene, mechanism of action | 01.2026 | CC BY-SA 4.0 |
| [ChEBI](https://www.ebi.ac.uk/chebi/) | therapeutic roles | rel252 | CC BY 4.0 |
| [NCI Thesaurus](https://evs.nci.nih.gov/) | oncology drug classes | 26.07d | CC BY 4.0 |
| [PubChem](https://pubchem.ncbi.nlm.nih.gov/) | CID cross-references | retrieved at build time | public domain |

The grouping columns, from coarse to fine, are `drug_role`, `ncit_branch`,
`ncit_class`, `target_class_l1` to `target_class_l5` and `target_gene`. A value no
source provides is the string `unknown`. About 42% of the 1143 drugs get a target
class, and about 32% get a therapeutic role.

If more than 5% of the PubChem lookups in a stage fail, the stage fails. The
failures are written to `data/03_drugs/pubchem_failures.tsv` and to
`data/03_drugs/mapping/<dataset>_pubchem_failures.tsv`.

### Attribution

Probes & Drugs is licensed CC BY-SA 4.0, so `ontology.parquet` and the `var`
columns derived from it are share-alike. Keep this section when you redistribute
those files.

## Curate your own dataset

A dataset whose raw data cannot be public can still be curated with this
pipeline. Your own code writes five Parquet files, and a separate repository
runs the public stages on them through the `data-curation` commands. The
contract is defined and checked in `src/data_curation/core/standard.py`; call
`standard.validate(kind, frame)` before you write each file.

| File | Directory | One row per | Columns |
| --- | --- | --- | --- |
| `wells.parquet` | `data/01_raw/response/<dataset>/` | well | `sample`, `drug`, `concentration` (µM), `intensity` (viability relative to the untreated control), optional `experiment` (biological replicate), `replicate` (technical replicate), `position`, and any metadata |
| `published.parquet` (optional) | same | published fit | `sample`, `drug` and numeric metrics, each becoming the response layer `<metric>_published` |
| `drugs.parquet` | same | drug | `drug`, `drug_name` for the PubChem lookup, optional `pubchem_id` and `smiles` |
| `samples.parquet` | same | sample | `sample_id`, `tissue` and further metadata |
| `expression.parquet` | `data/01_raw/transcriptomics/<dataset>/` | sample and gene | `sample_id`, `gene_symbol`, `tpm`, optional `counts` |

The dataset's block in `config/sources.yaml` selects these routes:

```yaml
datasets:
  MyScreen:
    sample_metadata: standard
    response:
      format: standard
      normalize: false # true if intensity is not yet relative to the controls
    drug_mapping:
      cid_overrides: {} # name -> PubChem CID, used when the lookup finds nothing
      no_lookup: [] # names that are never sent to PubChem, alone or in a combination
    transcriptomics:
      format: standard
      counts_unit: counts
```

Samples keep their own identifiers as `sample_id`. Expression is processed like
the public datasets: `X` is log2(TPM+1), and genes missing in any sample are dropped.

## Publish a dataset generation

`scripts/publish_datasets_wizard.sh` walks a maintainer through publishing the
assembled files to `s3://orakl-open-source-data/drevalpy_h5mu/genN/`. It checks
and uploads each file, then prints the `available_datasets.json` entries for
drevalpy.
