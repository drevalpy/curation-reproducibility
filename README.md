# data_curation

DVC pipeline that curates drug response datasets (CCLE, GDSC1, GDSC2, CTRPv1,
CTRPv2, BeatAML2, PDX_Bruna, PRISM_19Q4, PRISM_20Q2) into MuData (`.h5mu`) files for
[drevalpy](https://github.com/daisybio/drevalpy).

Run the whole pipeline with:

```bash
dvc repro
```

The repository configures no DVC remote: `dvc repro` rebuilds everything from the
public sources. To share the outputs, add your own remote locally, e.g.
`dvc remote add --local -d storage s3://<bucket>/<prefix>`. The published datasets
themselves are listed in drevalpy's `available_datasets.json`.

## Layout

The package is organised by domain. Each domain owns the modules DVC runs, named
after the stage that runs them, with helper clusters as subpackages. The whole
command surface lives in `cli` instead — one module per command group beside the
shared plumbing — so the domain packages carry no CLI code. `core` is the bottom
of the graph and knows nothing about the biology.

| Package | Owns |
| --- | --- |
| `core` | `config/` access, the data directory layout, raw-file downloads, glob-based path lookup, `CurationError` |
| `cli` | The `data-curation` command tree (one module per group), logging, progress |
| `cell_lines` | Cellosaurus and Cell Model Passports identity, and the sample metadata tables (`sample_id`, `tissue`, and for cell lines `cellosaurus_id`, `cell_line_name`) |
| `omics` | Transcriptomics, mutations, CNV, proteomics |
| `response` | Raw plates to well tables (`prepare/`), fitted curves through the drevalpy/curate Nextflow pipeline (`pipeline.py`), canonical identifiers (`harmonize/`) |
| `drugs` | PubChem CID resolution (`mapping/`), public drug classifications (`ontology/`) |
| `assembly` | MuData assembly and the toy subsample |

Dependencies run one way: `core` -> `cell_lines` -> `omics`/`response` ->
`drugs` -> `assembly`, with `cli` on top. Two edges read backwards, both
deliberate: `drugs/map_drugs.py` imports the CCLE response preparer, because
CCLE's compound list is a sheet of its response supplement (deferred inside the
function that needs it), and three `drugs` modules import `cli._progress` for the
progress helper their rate-limited PubChem loops report through.

`core` is a layer rather than a junk drawer, and the difference is checkable: a
module belongs there if several domains need it and it knows about none of them.
`tests/core/test_layering.py` asserts that nothing in `core` imports a domain
package, so the rule cannot erode one convenient import at a time.

`tests/` mirrors this tree.

## Data layout

Every stage reads and writes through `src/data_curation/core/layout.py`. The data
directory has one numbered directory per pipeline step, so a directory only ever
reads from lower-numbered ones:

```text
data/
  01_raw/          downloads: mapping/ (Cellosaurus), transcriptomics/ (non-CMP
                   datasets), response/, drug_ontology/, and cmp/ - the Cell Model
                   Passports files (model_list/, rnaseq/, mutations/, cnv/,
                   proteomics/) shared by every CMP dataset
  02_omics/        processed omics: transcriptomics/, mutations/, cnv/, proteomics/
  03_drugs/        mapping/ (per dataset, with <ds>_pubchem_failures.tsv),
                   structures.parquet and pubchem_structures_cache.json (one canonical
                   ID and SMILES per drug), ontology.parquet and pubchem_failures.tsv
                   (global)
  04_response/     wells/ -> curves/ (with uns["replicate_fits"]) -> harmonized/
  06_samples/      metadata/: cmp.parquet (shared by the CMP datasets) and
                   <dataset>.parquet for the others
  07_assembled/    final {dataset}.h5mu files
  08_reports/      data_summary.html, drug_structures.json (structure check)
  cache/           untracked scratch: nextflow/<dataset>/, the Nextflow launch
                   directory of each fit (history, work/, results/)
```

`layout.py` is not a DVC dependency of any stage - every path is also spelled out
in `dvc.yaml`, and `tests/cli/test_dvc_contract.py` checks that the two agree.
## Command line

Everything runs through one entry point, including every `dvc.yaml` stage, so
there is exactly one way to run a stage:

```
data-curation [--log-level LVL] [-v] [-q] [--traceback] [--version]
  cell-lines  download-cellosaurus
              process-cmp-meta
              process-meta               DATASET
  omics       download-cmp
              download                   MODALITY DATASET
              process                    MODALITY DATASET
  drugs       download-ontology
              map                        DATASET
              resolve-structures
              annotate
  response    download                   DATASET
              prepare-viability          DATASET
              fit-curves                 DATASET
              harmonize                  DATASET
  assembly    mudata                     DATASET
              check-drug-structures
              subsample-toy
              record-toy-gene-lists
  report
```

Dataset names are validated against `config/sources.yaml` before a stage starts,
scoped to the modality where one applies, so `omics process cnv PDX_Bruna` is
refused immediately rather than several minutes in. Expected failures (a bad
dataset, missing input, a tripped sanity threshold) are `CurationError`s and print
as one line; pass `--traceback` for the full trace. Anything else is a bug and
keeps its traceback. `params.yaml` holds only the
dataset lists DVC expands `foreach` over; the Python package never reads it.

Progress and logging adapt to the output stream: rate-limited PubChem loops draw a
progress bar in a terminal and emit periodic summary lines when piped, which is
what `dvc repro` sees.

**No `cli` module is listed in any stage's `deps`.** That layer parses arguments
and configures logging; it cannot change what a stage computes, so declaring it
would make an unrelated logging tweak invalidate every stage - including the ~9 h
`fit_curves`. A test asserts this stays true, alongside one that resolves every
`dvc.yaml` command against the real command tree so a rename cannot silently
break the pipeline.

## Cell Model Passports

CCLE, CTRPv1, CTRPv2, GDSC1 and GDSC2 take their omics from the same Cell Model
Passports (CMP) files (`omics_source: cell_model_passports` in
`config/sources.yaml`). A single `download_cmp` stage (`omics download-cmp`) fetches
the model list, RNA-seq, mutations, CNV and proteomics once into
`data/01_raw/cmp/<kind>/`, with the pinned hash of each file checked by `pooch`;
the per-modality process stages read from there. Likewise one stage
(`process_sample_meta`, `cell-lines process-cmp-meta`) builds the sample (cell-line)
metadata shared by those datasets, `data/06_samples/metadata/cmp.parquet`;
datasets outside CMP get their own `<dataset>.parquet`. Asking `omics download`
for a CMP dataset's omics is refused and points at `download-cmp`.

## Samples

drevalpy calls the biological source of a response a *sample* (a cell line, a PDX
model, primary patient cells). Every assembled `.h5mu` indexes `mdata.obs` and every
modality by a unique sample ID under the index name `sample_id`:

| Datasets | Sample ID | Extra `obs` columns |
| --- | --- | --- |
| CCLE, CTRPv1/v2, GDSC1/2, PRISM_19Q4/20Q2 (and the TOY subsample) | Cellosaurus accession | `tissue`, plus the optional `cellosaurus_id` and `cell_line_name` (and CMP annotations) |
| PDX_Bruna | PDX model ID | `tissue` only |
| BeatAML2 | patient sample ID (dbGaP subject) | `tissue` only |

`tissue` is the only column drevalpy requires; the two cell-line columns are written
only where the samples really are cell lines, never filled in for PDX or patient
samples. The metadata tables behind this (`process_sample_meta`,
`process_sample_meta_rna_only`) are keyed by a `sample_id` column.

## Response curation

Response is curated in three stages rather than one, because curve fitting is by
far the most expensive thing the pipeline does (~9 h for all datasets locally at
`fit_speed: exhaustive` in `config/processing.yaml`) and identity resolution is
among the most frequently changed.

| Stage | Command | Output | Cost |
| --- | --- | --- | --- |
| `prepare_viability@{ds}` | `response prepare-viability {ds}` | `data/04_response/wells/{ds}.parquet` | minutes |
| `fit_curves@{ds}` | `response fit-curves {ds}` | `data/04_response/curves/{ds}.h5ad` | hours |
| `harmonize_response@{ds}` | `response harmonize {ds}` | `data/04_response/harmonized/{ds}.h5ad` | minutes |

The split works because **CurveCurator is identifier-agnostic**: it groups by
dose range and treats `sample`/`drug` as opaque labels. So `prepare_viability`
emits each dataset's *own* identifiers (`SANGER_MODEL_ID`/`DRUG_ID` for GDSC,
`ccl_name`/`master_cpd_id` for CTRPv2, and so on), `fit_curves` fits those, and
`harmonize_response` renames both axes to the sample IDs and PubChem CIDs
drevalpy expects (Cellosaurus accessions for the cell-line datasets). `fit_curves` therefore depends on the well table,
the `curvecurator` and `replicates` blocks and the pipeline revision
(`execution.pipeline`, `execution.revision`) in `config/processing.yaml`, the stage
code and **nothing else** - a change to Cellosaurus name resolution or the PubChem
drug mapping can no longer invalidate a fit it cannot affect. The same run fits
every replicate series on its own into the curve file's `uns["replicate_fits"]`,
which `harmonize_response` renames along with the curves.

### Replicates

A replicate is one dose series of a (sample, drug) on one plate. Each preparer
names the columns that make up the plate key, the experiment and the metadata to
keep (`response/prepare/wells.py`); drevalpy (`drevalpy.curation._series`) numbers
the series. A series whose every dose
is repeated `k >= 2` times (GDSC2's repeated tags, a few CTRPv2 series) is split
into `k` series by well position; any other repetition is averaged per dose.
Series with fewer than `replicates.min_doses` (4) distinct doses are fitted
neither pooled nor on their own.

Replicates are nested. An **experiment** is one independently seeded culture of
a sample, and its series (parallel plates, library slots or repeated blocks) are
**technical** replicates of each other; different experiments of a pair are
**biological** replicates. Each preparer names the experiment columns:

| Dataset | One experiment (biological replicate) | Technical replicates within it |
|---|---|---|
| GDSC1, GDSC2 | `CELL_ID` (cell line stock) seeded on one day (`DATE_CREATED`, UK date) | plates, libraries and slots of that seeding |
| CTRPv2 | `experiment_id` | assay plates |
| PRISM_19Q4, PRISM_20Q2 | `compound_plate` and cell-line pool | the replicate detection plates |
| BeatAML2 | run and specimen (`run_index`, DNA/RNA-seq sample) | panels, plates and native replicates |
| PDX_Bruna | the PDTC | plates |
| CCLE, CTRPv1 | the published (aggregated) series | none |

The assembled response modality carries every
replicate's fit as `response.uns["replicate_fits"]`, a long table with one row
per series:

| column | meaning |
|---|---|
| `sample_id`, `pubchem_id` | canonical identifiers (the response `obs`/`var` names) |
| `native_sample`, `native_drug` | the dataset's own identifiers |
| `replicate` | `0..n-1` per native pair in (`biological_replicate`, `technical_replicate`) order, fittable series first |
| `series_key`, `experiment` | the plate key and the experiment it belongs to |
| `biological_replicate` | `0..m-1` per native pair, one per `experiment` (sorted) |
| `technical_replicate` | `0..k-1` within the biological replicate, ordered by `series_key` |
| `n_doses` | distinct doses in the series |
| `source` | `replicate` (own fit) or `pooled` (the pair has one fittable series; copied from `fit_curves`) |
| `status` | `fitted`, `fit_failed` (CurveCurator dropped it) or `too_few_doses` |
| `pEC50` ... `back_error` | the 22 metrics of the response layers, float32, NaN unless fitted |
| further columns | experiment-level metadata, per dataset (e.g. `barcode`, `date_created` for GDSC) |

### Fitting with Nextflow

`fit_curves` runs the [drevalpy/curate](https://github.com/drevalpy/curate) Nextflow
pipeline (`response/pipeline.py`) on one dataset: it passes the absolute path of the
dataset's well table as `--input`, its `normalize` flag as `--normalize` and its
metadata columns as `--metadata`, runs `nextflow run` from
`data/cache/nextflow/<dataset>/` and copies `results/curves.h5ad` to
`data/04_response/curves/<dataset>.h5ad`. The pipeline cuts the
curves into chunks of at most `execution.max_curves_per_job`, fits each chunk as a
single-core task and collects them into the same AnnData `drevalpy.curation.curate()`
writes. `nextflow` must be on the `PATH`; its output goes to the stage log.

The pipeline is `execution.pipeline` at `execution.revision`, both DVC params of
`fit_curves`: the revision pins the pipeline's container, and with it the drevalpy
and CurveCurator versions, so bumping it marks every fit stale. To run a local
checkout instead (passed without `-r`), set `DATA_CURATION_PIPELINE`:

```bash
DATA_CURATION_PIPELINE=/path/to/curate uv run data-curation response fit-curves GDSC2
```

Where the tasks run is a deployment choice, not a parameter: `execution.backend`
(or the environment variable `DATA_CURATION_BACKEND`) selects the executor config
`config/nextflow/<backend>.config`, passed with `-c`, and neither is a DVC dep.

| Backend | Runs tasks | Container |
| --- | --- | --- |
| `local` (default) | on this machine, 4 at once, at most 12 GB | Docker |
| `slurm` | as SLURM jobs on `shared-cpu`, at most 64 queued | Apptainer/Singularity |
| `aws_batch` | as AWS Batch jobs on the queues and S3 work directory set in the environment (see below) | Docker with Fusion, via Wave |

```bash
export DATA_CURATION_AWS_QUEUE=<queue for chunk fits>          # 2 vCPU / 8 GB instances suffice
export DATA_CURATION_AWS_LARGE_QUEUE=<queue with >= 32 GB>     # planning and collecting; optional
export DATA_CURATION_AWS_WORKDIR=s3://<bucket>/nextflow/work
DATA_CURATION_BACKEND=aws_batch uv run data-curation response fit-curves GDSC2
```

Every run passes `-resume`. A fitting task is cached by the content of its chunk
(its curves and fit options), so a run after an interruption, or after the wells
of a few curves changed, refits only the chunks that changed; the history lives in
the dataset's launch directory, and AWS Batch keeps its own work directory on S3.
Changing `max_curves_per_job` cuts the curves into different chunks and refits all
of them, though the results stay the same. The pipeline pins OpenBLAS to its
Haswell kernels and runs every task single-threaded, so the same chunk fits the
same on any x86-64 node with AVX2, whether Intel or AMD.

On AWS Batch the tasks run the public pipeline image `ghcr.io/drevalpy/curate`,
which holds drevalpy only. Fusion moves their files between S3 and the task, and
Wave adds Fusion to the image at launch, so neither the image nor the compute
environment's AMI needs the AWS CLI. Nextflow creates the job definitions itself.
The Batch instance role needs read and write access to the work directory.

### Duplicate collapse

Renaming to canonical identifiers can map several native identifiers onto one.
Samples are nearly injective (CTRPv2 has one collision, `RH30`/`SJRH30` ->
`CVCL_0041`), but drugs are not: GDSC1 collapses 402 native `DRUG_ID`s onto 378
CIDs and GDSC2 294 onto 285, all repeat screens of the same compound.

Those repeat screens use different dose ranges, so they were already separate
fitted rows even before the split - drevalpy's `_pivot_metric` just picked one
arbitrarily with `aggfunc="first"`. `harmonize_response` instead applies a
documented rule per cell: prefer a fitted value over a missing one, then the
higher `signal_quality`, then the lower `RMSE`, then the lexicographically
smaller native identifier. The last criterion only removes the row-order
dependence. Every collapse is logged, and the surviving entry records its sources
in `native_ids`.

### CCLE

CCLE response comes from supplementary table S11 of
[Barretina et al. 2012](https://doi.org/10.1038/nature11003) - 11,670 curves over
504 cell lines and 24 compounds, with each row's dose series and median activity
packed into two comma-separated strings.

The published `Activity Data (median)` is **not** viability: it is a percent
change in growth relative to the untreated control, where `0` means the treated
well matched control and `-100` means growth was abolished. It is converted with
`activity / 100 + 1` to the fractional viability every other preparer emits, then
clipped at zero. A third of the resulting values exceed 1 (treated wells that grew
faster than control, which is kept rather than truncated) and 0.16% of the raw
values fall below `-100` (clipped).

The file is pinned to Springer's copy rather than PMC's
`NIHMS361223-supplement-4.xlsx`: every PMC route now returns a proof-of-work
interstitial that `pooch` cannot solve, and the PMC Open Access package for
`PMC3320027` is a 404 because the article carries no OA licence. The two
workbooks are the same content; Springer's is the legacy `.xls`, which is why
`xlrd` is a dependency.

Omics ride the shared Cell Model Passports artifacts, mapped by cell-line name
exactly as CTRPv1/v2 already are. The old pipeline's SRA + nf-core/rnaseq +
GenePattern reprocessing route is deliberately not reproduced - it is not
expressible as a DVC stage.

## Drug ontology

Screened compounds are annotated with public, externally maintained drug
classifications so drugs can be grouped at several levels of detail. No
vocabulary is invented here; every term comes from one of the sources below.

| Source | Contributes | Version | License |
| --- | --- | --- | --- |
| [Probes & Drugs](https://www.probes-drugs.org/) | ChEMBL protein family classification (5 levels), target gene, mechanism of action | 01.2026 | CC BY-SA 4.0 |
| [ChEBI](https://www.ebi.ac.uk/chebi/) | Therapeutic roles (`antineoplastic agent`, ...) | rel252 | CC BY 4.0 |
| [NCI Thesaurus](https://evs.nci.nih.gov/) | Oncology drug classes and branches | 26.07d | CC BY 4.0 |
| [PubChem](https://pubchem.ncbi.nlm.nih.gov/) | CID cross references, CID to NCIt links | retrieved at build time | Public domain |

Two stages produce the annotations:

- `download_drug_ontology` fetches the pinned source releases into
  `data/01_raw/drug_ontology/`.
- `annotate_drugs` reads every `data/03_drugs/mapping/*.parquet`, takes the union
  of the drugs they screen, and writes one central dictionary to
  `data/03_drugs/ontology.parquet`. `assemble_mudata` then merges it onto
  the `var` of the `response` modality.

### PubChem failure budget

PubChem lookups retry transient errors, but a lookup that still fails (network,
5xx, throttling) is not the same as a compound PubChem does not know, and must not
silently become a pseudo-ID. `map_drugs` and `annotate_drugs` collect the failed
queries and fail the stage when more than `pubchem.max_failure_fraction` (default
`0.05`, in `config/processing.yaml`) of the lookups failed. The failures are
written either way, with their reasons, to
`data/03_drugs/mapping/<dataset>_pubchem_failures.tsv` and
`data/03_drugs/pubchem_failures.tsv` (declared `cache: false`, so they are plain
files rather than DVC-cache links). Re-run the stage once PubChem is reachable
again.

Grouping columns, from coarse to fine: `drug_role`, `ncit_branch`, `ncit_class`,
`target_class_l1` .. `target_class_l5`, `target_gene`. Unresolved values are the
literal string `unknown` rather than missing, so `groupby` works directly.
`drug_role_all` and `target_gene_all` keep the full pipe-separated sets for
drugs with several roles or targets.

### Drug identifiers and names

Drugs are keyed by their PubChem CID, which merges the same compound across
datasets and across repeat screens within one dataset. A drug PubChem cannot
resolve keeps a dataset-prefixed pseudo-ID (`GDSC_1195`, `CTRP_660221`) so its
measurements are never dropped. `drug_name` prefers PubChem's canonical name and
falls back to the name used by the source dataset, so pseudo-ID drugs are still
named.

### One SMILES and one ID per drug

`map_drugs` picks `canonical_smiles` per dataset (PubChem's stereo-free connectivity
SMILES, the study's own, a PRISM multi-structure string), so a CID used to carry a
different SMILES in every dataset, and one molecule could sit under two CIDs. The
`resolve_drug_structures` stage (`drugs/resolve_structures.py`) reads every
`mapping/*.parquet` and writes `data/03_drugs/structures.parquet`
(`pubchem_id, canonical_id, canonical_smiles, inchikey`):

- **SMILES:** the CID's isomeric PubChem SMILES, rewritten with RDKit's
  `Chem.MolToSmiles`, so a drug has the same string in every dataset.
- **ID:** CIDs with the same InChIKey are one molecule and are all remapped onto the
  lowest numeric CID (sildenafil 135398744 -> 5212, XAV-939 135418940 -> 2726824,
  pemetrexed 135410875 -> 446556). The group shares that CID's SMILES.
- **No CID:** pseudo-IDs, and CIDs PubChem returns nothing for (logged), keep the
  study's SMILES, RDKit-canonicalized if it parses and unchanged otherwise.

PubChem's raw answers are cached in `data/03_drugs/pubchem_structures_cache.json`, a
`persist: true` output that DVC leaves in place, so a re-run only requests new CIDs
(delete an entry to re-request it). A failing request aborts the stage instead of
falling back to the study's SMILES.

`harmonize_response` and `annotate_drugs` read the table
(`response/harmonize/structures.py::apply_structures`), so `var` and the ontology
carry canonical IDs. Two native drugs of one dataset that land on the same canonical
ID collapse by the rule above, with their native `DRUG_ID`s in `native_ids`; the
original CID stays in `mapping/<ds>.parquet` and in `structures.parquet`.
`map_drugs` itself is unchanged and not re-run. The `check_drug_structures` stage
reads `mod/response/var` of every `data/07_assembled/*.h5mu` and fails if a drug ID has
more than one SMILES or an InChIKey more than one ID (report:
`data/08_reports/drug_structures.json`).

### PRISM drug identity

PRISM's `smiles` column lists every structure the Broad registered for a compound
(`a, a, b`): mostly identical copies, sometimes stereo, E/Z or tautomer variants,
sometimes empty entries, which PubChem rejects as a query. So `map_prism` (the
`map_drugs_PRISM_*` stages, `data-curation drugs map-prism`) resolves PRISM like
the name-mapped datasets and uses the structures only to check and, where needed,
override the name (`drugs/prism_structures.py`):

1. Parse the field, drop empty and invalid entries, and reduce each structure to
   its neutral parent (largest organic fragment, charges neutralized; metal
   complexes keep their metal). Its identity is the first block of the InChIKey.
2. Look the name up. If the CID's record has the same parent connectivity as any
   registered structure (or the same canonical tautomer), keep it, so PRISM agrees
   with GDSC and CTRP wherever the name does.
3. Otherwise look up the full InChIKey of the most stereo-specific structure, then
   the others. The reason is logged at info level.
4. A name CID that no structure confirms or replaces is kept; metal complexes never
   get a structure CID, because PubChem draws them differently from the registry.
   Combinations keep their pseudo-ID.

PRISM_20Q2 leaves 12 compounds without name or structure (imatinib, bortezomib,
...). `datasets.PRISM_20Q2.drug_mapping.name_fallback_from` in `config/sources.yaml`
names the release whose treatment info supplies them (matching `broad_id`, then the
`BRD-X12345678` compound stem when its names agree). Only `map_drugs_PRISM_20Q2`
therefore depends on the `PRISM_19Q4` download.

### Combination therapies

CTRPv2 (49 drugs) and PDX_Bruna (1) screen fixed-ratio combinations, encoded in
the drug name as `JQ-1:UNC0638 (2:1 mol/mol)` and
`Olaparib(1495) + Temozolomide(1375)` respectively. A mixture has no single CID,
so combinations always keep a pseudo-ID; resolving them to a CID would silently
merge them with an unrelated single compound. Three extra columns describe them:

| Column | Meaning |
| --- | --- |
| `is_combination` | `True` for a fixed-ratio combination |
| `n_components` | Number of component compounds, `0` for a single agent |
| `component_pubchem_ids` | Pipe-separated component IDs, resolved from the same dataset's single-agent screen where possible |

Combinations carry `drug_role = "combination"` so they group together instead of
falling into the `unknown` bucket.

### Coverage

Roughly 42% of the 1143 distinct drugs receive a target classification and 32% a
therapeutic role. Coverage is highest for the clinically oriented libraries
(BeatAML2 81%, PDX_Bruna 72%, GDSC ~62%) and lowest for CTRP (~37%), which
deliberately screens natural products and tool compounds that no public drug
ontology describes.

Compounds are left as `unknown` rather than being given a guessed annotation.
In particular, Probes & Drugs lists every measured target including weak
screening hits, so the primary target is restricted to rows with a curated
mechanism of action or a designated chemical probe.

### Attribution

Probes & Drugs is licensed CC BY-SA 4.0, so `ontology.parquet` and the
`var` columns derived from it are share-alike. Retain this attribution section
when redistributing those files.

## Publishing

`scripts/publish_datasets_wizard.sh` (run from the repository root) publishes every
assembled MuData file and TOYv1 to a new generation folder
`s3://orakl-open-source-data/drevalpy_h5mu/genN/`: it brings the assembly stages up
to date, checks each file (response modality, `replicate_fits` table inside the
response axes, TOYv1's shape and gene lists), uploads, re-downloads to compare
checksums and prints the `available_datasets.json` entries with their sha256 pins.
