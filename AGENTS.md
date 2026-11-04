# AGENTS.md

DVC pipeline that curates drug response datasets into MuData files for drevalpy. `README.md` explains the
pipeline and its data; this file is what you need to change the code safely. Run everything with
`uv run --frozen ...` from the repository root (stages resolve `config/` and `data/` relative to the cwd).

## Layout

```text
src/data_curation/
  core/          bottom layer: config.py, layout.py (all data paths), paths.py, download.py, errors.py,
                 standard.py (file contract of `format: standard` datasets curated from private repositories)
  cell_lines/ omics/ response/ drugs/ assembly/ report/   domain packages; one module per DVC stage;
                 response/pipeline.py runs the drevalpy/curate Nextflow pipeline for fit_curves
  cli/           the whole `data-curation` command tree, one module per group
tests/           mirrors src/; data-free (fixtures are built into tmp_path)
config/          sources.yaml (pinned URLs, hashes, per-dataset blocks), processing.yaml (fit, execution, pubchem),
                 nextflow/<backend>.config (executor settings for local, slurm, aws_batch)
params.yaml      only the dataset lists DVC expands `foreach` over; the Python code never reads it
dvc.yaml         one stage per command; dvc.lock is the recorded state
```

Layering, enforced by `tests/core/test_layering.py`: `core` imports nothing but `core`; domains import `core`
and domains lower in the order `cell_lines -> omics/response -> drugs -> assembly`; domains never import `cli`
except `cli._progress`. `cli` is on top.

## Commands

```bash
uv run --frozen pytest -q                    # whole suite, ~2.5 min; -m 'not slow' skips the real Nextflow run
uv run --frozen ruff check && uv run --frozen ruff format
uv run --frozen ty check src
uv run --frozen complexipy src
uv run --frozen prek run --all-files         # all of the above plus file hygiene, as in .pre-commit-config.yaml
```

## Conventions

- Raise `CurationError` (`core/errors.py`) for bad or missing user input, config and data; the CLI prints it as
  one line. Anything else (`ValueError`, `RuntimeError`, ...) is a bug and keeps its traceback. Do not wrap
  internal invariants in `CurationError`. It subclasses `Exception` only, so never rely on `except ValueError`
  to catch it.
- Keep `cli` imports lazy: every command imports its implementation inside the function body so
  `data-curation --help` does not load anndata/mudata/torch.
- `core/layout.py` is the only source of data paths; stage code never spells out `data/...`.
- Configuration is read through `core/config.py` (`source_config`, `processing_config`, `dataset_config`), not by
  opening YAML directly.
- Google-style docstrings; ruff line length 120. Tests mirror the source tree and use literals in `tmp_path`.
- The Nextflow executor is a deployment choice (`execution.backend`, env `DATA_CURATION_BACKEND`, selecting
  `config/nextflow/<backend>.config`) and must never become a DVC dep or param; neither may
  `execution.max_curves_per_job`. The pipeline source and revision (`execution.pipeline`, `execution.revision`)
  are params of `fit_curves`: the revision pins the container and so the fitting libraries. To test a
  pipeline change, push it to drevalpy/curate and pin its commit; there is no local-checkout override.

## DVC rules

- After any change run `uv run --frozen dvc repro --dry`. A stage listed there because code or deps changed
  whose outputs would be identical (refactor, formatting, docstring, added dep) is marked up to date with
  `uv run --frozen dvc commit -f <stage>` (`<stage>@<dataset>` for `foreach` stages) instead of being re-run.
  Only re-run a stage when its output should actually change.
- Never run the curve fitting stages (`fit_curves`) or `dvc push` without the user's approval. Fits take hours
  and can be run on SLURM or AWS Batch. Each dataset runs Nextflow with `-resume` from
  `data/cache/nextflow/<dataset>/`; keep that directory so an interrupted or re-run fit only refits changed chunks.
- Editing `response/fit_curves.py` or `response/pipeline.py`, or bumping `execution.revision`, invalidates every
  fit stage. Expect to `dvc commit -f` them after behaviour-preserving edits.
- `tests/cli/test_dvc_contract.py` requires each stage's `deps` to cover its full transitive `data_curation`
  import closure (`core/config.py`, `core/layout.py`, `core/errors.py` included; `cli` excluded). When you add an
  import, add the module to every stage that reaches it in `dvc.yaml`; the test names what is missing.
- PRISM drug mapping is its own stage group (`map_drugs_PRISM_*`, `drugs map-prism`, `drugs/map_prism.py`) so that
  the `map_drugs` stages of the other datasets do not list `prism_structures.py`. Keep `map_drugs.py` free of
  imports of `map_prism`/`prism_structures`; `PRISM_20Q2` alone also depends on the `PRISM_19Q4` download.
- Never edit or delete `data/`. Outputs there are DVC-managed.
- `earlyoom` on this host kills processes above 85% memory (exit code 143); a stage that dies with 143 ran out
  of memory, so lower parallelism (`executor.cpus`/`memory` in `config/nextflow/local.config`) or use another
  backend rather than retrying blindly.
