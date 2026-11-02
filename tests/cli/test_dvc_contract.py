"""Tests that dvc.yaml and the CLI cannot drift apart.

Since ``dvc.yaml`` invokes the CLI, a renamed command breaks the pipeline without
breaking any import. Nothing else in the suite would notice, so these tests walk
every command in ``dvc.yaml`` and resolve it against the real command tree.
"""

import re
import shlex
from pathlib import Path
from typing import Any

import pytest
import typer.main
import yaml
from import_graph import REPO_ROOT, command_modules, covered_by, import_closure
from upath import UPath

from data_curation import cli

DVC_PATH = UPath("dvc.yaml")
_INTERPOLATION = re.compile(r"\$\{([^}]+)\}")


def _load_pipeline() -> dict[str, Any]:
    """Parse ``dvc.yaml`` from the repository root."""
    with DVC_PATH.open() as handle:
        return yaml.safe_load(handle)


def _load_params() -> dict[str, Any]:
    """Parse ``params.yaml`` for foreach list expansion in these tests only."""
    with UPath("params.yaml").open() as handle:
        return yaml.safe_load(handle)


def _foreach_items(spec: object, params: dict[str, Any]) -> list[str]:
    """Resolve a ``foreach`` spec the way DVC does for simple list references."""
    if isinstance(spec, list):
        return [str(entry) for entry in spec]
    if isinstance(spec, str):
        match = _INTERPOLATION.fullmatch(spec.strip())
        if match:
            return [str(entry) for entry in params.get(match.group(1), [])]
    return []


def load_stages() -> list[dict[str, str]]:
    """Expand ``dvc.yaml`` into one record per concrete stage.

    Returns:
        Dicts with ``name``, ``group``, ``cmd`` and ``template`` keys. The
        template keeps ``${item}`` intact so the contract tests can assert on
        the command as written rather than on one expanded dataset.
    """
    pipeline = _load_pipeline()
    params = _load_params()
    stages: list[dict[str, str]] = []
    for name, spec in (pipeline.get("stages") or {}).items():
        if "foreach" in spec:
            body = spec.get("do") or {}
            template = str(body.get("cmd", ""))
            for item in _foreach_items(spec["foreach"], params):
                stages.append(
                    {
                        "name": f"{name}@{item}",
                        "group": name,
                        "cmd": template.replace("${item}", item),
                        "template": template,
                    }
                )
        else:
            cmd = str(spec.get("cmd", ""))
            stages.append({"name": name, "group": name, "cmd": cmd, "template": cmd})
    return stages


def dvc_commands() -> list[tuple[str, list[str]]]:
    """Return (stage group, CLI argv) for each distinct command in dvc.yaml."""
    seen: dict[str, list[str]] = {}
    for stage in load_stages():
        seen.setdefault(stage["group"], shlex.split(stage["template"]))
    return sorted(seen.items())


def resolve(argv: list[str]) -> object | None:
    """Walk the CLI command tree along *argv*, returning the command it names.

    Typed loosely and driven through the ``commands`` mapping on purpose: Typer
    0.26 vendors its own Click, so the concrete classes live under
    ``typer._click`` and naming them here would tie the test to a private path.

    Args:
        argv: Arguments after ``data-curation``, e.g. ``["response", "fit-curves"]``.

    Returns:
        The command the path names, or None if some segment does not exist.
    """
    command: object = typer.main.get_command(cli.app)
    for token in argv:
        if token.startswith("-"):
            break
        children = getattr(command, "commands", None)
        if not children:
            break
        if token not in children:
            return None
        command = children[token]
    return command


def split_command(argv: list[str]) -> tuple[object | None, list[str]]:
    """Like :func:`resolve`, but also return the tokens left after the command path."""
    command: object = typer.main.get_command(cli.app)
    for position, token in enumerate(argv):
        children = getattr(command, "commands", None)
        if token.startswith("-") or not children:
            return command, argv[position:]
        if token not in children:
            return None, []
        command = children[token]
    return command, []


def command_arguments(command: object, tokens: list[str]) -> dict[str, str]:
    """Bind the positional *tokens* of a ``dvc.yaml`` cmd to the command's argument names."""
    names = [
        param.name for param in getattr(command, "params", []) if getattr(param, "param_type_name", "") == "argument"
    ]
    positional = [token for token in tokens if not token.startswith("-")]
    return dict(zip(names, positional, strict=False))


def stage_closure(template: str) -> set[Path]:
    """Source files a stage's command runs: the import closure of its command module(s).

    Starts from the modules the command function imports (deferred imports
    included, the branch of a ``match`` on a command argument chosen by the
    arguments in *template*) and follows every in-package import from there,
    ``core`` included. The CLI layer is excluded; see ``test_no_stage_depends_on_the_cli_layer``.
    """
    command, rest = split_command(shlex.split(template)[3:])
    assert command is not None, f"unknown command: {template}"
    seeds = command_modules(getattr(command, "callback"), command_arguments(command, rest))  # noqa: B009
    return import_closure(seeds)


def is_group(command: object) -> bool:
    """Report whether *command* is a group rather than an invocable leaf."""
    return bool(getattr(command, "commands", None))


def _lookup(mapping: dict[str, Any], dotted: str) -> Any:
    """Walk a dotted key through nested dicts, returning None if any step is missing."""
    node: Any = mapping
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _params_entries(spec: dict[str, Any]) -> list[tuple[str, str]]:
    """Return (params_file, dotted_key) pairs from a stage's ``params:`` list.

    Bare keys (legacy form) are treated as belonging to ``params.yaml``. File-
    scoped entries are the dict-with-one-key form DVC uses for custom files.
    """
    entries: list[tuple[str, str]] = []
    for item in spec.get("params") or []:
        entries.extend(_one_params_item(item))
    return entries


def _one_params_item(item: object) -> list[tuple[str, str]]:
    """Expand one ``params:`` list entry into (file, key) pairs."""
    if isinstance(item, str):
        return [("params.yaml", item)]
    if not isinstance(item, dict):
        return []
    entries: list[tuple[str, str]] = []
    for path, keys in item.items():
        if not isinstance(keys, list):
            continue
        entries.extend((str(path), str(key)) for key in keys)
    return entries


def _stage_foreach_items(spec: dict[str, Any], params: dict[str, Any]) -> list[str | None]:
    """Return foreach items for *spec*, or ``[None]`` for a plain stage."""
    if "foreach" not in spec:
        return [None]
    return list(_foreach_items(spec["foreach"], params))


def _cached_config(caches: dict[str, dict[str, Any]], path: str) -> dict[str, Any]:
    """Load *path* once into *caches* and return it."""
    if path not in caches:
        with UPath(path).open() as handle:
            caches[path] = yaml.safe_load(handle)
    return caches[path]


def _assert_params_resolve(
    name: str,
    body: dict[str, Any],
    items: list[str | None],
    caches: dict[str, dict[str, Any]],
) -> None:
    """Assert every ``params:`` key on a stage resolves in its config file."""
    for item in items:
        for path, dotted in _params_entries(body):
            resolved_key = dotted.replace("${item}", item) if item is not None else dotted
            assert _lookup(_cached_config(caches, path), resolved_key) is not None, (
                f"{name} params key '{resolved_key}' missing from {path}"
            )


class TestDvcCommands:
    def test_dvc_yaml_is_readable(self) -> None:
        assert load_stages(), "no stages parsed from dvc.yaml"

    @pytest.mark.parametrize(("group", "argv"), dvc_commands())
    def test_every_stage_command_exists_in_the_cli(self, group: str, argv: list[str]) -> None:
        assert argv[:3] == ["uv", "run", "data-curation"], f"{group} does not invoke the CLI: {argv}"

        resolved = resolve(argv[3:])

        assert resolved is not None, f"{group} names a command the CLI does not define: {argv[3:]}"
        assert not is_group(resolved), f"{group} stops at a group rather than a command: {argv[3:]}"

    def test_no_stage_depends_on_the_cli_layer(self) -> None:
        """The CLI cannot change a stage's output, so listing it would only force re-runs."""
        pipeline = _load_pipeline()

        for name, spec in pipeline["stages"].items():
            deps = (spec.get("do") or spec).get("deps") or []
            offenders = [dep for dep in deps if "/cli" in dep]
            assert not offenders, f"{name} depends on the CLI layer: {offenders}"

    def test_every_source_dep_exists(self) -> None:
        """A moved module leaves a dangling dep that DVC only reports at run time."""
        pipeline = _load_pipeline()

        for name, spec in pipeline["stages"].items():
            deps = (spec.get("do") or spec).get("deps") or []
            for dep in deps:
                if dep.startswith("src/"):
                    assert UPath(dep).exists(), f"{name} depends on a missing path: {dep}"

    def test_foreach_groups_expand_over_their_dataset_list(self) -> None:
        fits = [stage for stage in load_stages() if stage["group"] == "fit_curves"]

        assert len(fits) > 1
        assert all("${item}" not in stage["cmd"] for stage in fits)
        assert any(stage["name"].endswith("@GDSC2") for stage in fits)

    def test_every_params_reference_resolves_in_its_config_file(self) -> None:
        """A split config file that drifts from dvc.yaml would only fail at repro time."""
        pipeline = _load_pipeline()
        params = _load_params()
        caches: dict[str, dict[str, Any]] = {"params.yaml": params}

        for name, spec in pipeline["stages"].items():
            body = spec.get("do") or spec
            _assert_params_resolve(name, body, _stage_foreach_items(spec, params), caches)


def _stage_groups() -> list[tuple[str, str, list[str]]]:
    """Each stage group of dvc.yaml as (name, command template, deps)."""
    groups = []
    for name, spec in _load_pipeline()["stages"].items():
        body = spec.get("do") or spec
        groups.append((name, str(body["cmd"]), list(body.get("deps") or [])))
    return groups


class TestImportClosure:
    """A stage's deps must name every module its command can reach.

    DVC hashes only the files listed under ``deps``. A module that a stage imports
    but does not list can change the stage's output without DVC noticing, leaving
    stale data that looks up to date. The CLI layer is exempt (it cannot change
    what a stage computes), as are docstring-only package ``__init__`` files.
    """

    @pytest.mark.parametrize(
        ("name", "template", "deps"), [pytest.param(*group, id=group[0]) for group in _stage_groups()]
    )
    def test_every_imported_module_is_a_dep(self, name: str, template: str, deps: list[str]) -> None:
        missing = sorted(
            str(path.relative_to(REPO_ROOT)) for path in stage_closure(template) if not covered_by(path, deps)
        )

        assert not missing, f"{name} imports modules missing from its deps (add to dvc.yaml): {missing}"

    def test_the_closure_follows_deferred_and_core_imports(self) -> None:
        closure = {
            str(path.relative_to(REPO_ROOT)) for path in stage_closure("uv run data-curation response fit-curves X")
        }

        assert "src/data_curation/response/fit_curves.py" in closure
        assert "src/data_curation/core/layout.py" in closure
        assert not any("/cli/" in path for path in closure)

    def test_process_picks_the_module_of_its_modality(self) -> None:
        closure = {path.name for path in stage_closure("uv run data-curation omics process mutations ${item}")}

        assert "process_mutations.py" in closure
        assert not closure & {"process_cnv.py", "process_proteomics.py", "process_transcriptomics.py"}

    def test_a_directory_dep_covers_the_files_under_it(self) -> None:
        path = REPO_ROOT / "src/data_curation/response/harmonize/identity.py"

        assert covered_by(path, ["src/data_curation/response/harmonize"])
        assert not covered_by(path, ["src/data_curation/core/config.py"])
