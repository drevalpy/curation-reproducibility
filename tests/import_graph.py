"""Static import graph of the ``data_curation`` package, shared by the architecture tests.

Everything here reads source with :mod:`ast` and never imports the package under
test, so it sees deferred imports (inside function bodies) exactly like top-level
ones. Two tests build on it:

- ``tests/core/test_layering.py`` asks which prefixes each module imports.
- ``tests/cli/test_dvc_contract.py`` asks for the transitive closure of modules a
  pipeline stage runs, to compare against the stage's ``deps``.
"""

import ast
import inspect
import textwrap
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import data_curation

PACKAGE = "data_curation"
SRC_DIR = Path(data_curation.__file__).parent
REPO_ROOT = SRC_DIR.parents[1]
# The CLI parses arguments and configures logging; it cannot change what a stage
# computes, so it is never part of a stage's closure (see dvc.yaml).
CLI_PACKAGE = f"{PACKAGE}.cli"


def imported_names(path: Path) -> set[str]:
    """Return every in-package name *path* imports, relative ones included.

    Walks the whole tree rather than the module header, so an import deferred
    inside a function body is caught too. Relative imports are returned in their
    dotted form (``"..omics"``) since the level is what matters for layering.

    Args:
        path: Module to inspect.

    Returns:
        Dotted names belonging to this package, plus any relative import.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names if alias.name.split(".")[0] == PACKAGE)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                names.add("." * node.level + (node.module or ""))
            elif node.module and node.module.split(".")[0] == PACKAGE:
                names.add(node.module)
    return names


def module_name(path: Path) -> str:
    """Dotted name of the source file *path* inside the package."""
    parts = path.resolve().relative_to(SRC_DIR.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def module_path(name: str) -> Path | None:
    """Source file of the in-package module *name*, or None if it is not one."""
    base = SRC_DIR.parent.joinpath(*name.split("."))
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


def has_code(path: Path) -> bool:
    """Report whether *path* holds anything beyond a docstring.

    A docstring-only package ``__init__`` cannot change what a stage computes, so
    the DVC contract does not demand it as a dependency.
    """
    body = ast.parse(path.read_text()).body
    return any(not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)) for node in body)


def _package_of(path: Path) -> str:
    """Dotted package that relative imports in *path* are resolved against."""
    name = module_name(path)
    return name if path.name == "__init__.py" else name.rpartition(".")[0]


def _resolve_from(node: ast.ImportFrom, package: str) -> list[str]:
    """Absolute in-package module names an ``ImportFrom`` node can load."""
    if node.level:
        anchor = package.split(".")
        anchor = anchor[: len(anchor) - (node.level - 1)]
        base = ".".join([*anchor, *([node.module] if node.module else [])])
    else:
        base = node.module or ""
    if base.split(".")[0] != PACKAGE:
        return []
    # `from pkg import name` loads pkg and, when `name` is a submodule, pkg.name.
    return [base, *(f"{base}.{alias.name}" for alias in node.names if module_path(f"{base}.{alias.name}"))]


def direct_imports(tree: ast.AST, package: str) -> set[str]:
    """In-package modules imported anywhere under *tree*, resolved to absolute names."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names if alias.name.split(".")[0] == PACKAGE)
        elif isinstance(node, ast.ImportFrom):
            found.update(_resolve_from(node, package))
    return {name for name in found if module_path(name) is not None}


def _with_parent_packages(names: Iterable[str]) -> set[str]:
    """Add the enclosing packages of each module, whose ``__init__`` runs on import."""
    out: set[str] = set()
    for name in names:
        parts = name.split(".")
        out.update(".".join(parts[:i]) for i in range(1, len(parts) + 1))
    return out


def import_closure(seeds: Iterable[str]) -> set[Path]:
    """Source files reachable from *seeds* through in-package imports.

    The CLI package is not followed. Package ``__init__`` files are followed for
    their imports but only returned when they contain code.

    Args:
        seeds: Dotted module names to start from.

    Returns:
        The set of source files, seeds included.
    """
    files = {path for name in _reachable_modules(seeds) if (path := module_path(name)) is not None}
    return {path for path in files if path.name != "__init__.py" or has_code(path)}


def _imports_of(name: str) -> set[str]:
    """Modules (with their parent packages) imported by the in-package module *name*."""
    path = module_path(name)
    if path is None:
        return set()
    return _with_parent_packages(direct_imports(ast.parse(path.read_text()), _package_of(path)))


def _reachable_modules(seeds: Iterable[str]) -> set[str]:
    """Module names reachable from *seeds*, never entering the CLI package."""
    pending = [name for name in _with_parent_packages(seeds) if not _in_cli(name)]
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        pending.extend(dep for dep in _imports_of(name) if dep not in seen and not _in_cli(dep))
    return seen


def _in_cli(name: str) -> bool:
    return name == CLI_PACKAGE or name.startswith(f"{CLI_PACKAGE}.")


def _matches_case(pattern: ast.pattern, value: str) -> bool:
    """Report whether a ``case`` pattern selects the CLI argument *value*."""
    if isinstance(pattern, ast.MatchValue) and isinstance(pattern.value, ast.Attribute):
        return pattern.value.attr.lower() == value.lower()
    if isinstance(pattern, ast.MatchOr):
        return any(_matches_case(sub, value) for sub in pattern.patterns)
    return isinstance(pattern, ast.MatchAs) and pattern.pattern is None


def _selected_statements(body: list[ast.stmt], arguments: Mapping[str, str]) -> list[ast.AST]:
    """Statements that run for *arguments*, dropping ``match`` cases that do not apply.

    Only a ``match`` whose subject is a command argument is narrowed; anything
    else is kept whole, which errs towards demanding more dependencies.
    """
    kept: list[ast.AST] = []
    for stmt in body:
        if isinstance(stmt, ast.Match) and isinstance(stmt.subject, ast.Name) and stmt.subject.id in arguments:
            value = arguments[stmt.subject.id]
            for case in stmt.cases:
                if _matches_case(case.pattern, value):
                    kept.extend(_selected_statements(case.body, arguments))
        else:
            kept.append(stmt)
    return kept


def command_modules(callback: Any, arguments: Mapping[str, str]) -> set[str]:
    """In-package modules a CLI command function imports for the given arguments.

    Args:
        callback: The command function (Typer wrappers are unwrapped).
        arguments: Parameter name -> value as written in ``dvc.yaml``, used to
            pick the branch of a ``match`` on a command argument.

    Returns:
        Dotted module names, CLI modules excluded.
    """
    func = inspect.unwrap(callback)
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    function = next(node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
    package = func.__module__.rpartition(".")[0]
    found: set[str] = set()
    for stmt in _selected_statements(function.body, arguments):
        found |= direct_imports(stmt, package)
    return {name for name in found if not _in_cli(name)}


def covered_by(path: Path, deps: Iterable[str]) -> bool:
    """Report whether *path* is a listed dep or lies under a listed directory dep."""
    rel = path.resolve().relative_to(REPO_ROOT.resolve())
    for dep in deps:
        dep_path = Path(dep.rstrip("/"))
        if rel == dep_path or dep_path in rel.parents:
            return True
    return False
