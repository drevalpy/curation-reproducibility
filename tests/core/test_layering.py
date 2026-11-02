"""Tests that the package layering stays acyclic.

Two rules keep the domain packages readable, and neither is visible at run time -
a cycle still imports fine, so the layering diagram can quietly become fiction:

- ``core`` holds what more than one domain needs while knowing about none of
  them. That is also what separates ``core`` from a ``utils`` bag: "is this
  shared by several domains and ignorant of all of them?" is a question with an
  answer, where "is this a utility?" is not.
- The domain packages own no CLI code, so they do not depend on ``cli``. The one
  intentional exception is the progress helper the rate-limited PubChem loops
  report through, which is listed explicitly below rather than left implicit.

Both are expressed as "which prefixes may this module import" rather than as
deny-lists of domain names, so adding a new domain package is covered without
anyone remembering to update these tests.
"""

from pathlib import Path

import pytest
from import_graph import PACKAGE, SRC_DIR, imported_names

import data_curation.core

CORE_DIR = Path(data_curation.core.__file__).parent

# Domain packages: everything that is neither the bottom layer nor the CLI.
DOMAINS = ("cell_lines", "omics", "response", "drugs", "assembly")

# The single sanctioned domain -> cli import. Progress reporting is presentation,
# and the alternative - passing a reporter down through every mapper signature -
# buys nothing: see cli/_progress.py, which degrades to log lines off a TTY.
PROGRESS_HELPER = f"{PACKAGE}.cli._progress"


def offending_imports(path: Path, *, allowed: tuple[str, ...], max_relative_level: int = 1) -> list[str]:
    """Return the imports in *path* that break its layer's rule.

    Args:
        path: Module to inspect.
        allowed: Dotted prefixes this module may import from.
        max_relative_level: Deepest relative import that cannot escape the
            module's own package. Level 1 (``from . import x``) stays put; a
            deeper one climbs out to where another layer is reachable.

    Returns:
        Sorted offending names, empty when the module is clean.
    """
    offenders = set()
    for name in imported_names(path):
        if name.startswith("."):
            if len(name) - len(name.lstrip(".")) > max_relative_level:
                offenders.add(name)
        elif not any(name == prefix or name.startswith(f"{prefix}.") for prefix in allowed):
            offenders.add(name)
    return sorted(offenders)


def domain_modules() -> list[Path]:
    """Return every module in a domain package."""
    return sorted(path for domain in DOMAINS for path in (SRC_DIR / domain).rglob("*.py"))


class TestCoreLayering:
    def test_the_core_package_is_where_this_test_thinks_it_is(self) -> None:
        """Guards against the check silently passing over an empty directory."""
        names = {path.name for path in sorted(CORE_DIR.rglob("*.py"))}
        assert {"config.py", "paths.py", "download.py"} <= names, f"unexpected core contents: {sorted(names)}"

    @pytest.mark.parametrize("path", sorted(CORE_DIR.rglob("*.py")), ids=lambda path: path.name)
    def test_core_imports_nothing_from_the_rest_of_the_package(self, path: Path) -> None:
        offenders = offending_imports(path, allowed=(f"{PACKAGE}.core",))

        assert not offenders, (
            f"core/{path.name} imports {offenders}. core sits at the bottom of the "
            "layering and must know about no domain; move the shared piece down "
            "into core or the dependent piece up into the domain."
        )


class TestDomainLayering:
    def test_the_domains_are_where_this_test_thinks_they_are(self) -> None:
        for domain in DOMAINS:
            assert (SRC_DIR / domain / "__init__.py").is_file(), f"missing domain package: {domain}"

    def test_no_domain_package_defines_commands(self) -> None:
        """The command surface lives in cli/, one module per group."""
        stray = [str(path.relative_to(SRC_DIR)) for path in domain_modules() if path.name == "cli.py"]

        assert not stray, f"CLI code inside a domain package: {stray}. Move it to cli/<domain>.py."

    @pytest.mark.parametrize("path", domain_modules(), ids=lambda path: str(path.relative_to(SRC_DIR)))
    def test_a_domain_reaches_the_cli_only_for_progress(self, path: Path) -> None:
        """Anything beyond the progress helper means presentation is leaking down."""
        allowed = (f"{PACKAGE}.core", *(f"{PACKAGE}.{domain}" for domain in DOMAINS), PROGRESS_HELPER)
        offenders = [name for name in offending_imports(path, allowed=allowed) if name.startswith(f"{PACKAGE}.cli")]

        assert not offenders, (
            f"{path.relative_to(SRC_DIR)} imports {offenders} from the CLI layer. "
            f"Only {PROGRESS_HELPER} is sanctioned; argument parsing, console and "
            "logging setup belong to the CLI and must not be reached from a stage."
        )


class TestOffendingImports:
    """The detector is only worth having if it actually detects."""

    def test_a_domain_import_is_reported(self, tmp_path) -> None:
        module = tmp_path / "leaky.py"
        module.write_text("from data_curation.drugs.mapping import overrides\n")

        assert offending_imports(module, allowed=(f"{PACKAGE}.core",)) == ["data_curation.drugs.mapping"]

    def test_a_deferred_domain_import_is_reported(self, tmp_path) -> None:
        module = tmp_path / "deferred.py"
        module.write_text("def go():\n    import data_curation.response.fit_curves\n")

        assert offending_imports(module, allowed=(f"{PACKAGE}.core",)) == ["data_curation.response.fit_curves"]

    def test_a_relative_import_climbing_out_is_reported(self, tmp_path) -> None:
        module = tmp_path / "relative.py"
        module.write_text("from ..omics import cmp_matrix\n")

        assert offending_imports(module, allowed=(f"{PACKAGE}.core",)) == ["..omics"]

    def test_sibling_and_third_party_imports_are_allowed(self, tmp_path) -> None:
        module = tmp_path / "clean.py"
        module.write_text(
            "import yaml\n"
            "from upath import UPath\n"
            "from data_curation.core.config import source_config\n"
            "from . import paths\n"
        )

        assert offending_imports(module, allowed=(f"{PACKAGE}.core",)) == []

    def test_an_allowed_prefix_does_not_match_by_substring(self, tmp_path) -> None:
        """`cli._progressive` must not slip through on the `cli._progress` prefix."""
        module = tmp_path / "sneaky.py"
        module.write_text("from data_curation.cli._progressive import track\n")

        assert offending_imports(module, allowed=(PROGRESS_HELPER,)) == ["data_curation.cli._progressive"]
