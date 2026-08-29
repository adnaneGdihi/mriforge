"""The Makefile targets CI invokes must not assume a local .venv.

`make test-nightly` is the entry point of `manual-full-suite.yml` (and of the cluster
array). A hosted runner has no .venv, so an unguarded `. .venv/bin/activate` kills the
recipe on its first line -- the suite never runs, and the failure reads as a shell error
rather than a test failure.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MAKEFILE = _REPO_ROOT / "Makefile"

# Every lane target that a CI workflow may invoke.
_CI_TARGETS = (
    "test-precommit",
    "test-pr",
    "test-nightly",
    "test-release",
    "cov-unit",
    "cov-full",
)


def _recipe(target: str) -> str:
    """The tab-indented body of a Makefile target."""
    text = _MAKEFILE.read_text()
    match = re.search(rf"^{re.escape(target)}:.*?\n((?:\t.*\n|\n)*)", text, re.MULTILINE)
    assert match is not None, f"target {target!r} not found in Makefile"
    return match.group(1)


@pytest.mark.parametrize("target", _CI_TARGETS)
def test_ci_target_does_not_activate_the_venv_unconditionally(target: str) -> None:
    recipe = _recipe(target)
    assert ". .venv/bin/activate &&" not in recipe, (
        f"`make {target}` activates .venv unconditionally, so it dies on a hosted "
        f"runner where no .venv exists. Use the $(VENV) guard."
    )


def test_venv_guard_is_defined_and_conditional() -> None:
    text = _MAKEFILE.read_text()
    match = re.search(r"^VENV\s*:?=(.*)$", text, re.MULTILINE)
    assert match is not None, "Makefile defines no VENV guard"
    assert "-f .venv/bin/activate" in match.group(1)


def test_format_target_uses_ruff_not_black() -> None:
    """One formatter. black is not a declared dependency and fights ruff-format."""
    recipe = _recipe("format")
    assert "ruff format" in recipe
    assert "black" not in recipe


def test_black_is_not_a_declared_dependency() -> None:
    """`make format` ran black while no extra declared it -- it was never installable."""
    pyproject = (_REPO_ROOT / "pyproject.toml").read_text()
    assert "black" not in pyproject
