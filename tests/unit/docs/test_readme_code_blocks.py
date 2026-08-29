"""The README's code blocks are executed, not proof-read.

Everything this module checks was false in the README at the same time, in the
most-read file in the repository:

* the first Python block omitted ``populate_model_registry()``. ``MODEL_REGISTRY``
  is **empty** on a bare import, so ``MODEL_REGISTRY.get(...)`` returned ``None``
  and the next line raised ``TypeError: 'NoneType' object is not subscriptable``.
  The identical defect had already been found and fixed in
  ``examples/quickstart_reconstruction.py``; nothing looked one file over.
* the YAML block declared ``config_version: '6.0'``, which the loader refuses
  (``Accepted values: ['1.0']``), and the ``mriforge audit`` line beneath it
  exited **2**.
* the extras block named ``[diffusion]`` as "+ einops" (it is ``diffusers``),
  claimed ``[all]`` excluded ``docs`` (it includes it), and omitted ten of the
  seventeen extras that exist.
* the contributing section told a contributor to run
  ``scripts/ci/smoke_test_vf_configs.sh``, which is not in the published tree.
* the CLI row advertised a ``preprocess`` verb. There is no such verb.

Every one of those is a claim the tree can answer. None of them could fail a
build, because prose is not executed and Markdown is not linted.

Scope, deliberately
-------------------
Only the ``python`` blocks are *executed*. ``bash`` blocks are checked
structurally -- verbs against the real parser, paths against the tree -- because
running them would train models and install packages. That is a real bound, not
an oversight: ``pip install`` lines are verified to name an extra that exists,
not to resolve on PyPI, and ``test_no_bash_block_names_a_path_that_is_absent``
covers the file the contributor is told to run.

The path check answers about **the tree it runs in**, and that is the whole of
what it can answer. ``scripts/ci/smoke_test_vf_configs.sh`` -- the very file
whose absence broke contributing step 3 -- is *present* in the private tree and
dropped by the export allowlist, so planting it here scored green and the plant,
not the test, was wrong. This guard therefore only catches an unshipped path
when it is run **inside the exported tree**. It is not a substitute for driving
the question off the export file list, and reading it as one is how a private-
tree pass gets mistaken for a public-tree claim.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

from mriforge.config.schemas.base import ACCEPTED_CONFIG_VERSIONS
from tests.utils.corpus import repo_root

_FENCE = re.compile(r"^```(\w+)\n(.*?)^```", re.MULTILINE | re.DOTALL)


def _readme() -> str:
    path = repo_root() / "README.md"
    if not path.is_file():  # pragma: no cover - defensive
        pytest.skip(f"README.md not present at {path}")
    return path.read_text(encoding="utf-8")


def _blocks(language: str) -> list[str]:
    return [body for lang, body in _FENCE.findall(_readme()) if lang == language]


def test_the_readme_has_blocks_of_each_kind_this_module_checks() -> None:
    """Vacuity guard: an empty extraction makes every test below pass silently.

    A renamed fence, a switch to indented code, or a regex that stops matching
    would otherwise turn this whole module green while checking nothing.
    """
    for language, minimum in (("python", 2), ("yaml", 1), ("bash", 3)):
        found = _blocks(language)
        assert len(found) >= minimum, (
            f"extracted {len(found)} ```{language} block(s) from README.md, expected >= {minimum} "
            "-- the fence regex has stopped matching and every check below is now vacuous"
        )


@pytest.mark.parametrize("index", range(2))
def test_every_python_block_runs_in_a_cold_interpreter(index: int, tmp_path: Path) -> None:
    """A reader pastes the block into a fresh shell. That is what is run here.

    Cold, because the defect this catches exists ONLY in an interpreter that has
    imported nothing else: in-process, some sibling test has already populated
    the registry and the missing ``populate_model_registry()`` is invisible.
    """
    blocks = _blocks("python")
    if index >= len(blocks):
        pytest.skip(f"README has {len(blocks)} python block(s)")

    script = tmp_path / f"readme_block_{index}.py"
    script.write_text(blocks[index], encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=repo_root(),
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            "PYTHONPATH": str(repo_root() / "src"),
            "MRIFORGE_SUPPRESS_CLINICAL_WARNING": "1",
        },
        timeout=600,
    )
    assert result.returncode == 0, (
        f"README python block #{index} exits {result.returncode}.\n"
        f"--- block ---\n{blocks[index]}\n--- stderr ---\n{result.stderr[-2000:]}"
    )


def test_every_yaml_block_declares_a_config_version_the_loader_accepts() -> None:
    """A README YAML that cannot load is a first-run failure, not a typo."""
    offenders = []
    for i, body in enumerate(_blocks("yaml")):
        try:
            parsed = yaml.safe_load(body)
        except yaml.YAMLError as exc:
            offenders.append(f"block #{i}: not parseable YAML -- {exc}")
            continue
        if not isinstance(parsed, dict):
            continue
        declared = parsed.get("config_version")
        if declared is not None and str(declared) not in ACCEPTED_CONFIG_VERSIONS:
            offenders.append(f"block #{i}: config_version {declared!r}")
    assert not offenders, (
        f"README YAML declares a version the loader rejects "
        f"(accepted: {sorted(ACCEPTED_CONFIG_VERSIONS)}):\n  " + "\n  ".join(offenders)
    )


def test_every_extra_the_readme_tells_you_to_install_exists() -> None:
    """``pip install mriforge[x]`` for an ``x`` that does not exist fails outright.

    Parsed with ``tomllib``, never grepped: a bare name match in
    ``pyproject.toml`` also hits comments and pytest marker names, which is how
    three packages read as "declared" while being absent.
    """
    pyproject = tomllib.loads((repo_root() / "pyproject.toml").read_text(encoding="utf-8"))
    declared = set(pyproject.get("project", {}).get("optional-dependencies", {}))
    assert declared, "no optional-dependencies parsed from pyproject.toml"

    advertised = set(re.findall(r"pip install [^\n]*mriforge\[([a-z0-9,\-_]+)\]", _readme()))
    advertised = {name for group in advertised for name in group.split(",")}
    assert advertised, "the README advertises no extras -- the pattern has stopped matching"

    missing = sorted(advertised - declared)
    assert not missing, (
        f"README tells the reader to install extra(s) that pyproject.toml does not declare: {missing}\n"
        f"declared: {sorted(declared)}"
    )


def test_every_mriforge_verb_in_a_bash_block_is_a_real_verb() -> None:
    """The README advertised a ``preprocess`` verb the CLI has never had."""
    import argparse

    from mriforge.cli.app import build_parser

    parser = build_parser()
    verbs = {
        name
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
        for name in action.choices
    }
    assert verbs, "no subcommands found on the parser"

    used = set()
    for body in _blocks("bash") + _blocks("markdown"):
        for line in body.splitlines():
            match = re.match(r"\s*mriforge\s+([a-z][a-z0-9_-]*)", line)
            if match:
                used.add(match.group(1))

    unknown = sorted(used - verbs)
    assert not unknown, f"README runs mriforge verb(s) that do not exist: {unknown}\nreal verbs: {sorted(verbs)}"


def test_no_bash_block_names_a_path_that_is_absent() -> None:
    """The contributing steps pointed at a script that is not in the tree."""
    root = repo_root()
    missing = []
    for body in _blocks("bash"):
        for token in re.findall(r"(?<![\w/.-])((?:src|tests|scripts|docs|examples|experiments)/[\w./-]+)", body):
            if not (root / token).exists():
                missing.append(token)
    assert not missing, (
        "README bash blocks reference path(s) absent from the tree: " + ", ".join(sorted(set(missing)))
    )


def test_the_registry_counts_the_readme_prints_are_the_live_ones() -> None:
    """Numbers drift daily; a stale one is the over-claim this release is about.

    Only the model count is pinned, and only as a floor with a stated ceiling:
    an exact equality would fail on every legitimate addition, and a check
    nobody can keep green gets deleted rather than fixed.
    """
    from mriforge.models.init_registry import populate_model_registry
    from mriforge.models.registry import MODEL_REGISTRY

    populate_model_registry()
    live = len(MODEL_REGISTRY)

    # `\s+` and not a space: the README wraps between "model" and
    # "architectures", so a literal-space pattern matches nothing and this test
    # would pass by finding no claim at all.
    claimed = re.search(r"(\d[\d,]*)\s+model\s+architectures", _readme())
    assert claimed, "the README no longer states a model-architecture count for this test to check"
    stated = int(claimed.group(1).replace(",", ""))

    assert abs(stated - live) <= 25, (
        f"README claims {stated} registered models; the registry returns {live}. "
        "Re-measure and update the table -- the counts drift, and a stale one is "
        "exactly the over-claim this file exists to prevent."
    )
