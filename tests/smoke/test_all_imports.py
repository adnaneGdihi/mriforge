"""Every module under ``src/mriforge`` must import cleanly.

Catches syntax errors, broken imports and import-time side effects before a
run reaches the cluster.

Two things this file used to get wrong, both worth keeping in view:

* ``SRC_ROOT``/``prefix`` predated the 2026-05 ``src`` -> ``src/mriforge``
  refactor, so it imported the package a *second* time as ``src.mriforge.*``.
  Every ``@register_*`` module then collided with its own real registration
  (861 failures in cluster job 8000966), and every module without a decorator
  succeeded, leaving ~1000 shadow module objects resident for the session.
* The import ran in-process. The conftest chain has already imported most of
  ``mriforge`` by then, so ``importlib.import_module`` was a ``sys.modules``
  dict hit -- verified by breaking ``core/compute_device.py`` and watching the
  in-process form still report green. The sweep therefore runs in one pristine
  subprocess (:mod:`tests.smoke._import_sweep`), shared across the whole
  parametrisation by a session fixture.
"""

from __future__ import annotations

import importlib.util
import json
import pkgutil
import subprocess
import sys
import tomllib
import warnings
from pathlib import Path

import pytest
from _pytest.config import parse_warning_filter

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "src" / "mriforge"
SWEEP_SCRIPT = Path(__file__).resolve().parent / "_import_sweep.py"
SWEEP_TIMEOUT_S = 900

if not PACKAGE_ROOT.is_dir():
    # A wrong path constant must not degrade into a sweep over zero modules --
    # that is how this suite would go green while testing nothing.
    raise RuntimeError(f"package root not found: {PACKAGE_ROOT}")


def discover_modules(package_root: Path, prefix: str) -> list[str]:
    """Every importable module under ``package_root``, dotted from ``prefix``."""
    modules: list[str] = []
    for _, name, is_package in pkgutil.iter_modules([str(package_root)]):
        dotted = f"{prefix}.{name}"
        modules.append(dotted)
        if is_package:
            modules.extend(discover_modules(package_root / name, dotted))
    return modules


ALL_MODULES = sorted(discover_modules(PACKAGE_ROOT, PACKAGE_ROOT.name))


@pytest.fixture(scope="session")
def import_failures(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, dict[str, str]]:
    """``{module: failure}`` from one fresh-interpreter pass over ``ALL_MODULES``."""
    workdir = tmp_path_factory.mktemp("import_sweep")
    modules_path = workdir / "modules.json"
    failures_path = workdir / "failures.json"
    modules_path.write_text(json.dumps(ALL_MODULES), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(SWEEP_SCRIPT), str(modules_path), str(failures_path)],
        capture_output=True,
        text=True,
        timeout=SWEEP_TIMEOUT_S,
        cwd=REPO_ROOT,
        check=False,
    )
    if completed.returncode != 0 or not failures_path.is_file():
        raise RuntimeError(
            f"import sweep did not complete (exit {completed.returncode}); "
            f"stderr tail:\n{completed.stderr[-4000:]}"
        )
    return json.loads(failures_path.read_text(encoding="utf-8"))


def _load_sweep_module():
    """Import ``_import_sweep`` by path -- ``tests/smoke`` is not a package."""
    spec = importlib.util.spec_from_file_location(
        "_import_sweep_under_test", SWEEP_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.smoke
def test_declared_warning_filters_survive_the_sweep_parser() -> None:
    """Every ``filterwarnings`` entry must parse, with its ``message`` field intact.

    The sweep's session fixture raises when the parser rejects an entry, so one
    bad entry errors *every* module's test at setup -- 1902 of them in cluster
    job 8012333, from the single message-keyed filter added in ``b58e99b24``.
    Nothing caught it because the only consumer is a subprocess.

    Asserting the message field lands (not merely that parsing succeeded) is
    what pins the second half of that defect: the previous hand-rolled parser
    dropped ``message`` and ``lineno`` from every entry it *did* accept.
    """
    sweep = _load_sweep_module()
    declared = tomllib.loads(sweep.PYPROJECT.read_text(encoding="utf-8"))["tool"][
        "pytest"
    ]["ini_options"]["filterwarnings"]

    with warnings.catch_warnings():
        warnings.resetwarnings()
        sweep.apply_session_warning_filters(sweep.PYPROJECT)
        applied = {f[1].pattern for f in warnings.filters if f[1] is not None}

    expected = {parse_warning_filter(spec, escape=False)[1] for spec in declared}
    dropped = {message for message in expected if message and message not in applied}
    assert (
        not dropped
    ), f"message-keyed filters never reached warnings.filters: {dropped}"


@pytest.mark.smoke
def test_discovery_covers_the_package() -> None:
    """The sweep is only meaningful if discovery actually found the package."""
    assert (
        len(ALL_MODULES) > 500
    ), f"only {len(ALL_MODULES)} modules discovered under {PACKAGE_ROOT}"
    assert all(m.startswith("mriforge.") for m in ALL_MODULES)


@pytest.mark.smoke
@pytest.mark.parametrize("module_name", ALL_MODULES)
def test_module_imports(
    module_name: str, import_failures: dict[str, dict[str, str]]
) -> None:
    failure = import_failures.get(module_name)
    if failure is None:
        return
    missing = failure["missing_module"]
    if missing and not missing.startswith("mriforge"):
        pytest.skip(f"third-party dependency '{missing}' is not installed")
    pytest.fail(f"{module_name}: {failure['type']}: {failure['message']}")
