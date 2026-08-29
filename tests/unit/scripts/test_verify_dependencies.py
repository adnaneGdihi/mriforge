"""Tests for scripts/verify/verify_dependencies.py.

The checker verifies the declared dependency set (from ``pyproject.toml``, the
SSOT) is installed, version-correct, and — with ``--import-check`` — importable.
These tests pin the two failure modes it must distinguish:

  * **missing / version-mismatch** from package metadata alone, and
  * **installed-but-unimportable** (the live ``torchmetrics`` case: satisfied by
    version yet failing to import under an incompatible ``huggingface-hub``),
    which only an actual import can catch.

The module is loaded by file path (it lives under ``scripts/``, outside the
import package) following the same convention as the other script tests.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "verify" / "verify_dependencies.py"


def _load():
    spec = importlib.util.spec_from_file_location("verify_dependencies", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


vd = _load()


# --------------------------------------------------------------------------- #
# Requirement / version parsing.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("spec", "name", "has_specifier"),
    [
        ("torch>=2.5", "torch", True),
        ("torchmetrics>=1.0,<2.0", "torchmetrics", True),
        ("numpy", "numpy", False),
        ("mriforge[mri,viz,test]", "mriforge", False),
    ],
)
def test_parse_requirement(spec, name, has_specifier):
    got_name, specifier, marker_ok = vd.parse_requirement(spec)
    assert vd._canon(got_name) == vd._canon(name)
    assert bool(specifier) == has_specifier
    assert marker_ok is True


def test_canon_pep503():
    assert vd._canon("PyWavelets") == "pywavelets"
    assert vd._canon("huggingface_hub") == "huggingface-hub"
    assert vd._canon("scikit.learn") == "scikit-learn"


@pytest.mark.parametrize(
    ("installed", "specifier", "ok"),
    [
        ("2.5.0", ">=2.5", True),
        ("2.4.0", ">=2.5", False),
        ("1.9.0", ">=1.0,<2.0", True),
        ("2.0.0", ">=1.0,<2.0", False),
        ("1.0.0", "", True),  # no specifier -> always satisfied
    ],
)
def test_version_satisfies(installed, specifier, ok):
    assert vd._version_satisfies(installed, specifier) is ok


# --------------------------------------------------------------------------- #
# pyproject reading (the SSOT).
# --------------------------------------------------------------------------- #
def test_load_dependency_groups_has_core_and_extras():
    groups = vd.load_dependency_groups()
    assert "core" in groups
    assert groups["core"], "core dependency list must be non-empty"
    # Core deps declared in pyproject must be present in the parsed set.
    core_names = {vd._canon(vd.parse_requirement(s)[0]) for s in groups["core"]}
    assert {"torch", "numpy", "pydantic", "torchmetrics"} <= core_names
    # Optional groups exist (drift guard against the pyproject layout changing).
    assert {"mri", "viz", "test"} <= set(groups)


# --------------------------------------------------------------------------- #
# `all` meta-group completeness (the "really install all on the cluster" guard).
#
# `all` must reference EVERY optional extra that resolves in a single
# `pip install` (build isolation ON). `mamba` is the sole exception — it
# compiles the CUDA selective-scan kernel and needs `--no-build-isolation` +
# nvcc, so it cannot ride in a one-shot `.[all]`. `dev` must reference `all`
# (not a hand-list) so the two meta-groups cannot drift.
# --------------------------------------------------------------------------- #
# Extras deliberately kept OUT of `all` because they cannot resolve one-shot.
_ONESHOT_EXCLUDED = {"mamba"}
# Meta-groups are themselves self-references, not concrete feature extras.
_META_GROUPS = {"all", "dev"}


def _referenced_extras(req_strings: list[str]) -> set[str]:
    """Extract the extras named in a ``mriforge[a,b,c]`` self-reference list."""
    referenced: set[str] = set()
    for spec in req_strings:
        name, _, _ = vd.parse_requirement(spec)
        if vd._canon(name) != "mriforge":
            continue
        inside = spec[spec.index("[") + 1 : spec.index("]")]
        referenced |= {e.strip() for e in inside.split(",") if e.strip()}
    return referenced


def test_all_extra_covers_every_oneshot_installable_extra():
    """`all` must include every one-shot-installable optional extra.

    Adding a new extra to pyproject without wiring it into `all` fails here — the
    guard that keeps a cluster `.[all]` install genuinely complete. `mamba` is
    the documented exception (CUDA-kernel build, not one-shot resolvable).
    """
    groups = vd.load_dependency_groups()
    assert "all" in groups, "the `all` convenience group must exist"

    concrete_extras = set(groups) - {"core"} - _META_GROUPS
    expected_in_all = concrete_extras - _ONESHOT_EXCLUDED
    referenced = _referenced_extras(groups["all"])

    missing = expected_in_all - referenced
    assert not missing, f"`all` is missing one-shot-installable extras: {sorted(missing)}"


def test_all_extra_excludes_mamba():
    """`mamba` stays OUT of `all` — it needs `--no-build-isolation` + nvcc and so
    cannot resolve in a one-shot `pip install .[all]`."""
    groups = vd.load_dependency_groups()
    assert "mamba" not in _referenced_extras(groups["all"])


def test_dev_references_all_to_avoid_drift():
    """`dev` must be defined in terms of `all` (plus lint tooling), so the full
    feature set and the dev set cannot silently diverge."""
    groups = vd.load_dependency_groups()
    assert "all" in _referenced_extras(groups["dev"])


# --------------------------------------------------------------------------- #
# check_requirement — status classification.
# --------------------------------------------------------------------------- #
def test_check_requirement_ok_for_installed_stdlib_adjacent():
    # pytest is always installed in the test env and satisfies a loose bound.
    rec = vd.check_requirement("pytest>=1.0", self_name="mriforge", import_check=False)
    assert rec["status"] == vd.OK
    assert rec["installed"] is not None


def test_check_requirement_missing():
    rec = vd.check_requirement(
        "this-package-does-not-exist-xyz>=1.0",
        self_name="mriforge",
        import_check=False,
    )
    assert rec["status"] == vd.MISSING


def test_check_requirement_version_mismatch():
    # pytest is installed but cannot satisfy an absurd future lower bound.
    rec = vd.check_requirement(
        "pytest>=999999", self_name="mriforge", import_check=False
    )
    assert rec["status"] == vd.MISMATCH
    assert rec["installed"] is not None


def test_check_requirement_self_reference_skipped():
    rec = vd.check_requirement(
        "mriforge[mri,viz]", self_name="mriforge", import_check=False
    )
    assert rec["status"] == vd.SKIPPED


def test_import_check_detects_installed_but_unimportable(monkeypatch):
    """The torchmetrics failure mode: version-satisfied yet import raises.

    We do not depend on torchmetrics actually being broken in CI — we simulate a
    distribution that IS installed and satisfied but fails to import, and assert
    the checker classifies it as IMPORT_FAIL only under ``--import-check`` (and
    as OK without it, proving the two modes are genuinely different).
    """
    monkeypatch.setattr(vd.md, "version", lambda _name: "1.9.0")
    monkeypatch.setattr(
        vd,
        "_probe_import",
        lambda _name: (False, "import x: ImportError: huggingface-hub<1.0 required"),
    )
    spec = "torchmetrics>=1.0,<2.0"

    without = vd.check_requirement(spec, self_name="mriforge", import_check=False)
    assert without["status"] == vd.OK  # metadata says satisfied

    with_import = vd.check_requirement(spec, self_name="mriforge", import_check=True)
    assert with_import["status"] == vd.IMPORT_FAIL
    assert "huggingface-hub" in with_import["detail"]


# --------------------------------------------------------------------------- #
# End-to-end run + exit-code semantics.
# --------------------------------------------------------------------------- #
def test_run_core_only_returns_records():
    results = vd.run(extras=[], want_all=False, import_check=False)
    assert set(results) == {"core"}
    assert all("status" in r for r in results["core"])


def test_has_failure_flags_missing():
    good = {"core": [{"status": vd.OK, "name": "a"}]}
    bad = {"core": [{"status": vd.MISSING, "name": "b"}]}
    assert vd._has_failure(good) is False
    assert vd._has_failure(bad) is True


def test_main_unknown_extra_exit_2():
    rc = vd.main(["--extras", "not-a-real-group"])
    assert rc == 2


def test_main_core_metadata_exit_0():
    # The project's own core deps are installed in the test env; metadata-only
    # (no import probe) must pass regardless of the torchmetrics import state.
    rc = vd.main([])
    assert rc == 0
