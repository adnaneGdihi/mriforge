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
import re
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
        ("spectramr[mri,viz,test]", "spectramr", False),
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
# Every entry here was verified by an actual `uv pip install --target` build with
# isolation ON (2026-08-29) — an entry added on reasoning alone silently narrows
# the guarantee `.[all]` makes, so measurement is the admission price:
#   mamba     - compiles the CUDA selective-scan kernel; needs nvcc.
#   attention - flash-attn omits torch from build-system.requires, so it cannot
#               build under isolation. xformers alone would resolve; the group
#               fails as a unit.
#   radiomics - pyradiomics has no cp312 wheel and its C extension fails to
#               compile.
# `bnb` and `deepspeed` were previously assumed to need a CUDA toolchain and were
# absent from `all`; both build clean under isolation, so they are now IN and
# this guard is green rather than standing red on them.
# Groups deliberately outside `all`. NOT a synonym for "unbuildable": three of
# these cannot resolve one-shot, while `registration` resolves perfectly well and
# is held out because it would drag numpy/scipy backwards under everything else.
_EXCLUDED_FROM_ALL = {"mamba", "attention", "radiomics", "registration"}
# Meta-groups are themselves self-references, not concrete feature extras.
_META_GROUPS = {"all", "dev"}


def _referenced_extras(req_strings: list[str]) -> set[str]:
    """Extract the extras named in a ``spectramr[a,b,c]`` self-reference list."""
    referenced: set[str] = set()
    for spec in req_strings:
        name, _, _ = vd.parse_requirement(spec)
        if vd._canon(name) != "spectramr":
            continue
        inside = spec[spec.index("[") + 1 : spec.index("]")]
        referenced |= {e.strip() for e in inside.split(",") if e.strip()}
    return referenced


def _missing_from_all(groups: dict[str, list[str]]) -> set[str]:
    """Concrete extras that `all` fails to reference. Sole owner of the rule.

    Both the live check and the planted-violation tests below go through this
    function, so a plant that scores green cannot mean the live check is blind.
    """
    concrete = set(groups) - {"core"} - _META_GROUPS - _EXCLUDED_FROM_ALL
    return concrete - _referenced_extras(groups.get("all", []))


def test_all_extra_covers_every_oneshot_installable_extra():
    """`all` must include every one-shot-installable optional extra.

    Adding a new extra to pyproject without wiring it into `all` fails here — the
    guard that keeps a cluster `.[all]` install genuinely complete.
    """
    groups = vd.load_dependency_groups()
    assert "all" in groups, "the `all` convenience group must exist"

    missing = _missing_from_all(groups)
    assert not missing, f"`all` is missing one-shot-installable extras: {sorted(missing)}"


@pytest.mark.parametrize("extra", sorted(_EXCLUDED_FROM_ALL))
def test_all_extra_excludes_the_isolated_groups(extra):
    """Every group held out of `all` stays out of it.

    Pinned per-group rather than for `mamba` alone: each is held out for its own
    physical reason (nvcc kernel / undeclared build dep / missing C-extension
    wheel / caps the core stack), and a single-group pin would have scored green
    while the others drifted in. The set is the sole owner of the membership
    question — do not restate the reasons here, or this docstring becomes a
    second, unsynced enumeration of them.
    """
    groups = vd.load_dependency_groups()
    assert extra in groups, f"{extra} must still be declared as its own extra"
    assert extra not in _referenced_extras(groups["all"])


# --- planted violations: the guard must be watched going red (non-negotiable 15).
def test_completeness_guard_catches_an_extra_absent_everywhere():
    """Shape 1: a new feature group nobody wired into `all` at all."""
    planted = {"core": [], "all": ["spectramr[mri]"], "mri": [], "newfeature": []}
    assert _missing_from_all(planted) == {"newfeature"}


def test_completeness_guard_catches_an_extra_reachable_only_through_dev():
    """Shape 2: the mistake this file actually caught — a group demoted into
    `dev` so `.[dev]` installs it but `.[all]` does not. `dev` is not a substitute
    for `all`, and reading only `dev` would have scored this green."""
    planted = {
        "core": [],
        "all": ["spectramr[mri]"],
        "dev": ["spectramr[all]", "spectramr[qa]"],
        "mri": [],
        "qa": [],
    }
    assert _missing_from_all(planted) == {"qa"}


def test_completeness_guard_is_not_vacuous_on_a_compliant_table():
    """Shape 3: the guard must stay silent when the table is correct, otherwise
    the two tests above would pass for the wrong reason."""
    planted = {"core": [], "all": ["spectramr[mri,qa]"], "mri": [], "qa": []}
    assert _missing_from_all(planted) == set()


def test_completeness_guard_honours_the_excluded_set():
    """Shape 4: an entry in `_EXCLUDED_FROM_ALL` is not reported as missing —
    proving the exemption is what keeps mamba/attention/radiomics green, rather
    than the guard simply never firing."""
    planted = {"core": [], "all": ["spectramr[mri]"], "mri": [], "mamba": []}
    assert _missing_from_all(planted) == set()


_EXCLUSION_MARKER = "# EXCLUDED FROM `all`:"


def _groups_marked_excluded_in_prose(text: str) -> set[str]:
    """Group names whose preceding comment block carries the exclusion marker.

    Sole owner of the prose side of the rule. The comment above a group is the
    only human-readable statement of WHY it sits outside `all`, and nothing
    compared it against the table — so it was free to keep saying "excluded"
    about a group that had been moved in. Exactly that happened to `bnb` and
    `deepspeed`: both were referenced by `all` and both still carried an
    EXCLUDED comment, each giving a *reason* that measurement had disproved.
    A wrong reason is worse than no reason, because the next reader inherits it.
    """
    marked: set[str] = set()
    pending = False
    for line in text.splitlines():
        if line.startswith(_EXCLUSION_MARKER):
            pending = True
            continue
        m = re.match(r"^([A-Za-z][A-Za-z0-9_.-]*)\s*=", line)
        if m:
            if pending:
                marked.add(m.group(1))
            pending = False
    return marked


def test_prose_exclusions_match_the_table():
    """The comments and the `all` list must name the SAME excluded groups."""
    text = (REPO / "pyproject.toml").read_text()
    assert _groups_marked_excluded_in_prose(text) == _EXCLUDED_FROM_ALL


# --- planted violations for the prose guard (non-negotiable 15).
def test_prose_guard_catches_a_stale_exclusion_comment():
    """Shape 1: the bnb/deepspeed defect — a group carrying an EXCLUDED comment
    that the table has since contradicted by putting it in `all`."""
    planted = '# EXCLUDED FROM `all`: needs nvcc\nbnb = ["bitsandbytes>=0.45"]\n'
    assert _groups_marked_excluded_in_prose(planted) == {"bnb"}


def test_prose_guard_catches_an_undocumented_exclusion():
    """Shape 2: the inverse — a group held out of `all` with no comment saying
    why. The prose set comes back empty and so cannot equal the table's."""
    planted = 'radiomics = ["pyradiomics>=3.1"]\n'
    assert _groups_marked_excluded_in_prose(planted) == set()


def test_prose_guard_does_not_attach_a_marker_across_a_group():
    """The marker binds to the NEXT group only. Without this, one comment would
    silently mark every group below it and the equality check would pass for
    the wrong reason."""
    planted = (
        "# EXCLUDED FROM `all`: reason\n"
        'mamba = ["mamba-ssm>=2.2"]\n'
        "# an ordinary comment\n"
        'viz = ["matplotlib>=3.8"]\n'
    )
    assert _groups_marked_excluded_in_prose(planted) == {"mamba"}


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
    rec = vd.check_requirement("pytest>=1.0", self_name="spectramr", import_check=False)
    assert rec["status"] == vd.OK
    assert rec["installed"] is not None


def test_check_requirement_missing():
    rec = vd.check_requirement(
        "this-package-does-not-exist-xyz>=1.0",
        self_name="spectramr",
        import_check=False,
    )
    assert rec["status"] == vd.MISSING


def test_check_requirement_version_mismatch():
    # pytest is installed but cannot satisfy an absurd future lower bound.
    rec = vd.check_requirement(
        "pytest>=999999", self_name="spectramr", import_check=False
    )
    assert rec["status"] == vd.MISMATCH
    assert rec["installed"] is not None


def test_check_requirement_self_reference_skipped():
    rec = vd.check_requirement(
        "spectramr[mri,viz]", self_name="spectramr", import_check=False
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

    without = vd.check_requirement(spec, self_name="spectramr", import_check=False)
    assert without["status"] == vd.OK  # metadata says satisfied

    with_import = vd.check_requirement(spec, self_name="spectramr", import_check=True)
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
