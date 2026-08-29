"""Tests for ``scripts/ci/check_health_check_names.py`` (issue #1355).

Every test below plants a violation the gate must turn red on -- one per rule it
claims to enforce, and one per *shape* that rule can take (CLAUDE.md #15). The
rule this gate exists for has a specific shape that the assertion it replaces was
blind to: a ``FATAL_HEALTH_CHECKS`` entry that names a real **method** while the
name that method **emits** is spelled differently. ``test_a_fatal_name_that_only_
exists_as_a_method_is_reported`` plants exactly that and also asserts the old
method-name predicate stays satisfied, so the two forms are shown to disagree
rather than assumed to.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "check_health_check_names.py"

#: A miniature ``config_health_checker`` carrying all three ways the real file
#: passes ``check_name``: as a keyword, positionally, and via a local constant.
BASE = """
from dataclasses import dataclass

FATAL_HEALTH_CHECKS = frozenset({"alpha"})


@dataclass
class HealthCheckResult:
    passed: bool
    check_name: str
    message: str
    severity: str


class ConfigHealthChecker:
    def check_alpha(self):
        return HealthCheckResult(passed=True, check_name="alpha", message="", severity="info")

    def check_beta(self):
        return HealthCheckResult(True, "beta", "", "info")

    def check_gamma(self):
        check_name = "gamma"
        return HealthCheckResult(True, check_name, "", "info")
"""


def _load():
    spec = importlib.util.spec_from_file_location("_check_health_check_names", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate():
    return _load()


def _check(gate, tmp_path: Path, source: str, *, ledger_from: str | None = None) -> int:
    """Record a ledger from ``ledger_from`` (default ``source``), then check ``source``."""
    ledger = tmp_path / "ledger.txt"
    base = tmp_path / "base.py"
    base.write_text(ledger_from if ledger_from is not None else source, encoding="utf-8")
    assert gate.main(["--source", str(base), "--ledger", str(ledger), "--refresh"]) == 0
    src = tmp_path / "src.py"
    src.write_text(source, encoding="utf-8")
    return gate.main(["--source", str(src), "--ledger", str(ledger)])


# --------------------------------------------------------------------------
# The real file, and the collector's own reading of it
# --------------------------------------------------------------------------


def test_the_real_source_matches_the_committed_ledger(gate) -> None:
    assert gate.main([]) == 0


def test_the_gate_reads_the_real_source_from_any_cwd(gate) -> None:
    """Run from ``/``: a repo-rooted gate that resolves paths off the cwd exits 0
    all-clean on an empty file set, which reads exactly like a pass."""
    done = subprocess.run(
        [sys.executable, str(SCRIPT)], cwd="/", capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr
    assert "check methods emit" in done.stdout


def test_all_three_emission_shapes_resolve(gate) -> None:
    mapping, unresolved = gate.collect(BASE)
    assert unresolved == []
    assert mapping == {"check_alpha": ["alpha"], "check_beta": ["beta"], "check_gamma": ["gamma"]}


def test_the_positional_index_comes_from_the_dataclass_not_a_constant(gate) -> None:
    """``check_name`` sits at index 1 today. Hardcoding that resolves every
    positional emission to the ``passed`` flag the day the fields are reordered --
    which reports the file as unreadable rather than as changed."""
    reordered = (
        BASE.replace(
            "    passed: bool\n    check_name: str", "    check_name: str\n    passed: bool"
        )
        .replace('HealthCheckResult(True, "beta"', 'HealthCheckResult("beta", True')
        .replace("HealthCheckResult(True, check_name", "HealthCheckResult(check_name, True")
    )
    mapping, unresolved = gate.collect(reordered)
    assert unresolved == []
    assert mapping["check_beta"] == ["beta"]
    assert mapping["check_gamma"] == ["gamma"]


# --------------------------------------------------------------------------
# Rule 1 -- every FATAL_HEALTH_CHECKS entry is a name something emits
# --------------------------------------------------------------------------


def test_a_fatal_name_that_only_exists_as_a_method_is_reported(gate, tmp_path, capsys) -> None:
    """The #1355 shape, and the discrimination against the assertion it replaces."""
    planted = (
        BASE.replace('frozenset({"alpha"})', 'frozenset({"alpha", "delta"})')
        + """
    def check_delta(self):
        return HealthCheckResult(True, "delta_thing", "", "info")
"""
    )
    mapping, _ = gate.collect(planted)
    assert "check_delta" in mapping, (
        "the old assertion -- hasattr(checker, f'check_{name}') -- is satisfied here"
    )
    assert _check(gate, tmp_path, planted) == 1
    assert "'delta'" in capsys.readouterr().err


def test_a_fatal_name_nothing_declares_at_all_is_reported(gate, tmp_path) -> None:
    assert _check(gate, tmp_path, BASE.replace('{"alpha"}', '{"alpha", "typoo"}')) == 1


# --------------------------------------------------------------------------
# Rule 2 -- the whole method -> names mapping is conserved
# --------------------------------------------------------------------------


def test_a_renamed_emission_is_reported(gate, tmp_path) -> None:
    assert _check(gate, tmp_path, BASE.replace('"beta"', '"beta_renamed"'), ledger_from=BASE) == 1


def test_a_new_check_method_is_reported(gate, tmp_path) -> None:
    added = (
        BASE
        + """
    def check_epsilon(self):
        return HealthCheckResult(True, "epsilon", "", "info")
"""
    )
    assert _check(gate, tmp_path, added, ledger_from=BASE) == 1


def test_a_removed_check_method_is_reported(gate, tmp_path) -> None:
    removed = BASE.split("    def check_gamma")[0]
    assert _check(gate, tmp_path, removed, ledger_from=BASE) == 1


def test_a_name_migrating_between_methods_is_reported(gate, tmp_path) -> None:
    """The case a bare set-of-names ledger cannot see: the name set is unchanged,
    but ``beta`` is now emitted by a different check, so a fatal entry aimed at it
    would fire from somewhere else."""
    swapped = BASE.replace(
        'HealthCheckResult(True, "beta", "", "info")',
        'HealthCheckResult(True, "gamma", "", "info")',
    ).replace('        check_name = "gamma"', '        check_name = "beta"')
    before, _ = gate.collect(BASE)
    after, _ = gate.collect(swapped)
    assert {n for v in before.values() for n in v} == {n for v in after.values() for n in v}
    assert _check(gate, tmp_path, swapped, ledger_from=BASE) == 1


# --------------------------------------------------------------------------
# Rule 3 -- a check that emits nothing is a loud state, never an inferred one
# --------------------------------------------------------------------------


def test_a_check_method_that_emits_nothing_is_reported(gate, tmp_path, capsys) -> None:
    """The helper-refactor shape: ``self._make_result(...)`` would empty this
    collector silently and leave a green gate over an unscanned file."""
    silent = BASE.replace(
        '        return HealthCheckResult(True, "beta", "", "info")',
        '        return self._make_result("beta")',
    )
    assert _check(gate, tmp_path, silent, ledger_from=BASE) == 1
    assert "emits no HealthCheckResult" in capsys.readouterr().err


def test_an_unresolvable_check_name_is_reported(gate, tmp_path, capsys) -> None:
    computed = BASE.replace('HealthCheckResult(True, "beta"', 'HealthCheckResult(True, f"be{1}ta"')
    assert _check(gate, tmp_path, computed) == 1
    assert "unresolvable check_name" in capsys.readouterr().err


def test_a_whitespace_bearing_name_is_reported_not_silently_split(gate, tmp_path, capsys) -> None:
    """The ledger is space-separated, so ``"be ta"`` would round-trip as two names.

    Left unguarded that wedges the gate red forever with a message naming two
    phantom checks. It must instead be reported at its site, as unresolvable.
    """
    spaced = BASE.replace('HealthCheckResult(True, "beta"', 'HealthCheckResult(True, "be ta"')
    assert _check(gate, tmp_path, spaced) == 1
    err = capsys.readouterr().err
    assert "unresolvable check_name" in err
    assert "'be ta'" in err, "the offending name must appear verbatim, not as two tokens"

    # Discrimination: the same source with a space-free name is clean, so the
    # plant is red for the whitespace and not for the rename.
    renamed = BASE.replace('HealthCheckResult(True, "beta"', 'HealthCheckResult(True, "beta2"')
    assert _check(gate, tmp_path, renamed, ledger_from=renamed) == 0


# --------------------------------------------------------------------------
# Reading failures are exit 2, distinct from a violation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mangled",
    [
        "class ConfigHealthChecker:\n    pass\n",
        BASE.replace("class ConfigHealthChecker:", "class SomethingElse:"),
        BASE.replace("FATAL_HEALTH_CHECKS = ", "OTHER = "),
        BASE.replace("    check_name: str\n", ""),
        "def broken(:\n",
    ],
)
def test_a_source_this_gate_cannot_read_exits_2(gate, tmp_path, mangled) -> None:
    src = tmp_path / "src.py"
    src.write_text(mangled, encoding="utf-8")
    assert (
        gate.main(
            [
                "--source",
                str(src),
                "--ledger",
                str(gate.LEDGER),
            ]
        )
        == 2
    )


def test_refresh_records_the_current_state_and_then_passes(gate, tmp_path) -> None:
    added = (
        BASE
        + """
    def check_epsilon(self):
        return HealthCheckResult(True, "epsilon", "", "info")
"""
    )
    assert _check(gate, tmp_path, added, ledger_from=BASE) == 1
    src, ledger = tmp_path / "src.py", tmp_path / "ledger.txt"
    assert gate.main(["--source", str(src), "--ledger", str(ledger), "--refresh"]) == 0
    assert gate.main(["--source", str(src), "--ledger", str(ledger)]) == 0
    assert gate._parse_ledger(ledger.read_text())["check_epsilon"] == ["epsilon"]
    assert ledger.read_text().startswith("# "), "the ledger says it is generated"
