"""Unit tests for ``scripts/ci/check_f821_ratchet.py``.

Non-negotiable #15: a gate is only a gate for the violation shape you have
watched it fail on. So every rule this ratchet claims is planted here, one plant
per shape, and the plants run END TO END -- a real ``ruff`` over a real (tiny)
tree, then through :func:`compare` -- rather than hand-feeding ``compare`` a
Counter. Hand-fed counters would prove the comparison arithmetic and nothing
about the half that actually decides what a violation *is*: the parse of ruff's
output. That half is where this gate's specific hazard lives, because ruff
reports an unparseable file as ``invalid-syntax``, not ``F821``, and one broken
file in this repo emits 270 of them against 55 real findings.

The shapes, and why each earns a plant rather than being a variation of another:

* a name undefined in a file with no recorded findings   (the ordinary case)
* a SECOND occurrence beside a recorded one              (identity-keying blind spot)
* a recorded name that is now fixed                      (ratchet must tighten)
* a recorded name whose file is now clean                (stale row pre-exempts it)
* a file that stopped parsing                            (silently exempt from every AST gate)
* a recorded-unparseable file that now parses            (stale in the other list)
"""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "ci" / "check_f821_ratchet.py"


@pytest.fixture(scope="module")
def gate():
    """The checker module.

    Loaded from its path because ``scripts/`` is not an importable package. A
    failure here RAISES rather than skips: a skip would let a deleted or renamed
    script read as a green test file.
    """
    spec = importlib.util.spec_from_file_location("_f821_ratchet_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None, f"cannot load {_SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _plant(root: Path, files: dict[str, str]) -> Path:
    """Materialize a tiny tree and return its root."""
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _measure(gate, root: Path) -> tuple[Counter, set[str]]:
    """What the gate sees on ``root`` — the real ruff, not a stub."""
    return gate.split_diagnostics(gate._run_ruff(root), root)


# A file whose only finding is one undefined name, and a clean sibling.
_ONE_UNDEFINED = "def f():\n    return missing_name\n"
_CLEAN = "def g():\n    return 1\n"


@pytest.fixture
def planted(tmp_path, gate):
    """A two-file tree measured and recorded, ready to be perturbed.

    Returns ``(root, baseline, unparseable_baseline)`` where the baseline is
    exactly what the tree currently reports — so the control case is green and
    every later assertion isolates the perturbation.
    """
    root = _plant(tmp_path, {"a.py": _ONE_UNDEFINED, "b.py": _CLEAN})
    found, unparseable = _measure(gate, root)
    assert found == Counter({("a.py", "missing_name"): 1}), (
        f"the fixture itself must report exactly one finding, got {found!r}"
    )
    return root, Counter(found), set(unparseable)


class TestControl:
    """The tree the baseline describes is green — otherwise nothing below means anything."""

    def test_unperturbed_tree_is_green(self, planted, gate):
        root, baseline, known = planted
        found, unparseable = _measure(gate, root)
        problems, _ = gate.compare(found, baseline, unparseable, known)
        assert problems == []


class TestPlantedViolations:
    """One plant per shape. Each must turn the gate red."""

    def test_new_undefined_name_in_a_clean_file(self, planted, gate):
        root, baseline, known = planted
        (root / "b.py").write_text("def g():\n    return brand_new\n", encoding="utf-8")
        found, unparseable = _measure(gate, root)
        problems, _ = gate.compare(found, baseline, unparseable, known)
        assert any("b.py" in p and "brand_new" in p for p in problems), problems

    def test_second_occurrence_beside_a_recorded_one(self, planted, gate):
        """Counting, not identity.

        Keying on ``(path, name)`` alone would let this land green forever: the
        pair is already recorded, so a second ``missing_name`` in the same file
        is invisible. This is the same blind spot non-negotiable #20 records for
        the LOC ratchet, where 99 already-baselined files grew by 11,193 LOC
        without one gate going red.
        """
        (root := planted[0], baseline := planted[1], known := planted[2])
        (root / "a.py").write_text(
            "def f():\n    return missing_name\n\n\ndef f2():\n    return missing_name\n",
            encoding="utf-8",
        )
        found, unparseable = _measure(gate, root)
        assert found[("a.py", "missing_name")] == 2, found
        problems, _ = gate.compare(found, baseline, unparseable, known)
        assert any("x2" in p and "missing_name" in p for p in problems), problems

    def test_fixed_name_must_lower_the_baseline(self, planted, gate):
        """A ratchet that only rejects growth never tightens."""
        root, baseline, known = planted
        (root / "a.py").write_text(
            "def f():\n    missing_name = 1\n    return missing_name\n", encoding="utf-8"
        )
        found, unparseable = _measure(gate, root)
        assert found == Counter(), found
        problems, _ = gate.compare(found, baseline, unparseable, known)
        assert any("missing_name" in p and "baseline" in p for p in problems), problems

    def test_recorded_file_deleted_leaves_a_stale_row(self, planted, gate):
        root, baseline, known = planted
        (root / "a.py").unlink()
        found, unparseable = _measure(gate, root)
        problems, _ = gate.compare(found, baseline, unparseable, known)
        assert any("a.py" in p for p in problems), problems

    def test_newly_unparseable_file(self, planted, gate):
        """Not merely a broken file.

        Every AST gate in this repo does ``except SyntaxError: continue``, so a
        file that stops parsing goes silently exempt from all of them at once.
        """
        root, baseline, known = planted
        (root / "b.py").write_text("def g(:\n", encoding="utf-8")
        found, unparseable = _measure(gate, root)
        assert "b.py" in unparseable, unparseable
        problems, _ = gate.compare(found, baseline, unparseable, known)
        assert any("b.py" in p and "parse" in p for p in problems), problems

    def test_recorded_unparseable_file_that_now_parses(self, planted, gate):
        root, baseline, _ = planted
        found, unparseable = _measure(gate, root)
        problems, _ = gate.compare(found, baseline, unparseable, {"b.py"})
        assert any("b.py" in p and "now parses" in p for p in problems), problems


class TestDiagnosticSplit:
    """``invalid-syntax`` is not F821, and folding them together destroys the number."""

    def test_broken_file_contributes_no_f821_rows(self, tmp_path, gate):
        root = _plant(tmp_path, {"broken.py": "def f(:\n    return also_undefined\n"})
        found, unparseable = _measure(gate, root)
        assert unparseable == {"broken.py"}
        assert found == Counter(), (
            "a file ruff cannot parse yields ZERO F821 findings — counting its "
            f"syntax diagnostics as undefined names would inflate the ratchet: {found!r}"
        )

    def test_one_broken_file_can_out_number_the_real_signal(self, tmp_path, gate):
        """The measurement that motivated the split, in miniature.

        On ``dev`` a single unparseable file produced 270 ``invalid-syntax``
        diagnostics against 55 real F821 — so a naive ``len(diagnostics)``
        ratchet would be tracking a broken file, not undefined names.
        """
        root = _plant(tmp_path, {"broken.py": "def f(:\n" + "x = 1\n" * 40})
        raw = gate._run_ruff(root)
        found, unparseable = gate.split_diagnostics(raw, root)
        assert len(raw) > 1 and not found and unparseable


class TestBaselineContract:
    def test_missing_baseline_raises_rather_than_defaulting_to_empty(self, tmp_path, gate):
        """Absent is a state to report, never one to infer (non-negotiable #18).

        An empty default would make the gate pass while checking nothing.
        """
        with pytest.raises(FileNotFoundError):
            gate.read_baseline(tmp_path / "nope.txt")

    def test_comments_only_baseline_is_a_legitimate_empty(self, tmp_path, gate):
        """Distinct from the case above: zero recorded findings is the END STATE."""
        path = tmp_path / "f821.txt"
        path.write_text("# nothing left\n\n", encoding="utf-8")
        assert gate.read_baseline(path) == Counter()

    def test_render_and_read_round_trip(self, tmp_path, gate):
        counts = Counter({("a/b.py", "np"): 3, ("c.py", "logger"): 1})
        path = tmp_path / "f821.txt"
        path.write_text(gate.render(counts), encoding="utf-8")
        assert gate.read_baseline(path) == counts

    def test_missing_ruff_raises_rather_than_passing(self, gate, monkeypatch):
        monkeypatch.setattr(gate.shutil, "which", lambda _name: None)
        with pytest.raises(RuntimeError, match="ruff"):
            gate._run_ruff(_REPO_ROOT)


class TestCommittedBaselineIsCurrent:
    """The gate is green on the tree it ships with, and the baseline is honest."""

    def test_repo_tree_matches_its_baseline(self, gate):
        root = _REPO_ROOT
        found, unparseable = _measure(gate, root)
        here = _SCRIPT.parent / "baselines"
        problems, _ = gate.compare(
            found,
            gate.read_baseline(here / gate.BASELINE_FILENAME),
            unparseable,
            gate.read_lines(here / gate.UNPARSEABLE_FILENAME),
        )
        assert problems == [], problems

    def test_baseline_is_not_silently_empty(self, gate):
        """A ratchet recording nothing would be green forever."""
        recorded = gate.read_baseline(_SCRIPT.parent / "baselines" / gate.BASELINE_FILENAME)
        assert sum(recorded.values()) > 0


class TestCoverageIsWholeTree:
    """The reach of the gate, pinned — because narrowing it is invisible.

    ``ruff`` decides which files it visits from ``pyproject.toml`` (``exclude``,
    ``extend-exclude``) and from ``.gitignore``. None of that is stated at the
    gate's call site, so adding one ``exclude`` entry would quietly drop a
    subtree out of the ratchet's reach while leaving it green -- the same shape
    as the hand-written vocabulary that made the DataLoader SSOT gate blind to
    ``tio.SubjectsLoader`` (#1362), and the reason non-negotiable #15 asks for a
    plant per shape rather than a passing run.
    """

    def test_ruff_visits_every_tracked_python_file(self):
        import subprocess

        # Ask the GATE what it scans, rather than restating ``["."]`` here.
        # Spelling it out made this test a second owner of the gate's reach
        # (non-negotiable 17): it measured ``.`` while the gate had been widened
        # to descend into dot-directories, so the assertion could not see the fix
        # to the very gate it audits.
        from scripts.ci.check_f821_ratchet import ruff_scan_roots

        visited = subprocess.run(
            [
                "ruff",
                "check",
                "--no-cache",
                "--select",
                "F821",
                "--show-files",
                *ruff_scan_roots(_REPO_ROOT),
            ],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.split()
        visited = {Path(line).resolve() for line in visited}
        tracked = subprocess.run(
            ["git", "ls-files", "*.py"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        tracked = {(_REPO_ROOT / line).resolve() for line in tracked}
        assert tracked, "git ls-files returned nothing — the measurement itself failed"
        missed = sorted(str(p.relative_to(_REPO_ROOT)) for p in tracked - visited)
        assert not missed, (
            f"{len(missed)} tracked file(s) are outside the ratchet's reach, so an "
            f"undefined name there can never turn it red: {missed[:10]}"
        )


class TestAbsentFilesOk:
    """``--absent-files-ok`` must waive absence and nothing else.

    The public export drops ``scripts/sim2rank/`` and ``scripts/preprocessing/``
    wholesale, so five baseline rows name files that tree does not contain and
    the ratchet went red on arrival. The shape worth planting is not "does the
    flag let the export pass" -- it is "does the flag ALSO let a real regression
    pass", because a waiver keyed on the wrong thing turns a ratchet into an
    empty ceremony. Presence is decided by the real filesystem here, never by a
    mock that would agree with the docstring.
    """

    _PRESENT = "src/pkg/present.py"
    _ABSENT = "scripts/sim2rank/dropped.py"

    def _tree(self, tmp_path: Path) -> Path:
        f = tmp_path / self._PRESENT
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x = 1\n")
        return tmp_path

    def _compare(self, gate, root, found, baseline, absent_ok):
        return gate.compare(
            Counter(found),
            Counter(baseline),
            set(),
            set(),
            root=root,
            absent_files_ok=absent_ok,
        )

    def test_an_absent_file_is_waived_and_reported_not_dropped(self, tmp_path, gate):
        root = self._tree(tmp_path)
        problems, absent = self._compare(gate, root, {}, {(self._ABSENT, "np"): 1}, True)
        assert problems == []
        assert len(absent) == 1 and self._ABSENT in absent[0]

    def test_a_present_file_whose_count_dropped_still_fails(self, tmp_path, gate):
        """The discrimination test: the flag waives ABSENCE, never a lowered count."""
        root = self._tree(tmp_path)
        problems, absent = self._compare(gate, root, {}, {(self._PRESENT, "np"): 1}, True)
        assert absent == []
        assert len(problems) == 1 and "lower the baseline" in problems[0]

    def test_a_new_undefined_name_still_fails_under_the_flag(self, tmp_path, gate):
        root = self._tree(tmp_path)
        problems, absent = self._compare(
            gate, root, {(self._PRESENT, "np"): 2}, {(self._PRESENT, "np"): 1}, True
        )
        assert absent == []
        assert len(problems) == 1 and "may only go down" in problems[0]

    def test_absence_still_fails_when_the_flag_is_off(self, tmp_path, gate):
        """Default behaviour unchanged -- the tree that recorded the baseline
        keeps the strict check, so a genuinely deleted file still prompts a
        baseline lowering."""
        root = self._tree(tmp_path)
        problems, absent = self._compare(gate, root, {}, {(self._ABSENT, "np"): 1}, False)
        assert absent == []
        assert len(problems) == 1 and "lower the baseline" in problems[0]

    def test_a_regression_in_an_absent_file_cannot_be_hidden(self, tmp_path, gate):
        """``found`` can only name files ruff scanned, so an absent file should
        never be the count-went-UP side. Pinned anyway: if a refactor ever lets a
        stale ``found`` entry through, the waiver must not swallow it."""
        root = self._tree(tmp_path)
        problems, absent = self._compare(
            gate, root, {(self._ABSENT, "np"): 3}, {(self._ABSENT, "np"): 1}, True
        )
        assert absent == []
        assert len(problems) == 1 and "may only go down" in problems[0]
