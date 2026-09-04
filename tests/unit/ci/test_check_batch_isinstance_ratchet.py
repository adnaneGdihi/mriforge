"""Planted violations for the ``isinstance(batch, dict)`` ratchet.

Non-negotiable 15: a gate is only a gate for the violation shape it has been
watched failing on — one per rule, and one per SHAPE that rule can take. Every
detector in this repo that turned out blind had gone red many times, on the easy
shape. So each test here plants a guard the gate must catch, and the
``sanity`` tests below assert the gate is not simply matching everything.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_GATE = Path(__file__).resolve().parents[3] / "scripts" / "ci" / "check_batch_isinstance_ratchet.py"


def _load():
    spec = importlib.util.spec_from_file_location("batch_ratchet", _GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gate = _load()


# --- shapes the gate MUST catch -------------------------------------------


def test_catches_block_form():
    """``if isinstance(batch, dict):`` — the shape everyone thinks of."""
    assert gate.count_in_source("if isinstance(batch, dict):\n    x = batch['a']\n") == 1


def test_catches_ternary_form():
    """The silent one: yields None for a TrainingBatch, and nothing raises."""
    src = "k = batch.get('kspace') if isinstance(batch, dict) else None\n"
    assert gate.count_in_source(src) == 1


def test_catches_function_local_guard():
    """THE shape that made check_layering.sh blind for its whole life (#1183).

    A line regex anchored at ``^`` sees a module-level guard and misses this one.
    """
    src = "def step(self, batch):\n    if isinstance(batch, dict):\n        return batch['x']\n"
    assert gate.count_in_source(src) == 1


def test_catches_guard_inside_an_fstring():
    """Diagnostic messages carry these too, and they take the wrong branch
    exactly when a human is reading the message to find out what went wrong."""
    src = 'msg = f"keys={sorted(batch) if isinstance(batch, dict) else type(batch)}"\n'
    assert gate.count_in_source(src) == 1


def test_catches_guard_inside_a_comprehension():
    src = "vals = [batch[k] for k in keys if isinstance(batch, dict)]\n"
    assert gate.count_in_source(src) == 1


def test_catches_tuple_classinfo():
    """``(dict, Mapping)`` reads as permissive but still answers False for a
    dataclass that implements the protocol without registering as a Mapping."""
    src = "if isinstance(batch, (dict, Mapping)):\n    pass\n"
    assert gate.count_in_source(src) == 1


def test_catches_the_other_batch_names():
    """``val_batch`` is the one that matters most: BatchAdapter.from_dict runs on
    the validation path, so that name is the one actually holding a TrainingBatch."""
    for name in ("val_batch", "train_batch", "batch_data"):
        assert gate.count_in_source(f"if isinstance({name}, dict):\n    pass\n") == 1, name


def test_counts_every_occurrence_not_just_the_file():
    """Non-negotiable 20: keying on identity alone lets a second guard land
    beside a recorded one forever."""
    src = (
        "a = batch.get('x') if isinstance(batch, dict) else None\n"
        "b = batch.get('y') if isinstance(batch, dict) else None\n"
    )
    assert gate.count_in_source(src) == 2


# --- shapes the gate must NOT flag (a gate that matches everything is noise) ---


def test_ignores_isinstance_on_other_objects():
    assert gate.count_in_source("if isinstance(config, dict):\n    pass\n") == 0


def test_ignores_non_dict_checks_on_batch():
    """Checking a batch against a real type is legitimate."""
    assert gate.count_in_source("if isinstance(batch, TrainingBatch):\n    pass\n") == 0


def test_ignores_the_canonical_accessor():
    assert gate.count_in_source("x = read_batch_field(batch, 'input')\n") == 0


# --- ratchet behaviour ------------------------------------------------------


def test_new_file_is_a_regression():
    regressions, stale = gate.compare({"a.py": 1}, {})
    assert regressions and "NEW" in regressions[0]
    assert not stale


def test_grown_count_is_a_regression():
    """The failure the LOC baseline could not see: growth inside a file already
    recorded (99 of 537 files grew 11,193 LOC without one gate going red)."""
    regressions, _ = gate.compare({"a.py": 3}, {"a.py": 1})
    assert regressions and "GREW" in regressions[0]


def test_fixed_but_still_recorded_is_a_hard_failure():
    """A stale entry pre-exempts the guard if it comes back."""
    _, stale = gate.compare({"a.py": 1}, {"a.py": 4})
    assert stale and "STALE" in stale[0]


def test_shrinking_to_zero_is_still_stale_not_silence():
    _, stale = gate.compare({}, {"a.py": 2})
    assert stale


def test_unchanged_is_clean():
    regressions, stale = gate.compare({"a.py": 2}, {"a.py": 2})
    assert not regressions and not stale


# --- the live baseline ------------------------------------------------------


def test_repo_currently_passes_its_own_baseline():
    """The recorded state must match the tree, or the gate is already lying."""
    repo = Path(__file__).resolve().parents[3]
    current, _ = gate.scan(repo / "src" / "spectramr")
    baseline = gate._read_baseline(repo / "scripts" / "ci" / "baselines" / "batch_isinstance.txt")
    regressions, stale = gate.compare(current, baseline)
    assert not regressions, f"unrecorded guards: {regressions}"
    assert not stale, f"stale baseline entries: {stale}"


def test_unparseable_file_is_reported_not_skipped():
    """Non-negotiable 18: a file that does not parse has UNKNOWN status, and
    every AST gate here does ``except SyntaxError: continue`` — silently exempt."""
    with pytest.raises(SyntaxError):
        gate.count_in_source("def broken(:\n")
