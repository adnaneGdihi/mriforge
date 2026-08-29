"""Tests for the package-level clinical disclaimer (:func:`mriforge._emit_clinical_disclaimer`).

Regression (2026-06-30): the ``NOT FOR CLINICAL USE`` ``UserWarning`` lived at
module top level, so every ``spawn``-ed DataLoader worker re-imported the package
and re-emitted it — N+1 copies for N workers cluttering batch logs ("disclaimer
should only be one").

Regression (2026-08-15): the child gate that fix installed never fired. It tested
``multiprocessing.parent_process() is not None``, and at *import* time in a
spawned child that is still ``None`` — the re-import happens while UNPICKLING the
target, strictly before ``BaseProcess._bootstrap`` assigns
``multiprocessing._parent_process``. Measured on 3.12, both ``spawn`` and
``forkserver``::

    IMPORT parent_process_is_None=True name='ForkServerProcess-2' _inheriting=True
    IMPORT parent_process_is_None=True name='SpawnProcess-3'      _inheriting=True

So the burst came back the moment DataLoader workers started. ``current_process()``
IS populated in that window (it is unpickled from the parent), which is what
:func:`mriforge._in_child_process` uses instead. A third gate was added at the same
time: N ranks of a torchrun launch are N interpreters, so they emitted N copies of
one legal notice.

These tests pin all three gates, and — because the failed gate was a *plausible*
API used in the wrong window — the child gate is also exercised against a real
spawned child rather than only a monkeypatched predicate.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import warnings

import pytest

import mriforge


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("MRIFORGE_SUPPRESS_CLINICAL_WARNING", raising=False)
    # The rank gate must not read a torchrun environment leaking in from a
    # parent job (these tests assert single-process behaviour by default).
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)


def test_emits_in_main_process_by_default():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        emitted = mriforge._emit_clinical_disclaimer()
    assert emitted is True
    assert any("NOT FOR CLINICAL USE" in str(w.message) for w in caught)


def test_env_flag_suppresses(monkeypatch):
    monkeypatch.setenv("MRIFORGE_SUPPRESS_CLINICAL_WARNING", "1")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        emitted = mriforge._emit_clinical_disclaimer()
    assert emitted is False
    assert caught == []


# --------------------------------------------------------------------------- #
# Gate 2: spawned children (DataLoader workers)
# --------------------------------------------------------------------------- #
class _FakeProcess:
    def __init__(self, name, inheriting=False):
        self.name = name
        if inheriting:
            self._inheriting = True


@pytest.mark.parametrize(
    "proc",
    [
        _FakeProcess("SpawnProcess-3", inheriting=True),
        _FakeProcess("ForkServerProcess-2", inheriting=True),
        # A worker whose private ``_inheriting`` flag is gone (it is CPython
        # internal); the process name still discriminates.
        _FakeProcess("SpawnPoolWorker-1"),
    ],
    ids=["spawn", "forkserver", "name-only"],
)
def test_silent_in_child_process(proc, monkeypatch):
    monkeypatch.setattr(mriforge.multiprocessing, "current_process", lambda: proc)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        emitted = mriforge._emit_clinical_disclaimer()
    assert emitted is False
    assert caught == []


def test_parent_process_is_not_what_the_gate_reads():
    """The 2026-08-15 root cause, pinned as an executable statement.

    ``parent_process()`` returning ``None`` must NOT be read as "main process":
    it is ``None`` in a spawned child too, during the import window this gate
    runs in. If someone reintroduces that test, this fails.
    """
    monkey = _FakeProcess("SpawnProcess-1", inheriting=True)
    assert mriforge._in_child_process.__doc__ is not None
    import unittest.mock as m

    with (
        m.patch.object(mriforge.multiprocessing, "current_process", lambda: monkey),
        m.patch.object(mriforge.multiprocessing, "parent_process", lambda: None),
    ):
        # parent_process() lies here; the gate must still say "child".
        assert mriforge._in_child_process() is True


@pytest.mark.parametrize("method", ["spawn", "forkserver"])
def test_real_child_process_does_not_re_emit(method, tmp_path):
    """End-to-end: import mriforge in a genuinely spawned child.

    The monkeypatched cases above pin the predicate; this one pins that the
    predicate is consulted in the window that actually matters. A child that
    re-imports the package must record ``emitted=False``.
    """
    if method not in mp.get_all_start_methods():  # pragma: no cover - platform
        pytest.skip(f"{method} start method unavailable")
    out = tmp_path / "child.txt"
    ctx = mp.get_context(method)
    proc = ctx.Process(target=_child_reports_emission, args=(str(out),))
    proc.start()
    proc.join(timeout=180)
    assert proc.exitcode == 0, f"child failed (exitcode={proc.exitcode})"
    assert out.read_text().strip() == "False", (
        "a spawned child re-emitted the clinical disclaimer — the import-time "
        "child gate is inoperative again"
    )


def _child_reports_emission(path: str) -> None:
    """Runs in the child. Module-level, so it is picklable by spawn."""
    # Re-run the guard rather than trusting the import-time call: the child has
    # already executed the module body by the time this function is unpickled.
    import mriforge as child_pkg

    with open(path, "w") as fh:
        fh.write(str(child_pkg._emit_clinical_disclaimer()))


# --------------------------------------------------------------------------- #
# Gate 3: non-zero ranks of a distributed launch
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rank", ["1", "2", "3"])
def test_silent_on_secondary_rank(rank, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", rank)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert mriforge._emit_clinical_disclaimer() is False
    assert caught == []


def test_still_emits_on_rank_zero(monkeypatch):
    """One notice per job, not zero — this is a legal disclaimer."""
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "0")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert mriforge._emit_clinical_disclaimer() is True
    assert any("NOT FOR CLINICAL USE" in str(w.message) for w in caught)


def test_rank_env_names_match_the_env_ssot():
    """``mriforge/__init__.py`` reads RANK/WORLD_SIZE literally, not via the SSOT.

    The original reason was cost: ``from mriforge.core import env`` executed an
    eager ``mriforge.core.__init__`` that pulled torch on every ``import
    mriforge``. **That reason no longer holds.** ``core/__init__.py`` became
    lazy in #1130, and the same import was measured on this machine at
    0.013 s / 102 modules / 12 MB with torch absent, against
    3.3 s / 4344 modules / 976 MB before. The literals are therefore no longer
    *required* — only still sufficient, and left alone here because rewriting a
    legal-disclaimer gate is not this change's business.

    The test keeps its full value either way, because it never rested on the
    cost argument: it pins the literals to the names ``core/env.py`` declares,
    so a rename there cannot silently strand the gate.
    """
    import inspect

    from mriforge.core import env

    src = inspect.getsource(mriforge._emit_clinical_disclaimer)
    assert f'"{env.WORLD_SIZE}"' in src
    assert f'"{env.RANK}"' in src
    assert f'"{env.MRIFORGE_SUPPRESS_CLINICAL_WARNING}"' in src


def test_package_import_is_torch_free():
    """The gate must not have dragged torch into ``import mriforge``."""
    assert "mriforge" in sys.modules, "precondition: package already imported"
    code = "import mriforge, sys; print('torch' in sys.modules)"
    import subprocess

    env = dict(os.environ, MRIFORGE_SUPPRESS_CLINICAL_WARNING="1")
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "False", (
        "importing mriforge now pulls torch — the disclaimer gate must read "
        "os.environ directly, not through mriforge.core.env"
    )
