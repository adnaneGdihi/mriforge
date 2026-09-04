"""The dataloader worker clamp — a ceiling, never a floor.

``num_workers`` had no topology term anywhere on its path, so ``num_workers: 8``
on a 4-rank node meant 32 decoder processes competing for 16 cores and the step
time went UP. These tests pin the properties that make the clamp safe to apply
at every loader: it can only ever reduce, a declared ``0`` survives, and an
unknown core count declines rather than guesses.

Style note: like ``test_topology.py``, these pass ``environ=`` / ``cpu_probe=``
explicitly and use **no** monkeypatch -- there is no global autouse fixture
restoring ``os.environ`` between tests, so ambient state would leak.
"""

from __future__ import annotations

from spectramr.core.topology import resolve_run_topology
from spectramr.core.worker_policy import WorkerDecision, clamp_worker_count


def _probe(cores):
    """A stand-in for ``cpu_resources`` returning a fixed usable-core count."""

    def probe(environ=None):
        return {"usable_cores": cores}

    return probe


def _resolve(env=None, *, cores=16, exists=lambda _p: False):
    return resolve_run_topology(env or {}, cpu_probe=_probe(cores), path_exists=exists)


def test_clamp_is_a_ceiling_and_never_raises_the_declared_count():
    """The property the OOM-fix arms depend on: it can only ever go DOWN."""
    t = _resolve({"WORLD_SIZE": "1"}, cores=64)
    d = clamp_worker_count(2, t)
    assert d.workers == 2 and not d.clamped
    assert d.reason == "fits-allocation"


def test_single_rank_run_that_fits_is_byte_identical():
    """A 1-GPU run whose workers fit its cores must be untouched by this change."""
    t = _resolve(cores=24)
    assert clamp_worker_count(8, t).workers == 8


def test_clamp_fixes_the_rank_multiplied_oversubscription():
    """The actual bug: num_workers 8 x 4 ranks = 32 decoders on 16 cores."""
    t = _resolve({"WORLD_SIZE": "4", "LOCAL_WORLD_SIZE": "4"}, cores=16)
    d = clamp_worker_count(8, t)
    assert d.workers == 4 and d.clamped
    assert d.declared == 8 and d.cpus_per_rank == 4
    assert d.reason == "clamped-to-cpu-share"


def test_declared_zero_passes_through_untouched():
    """0 means "load in the main process" -- a deliberate choice, and the
    ``validation.loader`` default. The clamp must never turn it into 1."""
    t = _resolve({"WORLD_SIZE": "4", "LOCAL_WORLD_SIZE": "4"}, cores=16)
    d = clamp_worker_count(0, t)
    assert d.workers == 0 and not d.clamped
    assert d.reason == "declared-serial"


def test_clamp_floor_is_one_so_persistent_workers_cannot_be_broken():
    """``num_workers=0`` + ``persistent_workers=True`` is a torch ERROR.

    Fixing the floor at 1 for any declared value >= 1 makes that combination
    unreachable by construction rather than by a downstream guard.
    """
    t = _resolve({"WORLD_SIZE": "32", "LOCAL_WORLD_SIZE": "32"}, cores=4)
    assert clamp_worker_count(8, t).workers == 1


def test_clamp_declines_rather_than_guesses_when_cores_are_unknown():
    """Unknown cores must not silently become some assumed default.

    Declining to reduce keeps today's behaviour (and logs loudly); substituting
    a made-up core count would be the silent fallback non-negotiable 3 forbids.
    """
    t = _resolve(cores=None)
    d = clamp_worker_count(8, t)
    assert d.workers == 8 and not d.clamped
    assert d.reason == "cpus-unknown"
    assert d.cpus_per_rank is None


def test_worker_decision_is_serialisable_for_provenance():
    t = _resolve(cores=16)
    assert set(WorkerDecision(4, 8, 4, True, "x").to_dict()) == {
        "workers", "declared", "cpus_per_rank", "clamped", "reason",
    }
    assert isinstance(clamp_worker_count(4, t), WorkerDecision)
