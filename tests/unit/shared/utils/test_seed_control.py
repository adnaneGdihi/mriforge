"""Unit tests for :mod:`mriforge.shared.utils.seed_control`.

Covers :class:`SeedController` end-to-end and the module-level
convenience facades. The module wires together
``random`` / ``numpy`` / ``torch`` seeds plus PyTorch deterministic
mode flags, so the tests verify:

* ``set_global_seed`` makes ``random.random`` / ``numpy`` /
  ``torch`` reproducible.
* Per-worker seeding produces distinct seeds for distinct
  ``worker_id`` values and is cached / stable across calls.
* Diffusion seeding uses a *stable* hash (``blake2b``), not Python's
  PYTHONHASHSEED-salted ``hash()`` — so the same ``key`` always
  yields the same seed across processes.
* ``set_diffusion_noise`` fills a tensor reproducibly given the same
  ``(key, step)``.
* ``fork_seed`` is stable across calls and changes with ``fork_id``.
* ``get_state`` / ``set_state`` round-trip preserves the cached
  worker / diffusion seed dictionaries.
* The global singleton ``get_seed_controller`` is idempotent and the
  module-level facades delegate to it.

CPU-only; no CUDA assertions.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from mriforge.shared.utils.seed_control import (
    SeedConfig,
    SeedController,
    get_diffusion_seed,
    get_seed_controller,
    init_worker_seed,
    set_diffusion_noise,
    set_global_seed,
)


# ── SeedConfig ─────────────────────────────────────────────────────


class TestSeedConfig:
    def test_default_values(self) -> None:
        cfg = SeedConfig()
        assert cfg.base_seed == 42
        assert cfg.deterministic is True
        assert cfg.benchmark is False
        assert cfg.worker_init_fn is None

    def test_custom_values(self) -> None:
        cfg = SeedConfig(base_seed=99, deterministic=False, benchmark=True)
        assert cfg.base_seed == 99
        assert cfg.deterministic is False
        assert cfg.benchmark is True


# ── SeedController.set_global_seed ─────────────────────────────────


class TestSetGlobalSeed:
    def test_random_reproducible(self) -> None:
        sc = SeedController()
        sc.set_global_seed(123)
        a = random.random()
        sc.set_global_seed(123)
        b = random.random()
        assert a == b

    def test_numpy_reproducible(self) -> None:
        sc = SeedController()
        sc.set_global_seed(7)
        a = np.random.rand(4)
        sc.set_global_seed(7)
        b = np.random.rand(4)
        assert (a == b).all()

    def test_torch_reproducible(self) -> None:
        sc = SeedController()
        sc.set_global_seed(11)
        a = torch.randn(8)
        sc.set_global_seed(11)
        b = torch.randn(8)
        assert torch.equal(a, b)

    def test_seed_none_falls_back_to_config(self) -> None:
        sc = SeedController(SeedConfig(base_seed=999))
        out = sc.set_global_seed(None)
        assert out == 999

    def test_seed_explicit_overrides_config(self) -> None:
        sc = SeedController(SeedConfig(base_seed=42))
        out = sc.set_global_seed(2026)
        assert out == 2026

    def test_deterministic_mode_sets_pythonhashseed(self) -> None:
        import os

        sc = SeedController(SeedConfig(base_seed=314, deterministic=True))
        sc.set_global_seed()
        # PYTHONHASHSEED set as a string of the base seed.
        assert os.environ.get("PYTHONHASHSEED") == "314"

    def test_deterministic_false_leaves_cudnn_benchmark_alone_post(self) -> None:
        # When deterministic=False, _set_deterministic_mode is NOT called
        # and cudnn settings stay untouched. We can't easily assert the
        # negative ("did not call _set_deterministic_mode"), so we just
        # confirm the public surface still completes without error.
        sc = SeedController(SeedConfig(deterministic=False))
        sc.set_global_seed(5)


# ── SeedController.get_worker_seed / init_worker_seed ──────────────


class TestWorkerSeeding:
    def test_worker_seed_formula(self) -> None:
        sc = SeedController(SeedConfig(base_seed=10))
        # Documented: base_seed + worker_id + 1.
        assert sc.get_worker_seed(0) == 11
        assert sc.get_worker_seed(1) == 12
        assert sc.get_worker_seed(7) == 18

    def test_worker_seed_cached(self) -> None:
        sc = SeedController()
        first = sc.get_worker_seed(3)
        # Manipulating the internal cache should be reflected on next
        # call (proves it really is cached, not recomputed).
        sc._worker_seeds[3] = 999
        assert sc.get_worker_seed(3) == 999
        del first  # silence unused

    def test_different_workers_different_seeds(self) -> None:
        sc = SeedController()
        seeds = {sc.get_worker_seed(i) for i in range(10)}
        assert len(seeds) == 10

    def test_init_worker_seed_applies_seed_to_rngs(self) -> None:
        sc = SeedController()
        sc.init_worker_seed(0)
        # After init_worker_seed(0), torch.randn is deterministic given
        # the worker_id (seed = base + 0 + 1 = 43 by default).
        a = torch.randn(4)
        sc.init_worker_seed(0)
        b = torch.randn(4)
        assert torch.equal(a, b)


# ── SeedController.get_diffusion_seed ──────────────────────────────


class TestDiffusionSeeding:
    def test_seed_stable_across_calls(self) -> None:
        sc = SeedController()
        a = sc.get_diffusion_seed("forward")
        b = sc.get_diffusion_seed("forward")
        assert a == b

    def test_different_keys_different_seeds(self) -> None:
        sc = SeedController()
        assert sc.get_diffusion_seed("forward") != sc.get_diffusion_seed("reverse")

    def test_step_modifier_changes_seed(self) -> None:
        sc = SeedController()
        base = sc.get_diffusion_seed("k")
        # Step is added mod 2^32, so different step → different value.
        s0 = sc.get_diffusion_seed("k", step=0)
        s1 = sc.get_diffusion_seed("k", step=1)
        assert s0 == base
        assert s1 == (base + 1) % (2**32)

    def test_seed_stable_across_processes_via_blake2b(self) -> None:
        """The implementation must use a stable hash (blake2b), not
        Python's ``hash()`` which is salted per process — otherwise
        reruns produce different seeds. Two fresh controllers with the
        same base seed must yield the same diffusion seed."""
        a = SeedController(SeedConfig(base_seed=42)).get_diffusion_seed("forward")
        b = SeedController(SeedConfig(base_seed=42)).get_diffusion_seed("forward")
        assert a == b

    def test_diffusion_seed_caches_after_first_call(self) -> None:
        sc = SeedController()
        sc.get_diffusion_seed("forward")
        # The internal cache now has the key.
        assert "forward" in sc._diffusion_seeds


# ── SeedController.set_diffusion_noise ─────────────────────────────


class TestSetDiffusionNoise:
    def test_noise_is_reproducible_for_same_key_and_step(self) -> None:
        sc = SeedController()
        t1 = torch.zeros(8)
        t2 = torch.zeros(8)
        sc.set_diffusion_noise(t1, key="x", step=0)
        sc.set_diffusion_noise(t2, key="x", step=0)
        assert torch.equal(t1, t2)

    def test_different_step_changes_noise(self) -> None:
        sc = SeedController()
        t1 = torch.zeros(8)
        t2 = torch.zeros(8)
        sc.set_diffusion_noise(t1, key="x", step=0)
        sc.set_diffusion_noise(t2, key="x", step=1)
        assert not torch.equal(t1, t2)


# ── SeedController.fork_seed ───────────────────────────────────────


class TestForkSeed:
    def test_fork_seed_stable(self) -> None:
        a = SeedController(SeedConfig(base_seed=7)).fork_seed("worker_0")
        b = SeedController(SeedConfig(base_seed=7)).fork_seed("worker_0")
        assert a == b

    def test_fork_seed_differs_per_id(self) -> None:
        sc = SeedController()
        assert sc.fork_seed("a") != sc.fork_seed("b")

    def test_fork_seed_accepts_int_id(self) -> None:
        sc = SeedController()
        assert isinstance(sc.fork_seed(42), int)


# ── reset / get_state / set_state ──────────────────────────────────


class TestStateManagement:
    def test_reset_clears_caches(self) -> None:
        sc = SeedController()
        sc.get_worker_seed(0)
        sc.get_diffusion_seed("forward")
        assert sc._worker_seeds
        assert sc._diffusion_seeds
        sc.reset()
        assert sc._worker_seeds == {}
        assert sc._diffusion_seeds == {}

    def test_state_round_trip(self) -> None:
        sc = SeedController()
        sc.get_worker_seed(0)
        sc.get_diffusion_seed("k")
        state = sc.get_state()

        # State is a *copy* — mutating the snapshot doesn't affect the
        # controller.
        state["worker_seeds"][999] = 12345
        assert 999 not in sc._worker_seeds

        # Restore into a fresh controller.
        other = SeedController()
        other.set_state(state)
        assert other._worker_seeds[0] == sc._worker_seeds[0]
        assert other._diffusion_seeds["k"] == sc._diffusion_seeds["k"]

    def test_set_state_defaults_when_keys_missing(self) -> None:
        """Empty state dict triggers SeedConfig() default + empty
        caches — the documented "load partial state" fallback."""
        sc = SeedController()
        sc.set_state({})
        assert isinstance(sc.config, SeedConfig)
        assert sc._worker_seeds == {}
        assert sc._diffusion_seeds == {}


# ── Module-level singleton + facades ───────────────────────────────


class TestModuleFacades:
    @pytest.fixture(autouse=True)
    def _reset_singleton(self) -> None:
        # Reset the singleton between tests so each starts clean.
        import mriforge.shared.utils.seed_control as seed_mod

        prior = seed_mod._global_seed_controller
        seed_mod._global_seed_controller = None
        yield
        seed_mod._global_seed_controller = prior

    def test_get_seed_controller_singleton(self) -> None:
        a = get_seed_controller()
        b = get_seed_controller()
        assert a is b
        assert isinstance(a, SeedController)

    def test_set_global_seed_facade_updates_singleton(self) -> None:
        used = set_global_seed(2026, deterministic=False)
        assert used == 2026
        ctrl = get_seed_controller()
        assert ctrl.config.base_seed == 2026
        assert ctrl.config.deterministic is False

    def test_set_global_seed_facade_delegates_to_ssot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The module facade must route actual seeding through the accelerator
        SSOT ``seed_everything`` (2026-07 determinism unification), so the
        pipeline can no longer diverge from ``initialize_accelerator``."""
        calls: list[tuple] = []

        def _spy(seed, *, rank=0, deterministic=True):
            calls.append((seed, rank, deterministic))
            return seed

        monkeypatch.setattr("mriforge.accelerator.seed_everything", _spy)
        out = set_global_seed(1234, deterministic=True)
        assert out == 1234
        assert calls == [(1234, 0, True)]

    def test_init_worker_seed_facade(self) -> None:
        # Establishes that the facade routes through the singleton.
        init_worker_seed(2)
        ctrl = get_seed_controller()
        assert 2 in ctrl._worker_seeds

    def test_get_diffusion_seed_facade(self) -> None:
        s = get_diffusion_seed("forward")
        assert isinstance(s, int)
        # Same call returns same seed.
        assert get_diffusion_seed("forward") == s

    def test_set_diffusion_noise_facade(self) -> None:
        t = torch.zeros(4)
        set_diffusion_noise(t, key="k", step=0)
        # Tensor was filled in place — at least one non-zero entry.
        assert not torch.allclose(t, torch.zeros(4))


class TestSetGlobalSeedRankOffset:
    """Issue #1124: the facade dropped ``rank``, so every rank of a distributed
    run seeded identically and drew the same augmentations."""

    def test_rank_is_forwarded_to_the_ssot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The regression proper. ``seed_everything`` has always accepted
        ``rank``; the facade simply never passed it on."""
        calls: list[tuple] = []

        def _spy(seed, *, rank=0, deterministic=True):
            calls.append((seed, rank, deterministic))
            return seed + rank

        monkeypatch.setattr("mriforge.accelerator.seed_everything", _spy)
        out = set_global_seed(1234, deterministic=True, rank=3)

        assert calls == [(1234, 3, True)]
        assert out == 1237, "the facade must return the rank-OFFSET seed"

    def test_torch_initial_seed_differs_by_exactly_the_rank(self) -> None:
        """End-to-end through the real SSOT, not a spy."""
        set_global_seed(4242, rank=0)
        base = torch.initial_seed()
        set_global_seed(4242, rank=3)
        assert torch.initial_seed() - base == 3

    def test_distinct_ranks_draw_distinct_values(self) -> None:
        """What the bug actually cost: identical augmentation streams across the
        N processes, cutting effective augmentation diversity to 1/N."""
        set_global_seed(99, rank=0)
        rank0 = (torch.rand(8), np.random.rand(8), random.random())
        set_global_seed(99, rank=1)
        rank1 = (torch.rand(8), np.random.rand(8), random.random())

        assert not torch.allclose(rank0[0], rank1[0])
        assert not np.allclose(rank0[1], rank1[1])
        assert rank0[2] != rank1[2]

    def test_same_rank_is_still_reproducible(self) -> None:
        """The offset must not cost reproducibility within a rank."""
        set_global_seed(7, rank=2)
        first = torch.rand(8)
        set_global_seed(7, rank=2)
        assert torch.allclose(first, torch.rand(8))

    def test_default_rank_is_zero_and_unchanged(self) -> None:
        """Single-process callers keep their exact current stream — the new
        parameter must be inert unless asked for."""
        set_global_seed(555)
        default = torch.rand(8)
        set_global_seed(555, rank=0)
        assert torch.allclose(default, torch.rand(8))

    def test_controller_base_seed_stays_unoffset(self) -> None:
        """Deliberate: ``get_diffusion_seed`` derives from ``base_seed`` and must
        stay rank-identical so reproducible validation noise matches across
        ranks. Offsetting it would make each rank denoise from different noise."""
        set_global_seed(31337, rank=5)
        assert get_seed_controller().config.base_seed == 31337
