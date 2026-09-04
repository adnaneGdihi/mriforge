"""Determinism sentinel for :func:`spectramr.accelerator.initialize_accelerator`.

Several training paradigms in this repo (diffusion, cold diffusion,
cycle-Bloch) assume that two runs with the same seed and rank produce
bit-identical RNG streams across the three randomness sources that
:func:`initialize_accelerator` is responsible for:

* :mod:`torch` (CPU and CUDA generators)
* :mod:`numpy`
* the stdlib :mod:`random`

The unit tests in ``tests/unit/test_accelerator.py`` verify each branch
of :func:`initialize_accelerator` in isolation. This integration test
exercises the **end-to-end contract**: drive a short stochastic workload
with the public API, reset state, drive it again, and assert the two
runs produce byte-identical samples. This is the regression sentinel
that catches silent drift introduced by refactors to the seeding chain
— the kind of drift no other test surfaces, because every other test
either re-seeds itself or compares against tolerance.

Run with: ``pytest tests/integration/test_determinism_sentinel.py -v``
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterator

import numpy as np
import pytest
import torch

from spectramr.accelerator import initialize_accelerator

# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[None]:
    """Isolate the test from host env vars and pre-existing cache state.

    The accelerator module uses ``os.environ.setdefault(...)`` for several
    keys, so a pre-existing value would mask the seeding behaviour we want
    to assert. Strip them, force the cache root to a per-test tmpdir, and
    leave the host environment untouched.
    """
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.delenv("SPECTRAMR_CACHE_ROOT", raising=False)
    for key in ("TORCH_HOME", "CUDA_CACHE_CONFIG", "PYTORCH_CUDA_ALLOC_CONF"):
        monkeypatch.delenv(key, raising=False)
    yield


# ── Helpers ────────────────────────────────────────────────────────


def _drive_stochastic_workload(device: torch.device) -> dict[str, object]:
    """Exercise every RNG the accelerator is responsible for.

    Returns a dict of fingerprints that can be compared byte-for-byte
    across two runs. Each sample is sized to be small enough to keep the
    test under one second but large enough to make accidental collisions
    astronomically unlikely.
    """
    # torch generator (CPU path — this test runs on CPU CI runners)
    torch_sample = torch.randn(64, 64, device=device, dtype=torch.float32)

    # numpy generator
    np_sample = np.random.standard_normal(size=(64, 64)).astype(np.float64)

    # stdlib random
    py_sample = tuple(random.random() for _ in range(64))

    # A second torch draw — guards against the legitimate concern that
    # the first draw might match by coincidence on a buggy code path
    # that re-seeded between calls.
    torch_sample_2 = torch.randint(0, 2**31 - 1, (32,), device=device).tolist()

    return {
        "torch_sample": torch_sample.cpu().numpy().tobytes(),
        "torch_sample_2": tuple(torch_sample_2),
        "np_sample": np_sample.tobytes(),
        "py_sample": py_sample,
    }


# ── Tests ──────────────────────────────────────────────────────────


@pytest.mark.integration
class TestDeterminismSentinel:
    """The contract: same seed + same rank → bit-identical RNG streams."""

    def test_same_seed_produces_identical_streams(self, isolated_env: None) -> None:
        """Two consecutive calls with seed=42, rank=0 must yield identical
        torch / numpy / python-random outputs.

        If this test fails, something in the seeding chain has drifted —
        likely an ad-hoc ``torch.manual_seed`` call somewhere downstream of
        ``initialize_accelerator``, or a change in the order of RNG sources
        seeded inside the function itself.
        """
        device_a = initialize_accelerator("cpu", seed=42, rank=0)
        run_a = _drive_stochastic_workload(device_a)

        device_b = initialize_accelerator("cpu", seed=42, rank=0)
        run_b = _drive_stochastic_workload(device_b)

        assert run_a == run_b, (
            "initialize_accelerator(seed=42, rank=0) is not bit-identical "
            "across two consecutive calls. This breaks reproducibility for "
            "diffusion / cold-diffusion / cycle-Bloch paradigms. Inspect "
            "src/spectramr/accelerator.py for changes to the seed-application "
            "order or for stray torch.manual_seed calls in downstream code."
        )

    def test_different_seed_produces_different_streams(
        self, isolated_env: None
    ) -> None:
        """Sanity check: seed=42 and seed=43 must NOT produce identical
        streams. Catches a regression where the seed argument is silently
        ignored (e.g. someone hardcoded ``torch.manual_seed(42)``)."""
        device_a = initialize_accelerator("cpu", seed=42, rank=0)
        run_a = _drive_stochastic_workload(device_a)

        device_b = initialize_accelerator("cpu", seed=43, rank=0)
        run_b = _drive_stochastic_workload(device_b)

        assert run_a != run_b, (
            "initialize_accelerator produced identical streams for seed=42 "
            "and seed=43. The seed argument is being ignored."
        )

    def test_different_rank_produces_different_streams(
        self, isolated_env: None
    ) -> None:
        """Per-rank seeds (seed + rank) must diverge across ranks. This is
        the DDP-correctness contract: each rank must see a different data
        augmentation order even though every rank calls with the same base
        seed."""
        device_a = initialize_accelerator("cpu", seed=42, rank=0)
        run_a = _drive_stochastic_workload(device_a)

        device_b = initialize_accelerator("cpu", seed=42, rank=1)
        run_b = _drive_stochastic_workload(device_b)

        assert run_a != run_b, (
            "initialize_accelerator(seed=42, rank=0) and rank=1 produced "
            "identical streams. The rank offset (worker_seed = seed + rank) "
            "is broken; DDP training will see synchronised augmentations "
            "across GPUs, defeating the purpose of distributed training."
        )

    def test_cross_rng_independence(self, isolated_env: None) -> None:
        """If only one RNG source were being seeded, two runs would match
        on that source and diverge on the others. This test asserts that
        all three sources are tied to the same seed — i.e. seeding torch
        without also seeding numpy / random would fail this check."""
        initialize_accelerator("cpu", seed=42, rank=0)
        first_torch = torch.randn(8).numpy().tobytes()
        first_np = np.random.standard_normal(8).tobytes()
        first_py = tuple(random.random() for _ in range(8))

        initialize_accelerator("cpu", seed=42, rank=0)
        second_torch = torch.randn(8).numpy().tobytes()
        second_np = np.random.standard_normal(8).tobytes()
        second_py = tuple(random.random() for _ in range(8))

        assert first_torch == second_torch, "torch RNG not re-seeded"
        assert first_np == second_np, "numpy RNG not re-seeded"
        assert first_py == second_py, "python random not re-seeded"

    @pytest.mark.gpu
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA generator determinism requires a CUDA device",
    )
    def test_cuda_generator_seeded(self, isolated_env: None) -> None:
        """When CUDA is available, the CUDA generator must also be seeded
        deterministically. Skipped on CPU CI runners; runs on the
        maintainer's GPU validation pass."""
        device_a = initialize_accelerator("cuda", seed=42, rank=0)
        cuda_a = torch.randn(64, device=device_a).cpu().numpy().tobytes()

        device_b = initialize_accelerator("cuda", seed=42, rank=0)
        cuda_b = torch.randn(64, device=device_b).cpu().numpy().tobytes()

        assert cuda_a == cuda_b, (
            "CUDA RNG not deterministic across two initialize_accelerator "
            "calls with the same seed. Check that torch.cuda.manual_seed_all "
            "is being called and that cudnn.deterministic is True."
        )


@pytest.mark.integration
def test_alloc_conf_does_not_leak_across_runs(
    isolated_env: None,
) -> None:
    """``PYTORCH_CUDA_ALLOC_CONF`` is set with ``setdefault`` — once it's
    set in a process, subsequent calls must not overwrite it. This is the
    contract documented inline in ``accelerator.py``. If a refactor changed
    ``setdefault`` to ``__setitem__``, this test catches it."""
    initialize_accelerator("cpu", seed=42, rank=0)
    first = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    assert first is not None

    # Tamper with the value and confirm a second init leaves it alone.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.5"
    initialize_accelerator("cpu", seed=42, rank=0)
    second = os.environ["PYTORCH_CUDA_ALLOC_CONF"]

    assert second == "garbage_collection_threshold:0.5", (
        "initialize_accelerator overwrote a user-set PYTORCH_CUDA_ALLOC_CONF. "
        "It must use os.environ.setdefault() to preserve caller intent."
    )
