"""Unit tests for :mod:`mriforge.accelerator`.

The accelerator module wires three concerns together:

* DataLoader worker seeding (``seed_worker``).
* Process-level torch / numpy / random seeding via
  ``initialize_accelerator``.
* Cache-directory plumbing (``TMPDIR``, ``TORCH_HOME``,
  ``CUDA_CACHE_CONFIG``) using :func:`resolve_cache_root` from the
  config layer.

The tests run on CPU only — we patch ``torch.cuda.is_available`` to
``False`` and exercise the fallback branch.
"""

from __future__ import annotations

import importlib
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch

import mriforge.accelerator as accelerator_mod
from mriforge.accelerator import initialize_accelerator, seed_everything, seed_worker

# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Each test gets its own ``TMPDIR`` / cache root and a fresh seed
    state so we don't pollute the host environment or other tests."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    # Force resolve_cache_root() to use TMPDIR rather than ~/.cache/.
    monkeypatch.delenv("MRIFORGE_CACHE_ROOT", raising=False)
    # Pop any stale settings from earlier tests so initialize_accelerator
    # re-applies the setdefault chain.
    for key in ("TORCH_HOME", "CUDA_CACHE_CONFIG", "PYTORCH_CUDA_ALLOC_CONF"):
        monkeypatch.delenv(key, raising=False)
    return tmp_path


@pytest.fixture
def force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend CUDA is not available so the fallback branch is taken."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


# ── seed_worker ────────────────────────────────────────────────────


class TestSeedWorker:
    def test_seed_worker_seeds_numpy_and_random(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Calling ``seed_worker`` must seed both ``numpy`` and the
        stdlib ``random`` module using ``torch.initial_seed()`` modulo
        2**32. Patch torch.initial_seed to a known value and check both
        post-call states."""
        import torch

        monkeypatch.setattr(torch, "initial_seed", lambda: 12345)

        seed_worker(0)
        np_value_a = np.random.random()
        py_value_a = random.random()

        # Re-seed and confirm reproducibility.
        seed_worker(0)
        np_value_b = np.random.random()
        py_value_b = random.random()

        assert np_value_a == np_value_b, "numpy seed must be reproducible"
        assert py_value_a == py_value_b, "random seed must be reproducible"

    def test_different_torch_seeds_yield_different_states(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch

        monkeypatch.setattr(torch, "initial_seed", lambda: 11)
        seed_worker(0)
        a = np.random.random()

        monkeypatch.setattr(torch, "initial_seed", lambda: 22)
        seed_worker(0)
        b = np.random.random()

        assert a != b, "different torch.initial_seed must change numpy state"


# ── initialize_accelerator — CPU branch ────────────────────────────


class TestInitializeAcceleratorCPU:
    def test_cpu_returns_cpu_device(self, force_cpu: None) -> None:
        import torch

        device = initialize_accelerator("cpu", seed=42)
        assert isinstance(device, torch.device)
        assert device.type == "cpu"

    def test_cuda_request_raises_without_gpu(self, force_cpu: None) -> None:
        """Regression: a cuda request on a GPU-less host must RAISE.

        Pre-fix this fell back to CPU with a warning, so an sbatch that lost its
        GPU allocation ran ~100x slower and still reported success. The
        accelerated-run contract refuses to degrade silently.
        """
        from mriforge.core.compute_device import AcceleratorRequiredError

        with pytest.raises(AcceleratorRequiredError, match="CUDA is not available"):
            initialize_accelerator("cuda", seed=42)

    def test_auto_raises_on_heavy_pipeline_without_gpu(self, force_cpu: None) -> None:
        """``auto`` must not quietly mean "cpu" for a training run."""
        from mriforge.core.compute_device import AcceleratorRequiredError

        with pytest.raises(AcceleratorRequiredError, match="no accelerator"):
            initialize_accelerator("auto", seed=42, pipeline="train")

    def test_force_cpu_env_permits_cpu_run(
        self, force_cpu: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The sanctioned opt-out: the user dictates CPU explicitly."""
        monkeypatch.setenv("FORCE_CPU", "true")
        device = initialize_accelerator("auto", seed=42, pipeline="train")
        assert device.type == "cpu"

    def test_default_seed_is_42(
        self, monkeypatch: pytest.MonkeyPatch, force_cpu: None
    ) -> None:
        """When ``seed=None`` the module substitutes 42."""
        seen: list[int] = []
        import torch

        def _capture_manual_seed(value: int) -> None:
            seen.append(value)

        monkeypatch.setattr(torch, "manual_seed", _capture_manual_seed)
        initialize_accelerator("cpu", seed=None)
        # rank defaults to 0 → seed_used == 42.
        assert seen == [42]

    def test_rank_offset_applied_to_seed(
        self, monkeypatch: pytest.MonkeyPatch, force_cpu: None
    ) -> None:
        seen: list[int] = []
        import torch

        monkeypatch.setattr(torch, "manual_seed", lambda v: seen.append(v))
        initialize_accelerator("cpu", seed=100, rank=3)
        assert seen == [103], "Per-rank seed must be seed + rank"


# ── initialize_accelerator — cache directory plumbing ──────────────


class TestInitializeAcceleratorCachePaths:
    def test_torch_home_and_cuda_cache_set_under_tmpdir(
        self, force_cpu: None, tmp_path: Path
    ) -> None:
        initialize_accelerator("cpu", seed=42)
        # resolve_cache_root reads TMPDIR (from the fixture) and tacks
        # on `mriforge_cache`. Both env vars must be set AND on disk.
        torch_home = os.environ.get("TORCH_HOME")
        cuda_cache = os.environ.get("CUDA_CACHE_CONFIG")
        assert torch_home is not None
        assert cuda_cache is not None
        # Both paths exist as directories — initialize_accelerator
        # mkdir-s them.
        assert Path(torch_home).is_dir()
        assert Path(cuda_cache).is_dir()
        # Both roots are under the test's tmp_path.
        assert str(tmp_path) in torch_home
        assert str(tmp_path) in cuda_cache

    def test_pytorch_cuda_alloc_conf_set_when_unset(self, force_cpu: None) -> None:
        # Fixture deleted any pre-existing value; verify init sets the
        # documented default. The canonical default is now
        # ``expandable_segments:True,max_split_size_mb:512`` — accept either
        # the legacy expandable-only form or the full pair.
        initialize_accelerator("cpu", seed=42)
        value = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        assert (
            "expandable_segments:True" in value
        ), f"PYTORCH_CUDA_ALLOC_CONF must enable expandable_segments; got {value!r}"

    def test_pytorch_cuda_alloc_conf_preserved_when_already_set(
        self, force_cpu: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # User-set value must not be overwritten.
        monkeypatch.setenv(
            "PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8"
        )
        initialize_accelerator("cpu", seed=42)
        assert (
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "garbage_collection_threshold:0.8"
        )


# ── Module surface ─────────────────────────────────────────────────


def test_module_exports_documented_symbols() -> None:
    """Public symbols imported by the rest of the codebase must stay
    importable."""
    assert hasattr(accelerator_mod, "seed_worker")
    assert hasattr(accelerator_mod, "initialize_accelerator")


def test_module_reloads_cleanly() -> None:
    """A reload must not raise — defensive guard against accidental
    module-level side effects (e.g. eager CUDA init)."""
    importlib.reload(accelerator_mod)


# ── cuDNN determinism consistency (2026-06 infra audit) ────────────


def test_cudnn_flags_consistent_when_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``initialize_accelerator(deterministic=True)`` must NOT leave the
    contradictory ``benchmark=True`` + ``deterministic=True`` pair the old
    SSOT set. Deterministic means benchmark OFF.
    """
    import types as _types

    fake_cudnn = _types.SimpleNamespace(benchmark=None, deterministic=None)
    monkeypatch.setattr(torch.backends, "cudnn", fake_cudnn, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *a, **k: None)
    monkeypatch.setattr(
        torch, "use_deterministic_algorithms", lambda *a, **k: None, raising=False
    )

    accelerator_mod.initialize_accelerator("cuda", seed=1, deterministic=True)

    assert fake_cudnn.deterministic is True
    assert fake_cudnn.benchmark is False  # the old code set this True — the bug


def test_cudnn_benchmark_enabled_when_not_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opting out of determinism enables the autotuner for speed."""
    import types as _types

    fake_cudnn = _types.SimpleNamespace(benchmark=None, deterministic=None)
    monkeypatch.setattr(torch.backends, "cudnn", fake_cudnn, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *a, **k: None)
    monkeypatch.setattr(
        torch, "use_deterministic_algorithms", lambda *a, **k: None, raising=False
    )

    accelerator_mod.initialize_accelerator("cuda", seed=1, deterministic=False)

    assert fake_cudnn.benchmark is True
    assert fake_cudnn.deterministic is False


def test_determinism_policy_applies_on_cpu_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``use_deterministic_algorithms`` must be requested even without CUDA.

    The hardcoded main.py override this SSOT absorbed applied determinism
    unconditionally; CPU ops (scatter/index reductions) also have
    non-deterministic variants, so the policy cannot live behind the
    cuda-available gate.
    """
    import types as _types

    fake_cudnn = _types.SimpleNamespace(benchmark=None, deterministic=None)
    monkeypatch.setattr(torch.backends, "cudnn", fake_cudnn, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    calls: list[tuple] = []
    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        lambda *a, **k: calls.append((a, k)),
        raising=False,
    )

    device = accelerator_mod.initialize_accelerator("cpu", seed=7, deterministic=True)

    assert device.type == "cpu"
    assert calls, "use_deterministic_algorithms never requested on CPU"
    assert calls[0][0] == (True,)
    assert calls[0][1].get("warn_only") is True
    assert fake_cudnn.benchmark is False
    assert fake_cudnn.deterministic is True


# ── seed_everything — the shared seeding SSOT (2026-07 unification) ─


class TestSeedEverything:
    """``seed_everything`` is the single implementation both
    ``initialize_accelerator`` and ``set_global_seed`` delegate to, so the CLI
    entry and the training pipeline can no longer apply divergent policies."""

    def test_reproducible_across_torch_numpy_random(self, force_cpu: None) -> None:
        import torch

        seed_everything(123)
        t_a, np_a, py_a = torch.randn(4), np.random.rand(4), random.random()
        seed_everything(123)
        t_b, np_b, py_b = torch.randn(4), np.random.rand(4), random.random()
        assert torch.equal(t_a, t_b)
        assert (np_a == np_b).all()
        assert py_a == py_b

    def test_returns_rank_offset_seed(self, force_cpu: None) -> None:
        assert seed_everything(100, rank=0) == 100
        assert seed_everything(100, rank=3) == 103

    def test_rank_offset_changes_state(self, force_cpu: None) -> None:
        import torch

        seed_everything(50, rank=0)
        a = torch.randn(4)
        seed_everything(50, rank=1)
        b = torch.randn(4)
        assert not torch.equal(a, b), "rank offset must diversify per-rank RNG"

    def test_deterministic_true_sets_policy_incl_use_deterministic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import types as _types

        fake_cudnn = _types.SimpleNamespace(benchmark=None, deterministic=None)
        monkeypatch.setattr(torch.backends, "cudnn", fake_cudnn, raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        calls: list[tuple] = []
        monkeypatch.setattr(
            torch,
            "use_deterministic_algorithms",
            lambda *a, **k: calls.append((a, k)),
            raising=False,
        )

        seed_everything(7, deterministic=True)

        assert fake_cudnn.benchmark is False
        assert fake_cudnn.deterministic is True
        # The exact call the pipeline seeding previously OMITTED.
        assert calls and calls[0][0] == (True,)
        assert calls[0][1].get("warn_only") is True

    def test_deterministic_false_enables_autotuner(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import types as _types

        fake_cudnn = _types.SimpleNamespace(benchmark=None, deterministic=None)
        monkeypatch.setattr(torch.backends, "cudnn", fake_cudnn, raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        seed_everything(7, deterministic=False)
        assert fake_cudnn.benchmark is True
        assert fake_cudnn.deterministic is False


class TestTF32HasOneOwner:
    """TF32 and ``cudnn.benchmark`` had two writers with opposite intent.

    ``training_utils.initialize_device`` set ``matmul.allow_tf32 = True`` AND
    ``cudnn.benchmark = True`` unconditionally, while ``seed_everything`` sets
    ``benchmark = not deterministic``. ``initialize_device`` is reached lazily
    through ``_DEFAULT_DEVICE``, so whichever ran last won and a run that asked
    for determinism could be silently un-determinized.
    """

    @staticmethod
    def _fake_backends(monkeypatch: pytest.MonkeyPatch):
        import types as _types

        cudnn = _types.SimpleNamespace(
            benchmark=None, deterministic=None, allow_tf32=None
        )
        matmul = _types.SimpleNamespace(allow_tf32=None)
        monkeypatch.setattr(torch.backends, "cudnn", cudnn, raising=False)
        monkeypatch.setattr(
            torch.backends, "cuda", _types.SimpleNamespace(matmul=matmul), raising=False
        )
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        return cudnn, matmul

    def test_tf32_defaults_to_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The pre-existing behaviour is preserved -- this change gives the flag
        an owner, it does not change what a run does."""
        cudnn, matmul = self._fake_backends(monkeypatch)
        seed_everything(7)
        assert matmul.allow_tf32 is True
        assert cudnn.allow_tf32 is True

    def test_tf32_can_be_turned_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cudnn, matmul = self._fake_backends(monkeypatch)
        seed_everything(7, allow_tf32=False)
        assert matmul.allow_tf32 is False
        assert cudnn.allow_tf32 is False

    def test_tf32_is_orthogonal_to_determinism(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """TF32 is reproducible, merely lower-precision, so a deterministic run
        keeps it. Collapsing the two axes would cost every deterministic arm its
        fp32 matmul throughput for no reproducibility gain."""
        cudnn, matmul = self._fake_backends(monkeypatch)
        seed_everything(7, deterministic=True)
        assert cudnn.benchmark is False and cudnn.deterministic is True
        assert matmul.allow_tf32 is True

    def test_initialize_device_no_longer_writes_the_backends(self) -> None:
        """The second writer is gone.

        Asserted on the SOURCE because the branch only runs on a CUDA host, so a
        CPU test cannot otherwise observe that the lines are absent. Matched as
        an ASSIGNMENT rather than a substring: the function still *names* both
        symbols, in the comment explaining why it no longer sets them, and a
        substring test would forbid that explanation.
        """
        import ast
        import inspect
        import textwrap

        from mriforge.infrastructure.training.utils import training_utils

        tree = ast.parse(
            textwrap.dedent(inspect.getsource(training_utils.initialize_device))
        )
        written = {
            ast.unparse(target)
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Attribute)
        }
        offenders = {t for t in written if "allow_tf32" in t or "benchmark" in t}
        assert (
            not offenders
        ), f"initialize_device writes torch.backends again: {offenders}"
