"""Unit tests for training-loop utility functions.

Targets ``mriforge.infrastructure.training.utils.training_utils``:
- ``sample_diffusion_timesteps``
- ``clamp_to_range``
- ``handle_training_error``

The ``ProfilingState``, ``LimitedDataLoader``, and ``SafeGradScaler``
classes are thin wrappers around GPU / DataLoader resources — they are
explicitly excluded from this pass (GPU-only / dataloader-requiring).
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------


def test_canary_sample_diffusion_timesteps_shape() -> None:
    from mriforge.infrastructure.training.utils.training_utils import (
        sample_diffusion_timesteps,
    )

    t = sample_diffusion_timesteps(batch_size=8, num_timesteps=1000)
    assert t.shape == (8,)


# ---------------------------------------------------------------------------
# sample_diffusion_timesteps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size,num_timesteps", [(1, 10), (4, 100), (16, 1000)])
def test_sample_diffusion_timesteps_range(batch_size: int, num_timesteps: int) -> None:
    from mriforge.infrastructure.training.utils.training_utils import (
        sample_diffusion_timesteps,
    )

    t = sample_diffusion_timesteps(batch_size=batch_size, num_timesteps=num_timesteps)
    assert t.shape == (batch_size,)
    assert t.min().item() >= 0
    assert t.max().item() < num_timesteps


def test_sample_diffusion_timesteps_dtype_long() -> None:
    from mriforge.infrastructure.training.utils.training_utils import (
        sample_diffusion_timesteps,
    )

    t = sample_diffusion_timesteps(8, 1000)
    assert t.dtype == torch.long


def test_sample_diffusion_timesteps_device_cpu() -> None:
    from mriforge.infrastructure.training.utils.training_utils import (
        sample_diffusion_timesteps,
    )

    t = sample_diffusion_timesteps(4, 100, device=torch.device("cpu"))
    assert t.device.type == "cpu"


# ---------------------------------------------------------------------------
# clamp_to_range
# ---------------------------------------------------------------------------


def test_clamp_to_range_clips_out_of_bounds() -> None:
    from mriforge.infrastructure.training.utils.training_utils import clamp_to_range

    t = torch.tensor([-5.0, 0.0, 5.0])
    result = clamp_to_range(t, min_val=-1.0, max_val=1.0)
    assert result.min().item() >= -1.0
    assert result.max().item() <= 1.0


def test_clamp_to_range_no_clamp_when_disabled() -> None:
    from mriforge.infrastructure.training.utils.training_utils import clamp_to_range

    t = torch.tensor([-10.0, 10.0])
    result = clamp_to_range(t, enable=False)
    assert result[0].item() == pytest.approx(-10.0)
    assert result[1].item() == pytest.approx(10.0)


@pytest.mark.parametrize("min_val,max_val", [(-1.0, 1.0), (0.0, 1.0), (-0.5, 0.5)])
def test_clamp_to_range_parametrized_bounds(min_val: float, max_val: float) -> None:
    from mriforge.infrastructure.training.utils.training_utils import clamp_to_range

    t = torch.linspace(-5.0, 5.0, 20)
    result = clamp_to_range(t, min_val=min_val, max_val=max_val)
    assert result.min().item() >= min_val - 1e-6
    assert result.max().item() <= max_val + 1e-6


def test_clamp_to_range_in_bounds_unchanged() -> None:
    from mriforge.infrastructure.training.utils.training_utils import clamp_to_range

    t = torch.tensor([0.0, 0.5, -0.5])
    result = clamp_to_range(t, min_val=-1.0, max_val=1.0)
    assert torch.allclose(result, t)


# ---------------------------------------------------------------------------
# handle_training_error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error_msg,expected_type",
    [
        ("Input type does not match weight type", "device_mismatch"),
        ("criterion_gan_loss undefined", "loss_function_error"),
        ("invalid syntax error", "syntax_error"),
        ("something completely unknown happened", "unknown_error"),
    ],
)
def test_handle_training_error_returns_correct_type(error_msg: str, expected_type: str) -> None:
    from mriforge.infrastructure.training.utils.training_utils import handle_training_error

    result = handle_training_error(
        error=Exception(error_msg),
        batch_idx=0,
        epoch=1,
        model_type="gan",
    )
    assert result == expected_type


# ---------------------------------------------------------------------------
# Edge: single-step diffusion
# ---------------------------------------------------------------------------


def test_edge_single_timestep_always_zero() -> None:
    from mriforge.infrastructure.training.utils.training_utils import (
        sample_diffusion_timesteps,
    )

    t = sample_diffusion_timesteps(10, num_timesteps=1)
    # With num_timesteps=1 all samples must be 0 (randint [0, 1))
    assert (t == 0).all()


# ---------------------------------------------------------------------------
# No import-time device side effect (2026-06 infra audit)
# ---------------------------------------------------------------------------


def test_module_import_has_no_eager_device_global() -> None:
    """Importing the utils module must NOT eagerly run ``initialize_device``.

    The old module-scope ``device = initialize_device()`` initialised CUDA,
    set a 0.85 ``set_per_process_memory_fraction`` cap, and flipped
    ``cudnn.benchmark = True`` as a *side effect of import* — affecting every
    process that merely imported the package (CLI, audit, tests), with zero
    consumers of the resulting global. The eager global must be gone.
    """
    import mriforge.infrastructure.training.utils.training_utils as tu

    assert not hasattr(tu, "device"), (
        "module-scope `device` global re-introduced an import-time CUDA "
        "side effect (memory cap + cudnn.benchmark)"
    )


def test_get_default_device_is_lazy_and_callable() -> None:
    """A lazy accessor replaces the eager global for callers that want it."""
    from mriforge.infrastructure.training.utils.training_utils import (
        get_default_device,
    )

    dev = get_default_device()
    assert isinstance(dev, torch.device)
    # Cached: a second call returns the same object, no re-init.
    assert get_default_device() is dev


# ---------------------------------------------------------------------------
# MRIFORGE_GPU_MEMORY_FRACTION knob (2026-06 infra audit follow-up)
# ---------------------------------------------------------------------------


class TestGpuMemoryFractionKnob:
    """The 85 % GPU-memory cap inside ``initialize_device`` is now a wired
    env knob (pitfall #15: read + validate + stamp). Invalid values RAISE
    (pitfall #9) rather than silently falling back."""

    def test_default_fraction(self) -> None:
        from mriforge.infrastructure.training.utils.training_utils import (
            _resolve_gpu_memory_fraction,
        )

        assert _resolve_gpu_memory_fraction() == pytest.approx(0.85)

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from mriforge.infrastructure.training.utils.training_utils import (
            _resolve_gpu_memory_fraction,
        )

        monkeypatch.setenv("MRIFORGE_GPU_MEMORY_FRACTION", "0.5")
        assert _resolve_gpu_memory_fraction() == pytest.approx(0.5)

    @pytest.mark.parametrize("bad", ["0", "-0.2", "1.5", "abc", ""])
    def test_invalid_values_raise(
        self, monkeypatch: pytest.MonkeyPatch, bad: str
    ) -> None:
        from mriforge.infrastructure.training.utils.training_utils import (
            _resolve_gpu_memory_fraction,
        )

        monkeypatch.setenv("MRIFORGE_GPU_MEMORY_FRACTION", bad)
        with pytest.raises(ValueError, match="MRIFORGE_GPU_MEMORY_FRACTION"):
            _resolve_gpu_memory_fraction()
