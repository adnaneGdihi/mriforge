"""Unit tests for the calibration score registry.

Targets ``mriforge.infrastructure.calibration.score_registry``.

The registry is a thin dispatch layer that maps YAML keys to score
callables. We verify:
- Built-in scores are registered at import time.
- ``resolve_calibration_score`` raises :class:`KeyError` for unknown names.
- ``register_calibration_score`` raises :class:`ValueError` on conflicting
  re-registrations but allows idempotent re-registration of the *same*
  callable.
- Registered callables produce correct output shapes and non-negative values
  (for ``absolute_residual``).
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------


def test_canary_import_and_list() -> None:
    """Registry is importable and has built-in scores populated."""
    from mriforge.infrastructure.calibration.score_registry import (
        list_calibration_scores,
        resolve_calibration_score,
    )

    names = list_calibration_scores()
    assert isinstance(names, list)
    assert len(names) >= 1, "At least one built-in score must be registered."
    # All items must be strings
    for n in names:
        assert isinstance(n, str)
    # absolute_residual is always present
    fn = resolve_calibration_score("absolute_residual")
    assert callable(fn)


# ---------------------------------------------------------------------------
# Parametrized: built-in scores return correct shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "score_name",
    ["absolute_residual", "pixel_residual"],
)
def test_builtin_scores_return_same_shape(score_name: str) -> None:
    from mriforge.infrastructure.calibration.score_registry import (
        resolve_calibration_score,
    )

    fn = resolve_calibration_score(score_name)
    pred = torch.randn(4, 2, 8, 8)
    target = torch.randn(4, 2, 8, 8)
    out = fn(pred, target)
    assert out.shape == pred.shape, (
        f"{score_name}: expected output shape {pred.shape}, got {out.shape}"
    )


def test_absolute_residual_non_negative() -> None:
    from mriforge.infrastructure.calibration.score_registry import (
        resolve_calibration_score,
    )

    fn = resolve_calibration_score("absolute_residual")
    pred = torch.randn(8)
    target = torch.randn(8)
    out = fn(pred, target)
    assert (out >= 0).all(), "absolute_residual must be non-negative"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_resolve_unknown_raises_key_error() -> None:
    from mriforge.infrastructure.calibration.score_registry import (
        resolve_calibration_score,
    )

    with pytest.raises(KeyError, match="not registered"):
        resolve_calibration_score("__totally_nonexistent_score__")


def test_resolve_empty_string_raises() -> None:
    from mriforge.infrastructure.calibration.score_registry import (
        resolve_calibration_score,
    )

    with pytest.raises(KeyError):
        resolve_calibration_score("")


# ---------------------------------------------------------------------------
# Register / conflict guard
# ---------------------------------------------------------------------------


def test_register_new_score_and_resolve() -> None:
    from mriforge.infrastructure.calibration.score_registry import (
        CALIBRATION_SCORE_REGISTRY,
        register_calibration_score,
        resolve_calibration_score,
    )

    # Use a unique name to avoid polluting other tests.
    unique_name = "__test_register_new_score_xyz__"
    # Clean up if leftover from a previous run.
    CALIBRATION_SCORE_REGISTRY.pop(unique_name, None)

    @register_calibration_score(unique_name)
    def _dummy(
        prediction: "torch.Tensor", target: "torch.Tensor", **_
    ) -> "torch.Tensor":
        return (prediction - target).abs()

    fn = resolve_calibration_score(unique_name)
    assert callable(fn)
    pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([1.0, 2.0])
    out = fn(pred, target)
    assert (out == 0).all()

    # Clean up.
    CALIBRATION_SCORE_REGISTRY.pop(unique_name, None)


def test_register_conflict_raises_value_error() -> None:
    from mriforge.infrastructure.calibration.score_registry import (
        CALIBRATION_SCORE_REGISTRY,
        register_calibration_score,
    )

    unique_name = "__test_register_conflict_xyz__"
    CALIBRATION_SCORE_REGISTRY.pop(unique_name, None)

    @register_calibration_score(unique_name)
    def _fn_a(p: "torch.Tensor", t: "torch.Tensor", **_) -> "torch.Tensor":
        return p - t

    with pytest.raises(ValueError, match="already registered"):

        @register_calibration_score(unique_name)
        def _fn_b(p: "torch.Tensor", t: "torch.Tensor", **_) -> "torch.Tensor":
            return p + t

    # Clean up.
    CALIBRATION_SCORE_REGISTRY.pop(unique_name, None)


def test_physics_residual_registered_and_dispatches_kwargs() -> None:
    """``physics_residual`` is registered and forwards ``score_kwargs``.

    The registry contract is ``(prediction, target, **kwargs)``; PR-CC's
    physics score needs the acquisition vector and contrast, which must flow
    through the ``**kwargs`` (``score_kwargs``) channel.
    """
    from mriforge.infrastructure.calibration.score_registry import (
        list_calibration_scores,
        resolve_calibration_score,
    )

    assert "physics_residual" in list_calibration_scores()
    fn = resolve_calibration_score("physics_residual")

    rho = torch.tensor([[[[1.0, 0.8], [0.6, 0.9]]]])
    t1 = torch.tensor([[[[800.0, 1200.0], [600.0, 1000.0]]]])
    t2 = torch.tensor([[[[80.0, 120.0], [60.0, 100.0]]]])
    params = torch.cat([rho, t1, t2], dim=1)

    from mriforge.infrastructure.physics.multi_physics_bloch import (
        MultiPhysicsBlochLayer,
    )

    layer = MultiPhysicsBlochLayer(learnable_flip_angle=False)
    target = layer(
        {"rho": params[:, 0:1], "t1": params[:, 1:2], "t2": params[:, 2:3]},
        {"TR": 3000.0, "TE": 90.0, "TI": 0.0, "contrast_type": 0},
    )
    out = fn(
        params,
        target,
        acquisition={"TR": 3000.0, "TE": 90.0, "TI": 0.0},
        contrast_type="spin_echo",
    )
    assert out.shape == target.shape
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)


def test_idempotent_reregistration_allowed() -> None:
    """Re-registering the *same* callable under the same name is a no-op."""
    from mriforge.infrastructure.calibration.score_registry import (
        CALIBRATION_SCORE_REGISTRY,
        register_calibration_score,
    )

    unique_name = "__test_idempotent_xyz__"
    CALIBRATION_SCORE_REGISTRY.pop(unique_name, None)

    def _fn(p: "torch.Tensor", t: "torch.Tensor", **_) -> "torch.Tensor":
        return p - t

    register_calibration_score(unique_name)(_fn)
    # Second registration of the same object must not raise.
    register_calibration_score(unique_name)(_fn)

    CALIBRATION_SCORE_REGISTRY.pop(unique_name, None)
