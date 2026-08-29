"""The diffusion model-output snapshot must include the *model input* (x_t).

User report (2026-07-06): "I ... need in snapshot to see the input that was
given to the model, as in some instances the masks were applied to image space
data."

Before this change the training ``model_output`` snapshot captured
``input`` (the raw undersampled measurement ``input_batch``), ``target``, and
the model's output — but NOT ``model_input``, the *degraded* tensor
``x_t = target * mask_t`` (post any measurement conditioning) that is actually
fed to ``model.forward``. So a reviewer could not see what the model received,
and therefore could not tell whether the undersampling mask had been applied in
k-space (correct) or in the image domain (a domain-superposition bug).

``_build_output_snapshot`` is a pure tensor-dict builder, so we exercise it on a
bare instance (the strategy-unit-test pattern in
``test_base_model_output_snapshot_2026_06``) — no torch model / config needed.
"""
from __future__ import annotations

import torch

from mriforge.infrastructure.training.strategies.diffusion import (
    DiffusionTrainingStrategy,
)


def _bare() -> DiffusionTrainingStrategy:
    return object.__new__(DiffusionTrainingStrategy)


def test_snapshot_includes_model_input_x_t() -> None:
    s = _bare()
    x_t = torch.rand(1, 8, 16, 16)  # the degraded tensor fed to the model
    snap = s._build_output_snapshot(
        torch.rand(1, 8, 16, 16),   # hr_fakes_for_loss (post-DC output)
        torch.rand(1, 8, 16, 16),   # predicted_output
        torch.rand(1, 8, 16, 16),   # target_for_loss
        torch.rand(1, 8, 16, 16),   # input_batch (raw measurement)
        None,                        # mask
        model_input=x_t,
    )
    assert "model_input" in snap, "the tensor fed to model.forward must be captured"
    assert torch.equal(snap["model_input"], x_t)
    # input (measurement) and model_input (x_t) are distinct diagnostics.
    assert "input" in snap and "target" in snap


def test_model_input_is_detached() -> None:
    s = _bare()
    x_t = torch.rand(1, 8, 16, 16, requires_grad=True) * 2
    snap = s._build_output_snapshot(
        torch.rand(1, 8, 16, 16),
        torch.rand(1, 8, 16, 16),
        torch.rand(1, 8, 16, 16),
        torch.rand(1, 8, 16, 16),
        None,
        model_input=x_t,
    )
    assert not snap["model_input"].requires_grad  # snapshot must not pin the graph


def test_model_input_optional_backward_compatible() -> None:
    """Omitting model_input keeps the legacy dict (no crash, no key)."""
    s = _bare()
    snap = s._build_output_snapshot(
        torch.rand(1, 8, 16, 16),
        torch.rand(1, 8, 16, 16),
        torch.rand(1, 8, 16, 16),
        torch.rand(1, 8, 16, 16),
        None,
    )
    assert "model_input" not in snap
    assert {"model_output_post_dc", "target", "input"} <= set(snap)
