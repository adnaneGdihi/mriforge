"""Regression tests for OptimizerMixin's narrowed contract.

Pin the post-audit-05-F2 shape: the mixin only contributes the
optimizer-stepper builder. The runtime gradient ops live on
``BaseTrainingStrategy`` (which sits left of the mixin in MRO and used
to shadow these methods silently). Re-introducing the shadowed methods
would put the codebase right back into the "advertised contract that
never wins dispatch" trap CLAUDE.md pitfall #9 warns against.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
import torch.nn as nn

from spectramr.infrastructure.training.optimizers.amp_policy import AMPPolicy
from spectramr.infrastructure.training.strategies.mixins.optimizer import OptimizerMixin


def test_optimizer_mixin_does_not_define_runtime_gradient_ops() -> None:
    """Pin the audit-05-F2 fix.

    ``_zero_gradients``, ``_clip_and_log_gradients`` and
    ``_backward_and_step`` belong to ``BaseTrainingStrategy``. Defining
    them on the mixin advertised an MRO contribution that never
    fired.
    """
    own_methods = set(vars(OptimizerMixin))
    forbidden = {"_zero_gradients", "_clip_and_log_gradients", "_backward_and_step"}
    leaks = own_methods & forbidden
    assert not leaks, (
        f"OptimizerMixin re-introduced shadowed methods {sorted(leaks)}. "
        "These live on BaseTrainingStrategy; defining them here recreates the "
        "audit-05-F2 silent-MRO-shadowing bug."
    )


def test_optimizer_mixin_still_builds_stepper() -> None:
    """The one live method on the mixin must keep working."""
    assert callable(getattr(OptimizerMixin, "_build_optimizer_stepper", None)), (
        "OptimizerMixin._build_optimizer_stepper is the only live method "
        "BaseTrainingStrategy delegates to (base.py:319). Removing it would "
        "break strategy construction."
    )


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def _spy_optimizer(model: nn.Module) -> tuple[torch.optim.Optimizer, MagicMock]:
    """Wrap a real SGD optimizer with a zero_grad spy that still runs."""
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    real_zero_grad = opt.zero_grad
    spy = MagicMock(side_effect=real_zero_grad)
    opt.zero_grad = spy  # type: ignore[method-assign]
    return opt, spy


def _make_loss(model: nn.Module) -> torch.Tensor:
    x = torch.randn(4, 2)
    return model(x).pow(2).mean()


def test_backward_and_step_zero_grad_set_to_none_fp32_path() -> None:
    """TE-001: fp32 path must call zero_grad(set_to_none=True) (NN#9)."""
    model = _TinyModel()
    opt, spy = _spy_optimizer(model)
    loss = _make_loss(model)

    policy = AMPPolicy(enable_gradient_clipping=False)
    policy.backward_and_step(
        loss=loss,
        optimizer=opt,
        model=model,
        model_name="m",
        epoch=0,
        model_type="reconstruction",
        scaler=None,
        perform_step=True,
    )

    spy.assert_called_once_with(set_to_none=True)


def test_backward_and_step_zero_grad_set_to_none_amp_path() -> None:
    """TE-001: AMP path must call zero_grad(set_to_none=True) (NN#9)."""
    model = _TinyModel()
    opt, spy = _spy_optimizer(model)
    loss = _make_loss(model)

    # Minimal scaler stub: scale() returns the tensor, the rest are no-ops.
    scaler = MagicMock()
    scaler.scale.side_effect = lambda t: t
    # No recorded inf checks -> the policy takes the "clear stuck state" branch
    # and still reaches optimizer.zero_grad(set_to_none=True).
    scaler._per_optimizer_states = {}

    policy = AMPPolicy(enable_gradient_clipping=False)
    policy.backward_and_step(
        loss=loss,
        optimizer=opt,
        model=model,
        model_name="m",
        epoch=0,
        model_type="reconstruction",
        scaler=scaler,
        perform_step=True,
    )

    spy.assert_called_once_with(set_to_none=True)
