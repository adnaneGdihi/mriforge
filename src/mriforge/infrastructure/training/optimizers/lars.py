"""LARS — Layer-wise Adaptive Rate Scaling (You, Gitman & Ginsburg, 2017).

SGD with a per-tensor *trust ratio* that rescales each layer's update by
``eta * ||w|| / (||g|| + wd*||w|| + eps)``. The point is large-batch training:
without it, a single global learning rate is simultaneously too large for layers
with small weights and too small for layers with large ones.

``lars`` has been a member of the ``OptimizerType`` enum since it was introduced,
with **no implementation anywhere** — ``test_optimizer_type_coverage`` xfailed it
as "needs apex / timm". This is that implementation, dependency-free.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer

from mriforge.infrastructure.training.optimizer_registry import register_optimizer

__all__ = ["LARS"]

#: Denominator floor for the on-device trust ratio. ``torch.where`` evaluates
#: both branches, so the division must stay finite even where its result is
#: discarded -- a nan there survives the selection.
_TINY = 1e-38


@register_optimizer("lars")
class LARS(Optimizer):
    """SGD with a layer-wise trust ratio.

    Args:
        params: Parameters or param groups.
        lr: Base learning rate.
        momentum: Heavy-ball momentum applied to the *scaled* update.
        weight_decay: Decoupled from the trust-ratio denominator's own term only
            in the sense that it is added to the gradient before scaling, which
            is what the paper specifies.
        eta: Trust coefficient (the paper's ``eta``, typically 1e-3).
        epsilon: Guards the trust ratio when a layer's gradient is exactly zero.
        exclude_1d: Skip the trust ratio for 1-D parameters (biases and norm
            weights). This is the standard LARS/LARC convention and it matters:
            applying a norm-ratio to a bias vector produces wildly-scaled steps.
            Exposed as a real knob rather than hardcoded, because "which tensors
            are excluded" is a genuine experimental axis.
        dampening: Momentum dampening.
        nesterov: Use Nesterov momentum.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        eta: float = 1e-3,
        epsilon: float = 1e-8,
        exclude_1d: bool = True,
        dampening: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if eta <= 0.0:
            raise ValueError(f"Invalid trust coefficient eta: {eta}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires momentum > 0 and dampening = 0")

        super().__init__(
            params,
            {
                "lr": lr,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "eta": eta,
                "epsilon": epsilon,
                "exclude_1d": exclude_1d,
                "dampening": dampening,
                "nesterov": nesterov,
            },
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                excluded = group["exclude_1d"] and p.ndim <= 1
                if excluded:
                    # Plain SGD for biases / norm parameters.
                    d_p = grad.add(p, alpha=weight_decay) if weight_decay else grad
                    trust = 1.0
                else:
                    d_p = grad.add(p, alpha=weight_decay) if weight_decay else grad
                    w_norm = torch.norm(p)
                    g_norm = torch.norm(grad)
                    # A zero-norm layer has no direction to scale; leaving the
                    # ratio at 1 keeps it on plain SGD instead of producing
                    # 0/0 -> nan, which is what an unguarded division does on the
                    # first step of a zero-initialised layer.
                    #
                    # The guard stays ON DEVICE. `if w_norm == 0 or g_norm == 0`
                    # converts two tensors to Python bools, i.e. two host
                    # round-trips per parameter tensor per step inside the
                    # training loop (non-negotiable #9) -- measured at 9.4x an
                    # AdamW step on CUDA over 80 parameter tensors.
                    #
                    # clamp_min keeps torch.where's discarded branch finite:
                    # where evaluates BOTH sides, so a nan from 0/0 would
                    # survive the selection.
                    denom = g_norm + weight_decay * w_norm + group["epsilon"]
                    ratio = group["eta"] * w_norm / denom.clamp_min(_TINY)
                    trust = torch.where((w_norm > 0) & (g_norm > 0), ratio, torch.ones_like(ratio))

                update = d_p * trust

                if momentum != 0:
                    state = self.state[p]
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = torch.clone(update).detach()
                        state["momentum_buffer"] = buf
                    else:
                        buf.mul_(momentum).add_(update, alpha=1 - dampening)
                    update = update.add(buf, alpha=momentum) if nesterov else buf

                p.add_(update, alpha=-group["lr"])

        return loss
