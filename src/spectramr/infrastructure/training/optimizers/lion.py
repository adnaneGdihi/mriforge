"""Lion — EvoLved Sign Momentum (Chen et al., 2023).

    update = sign(beta1 * m + (1 - beta1) * g)
    m      = beta2 * m + (1 - beta2) * g

Two properties follow from the sign, and both matter operationally:

* **One momentum buffer instead of two.** Lion keeps no second moment, so its
  optimizer state is half AdamW's — the reason to reach for it on a memory-bound
  arm.
* **Every parameter moves by exactly ``lr``.** The update is a sign, so the step
  size is uniform and scale-free. That is why Lion's useful learning rate is
  **3-10x smaller** than AdamW's, with weight decay correspondingly larger; an
  AdamW-scale LR here diverges rather than merely training badly. A config check
  warns on that pairing rather than leaving it to be discovered from a NaN.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer

from spectramr.infrastructure.training.optimizer_registry import register_optimizer

__all__ = ["Lion"]


@register_optimizer("lion")
class Lion(Optimizer):
    """Sign-momentum optimizer with decoupled weight decay.

    Args:
        params: Parameters or param groups.
        lr: Learning rate. Expect 3-10x smaller than an AdamW LR.
        betas: ``(beta1, beta2)`` — beta1 interpolates for the *update*, beta2 for
            the *buffer*. Defaults ``(0.9, 0.99)`` are the paper's, and differ
            from Adam's ``(0.9, 0.999)``.
        weight_decay: Decoupled (AdamW-style). Expect it larger than an AdamW
            value, since the update magnitude no longer scales with the gradient.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        super().__init__(params, {"lr": lr, "betas": betas, "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")

                state = self.state[p]
                if not state:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]

                # Decoupled decay first, so the sign below is taken on the
                # gradient signal alone rather than on signal + decay.
                if wd != 0:
                    p.mul_(1 - lr * wd)

                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1).sign_()
                p.add_(update, alpha=-lr)

                # Buffer update uses beta2 and the CURRENT gradient — note this
                # happens after the step, and with a different beta than the
                # update above. Collapsing the two betas is the usual way this
                # optimizer gets reimplemented wrongly.
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss
