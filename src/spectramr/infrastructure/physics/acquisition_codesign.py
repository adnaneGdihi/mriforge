r"""End-to-end acquisition co-design (A-7.1 *E2E-AcqCo*).

The sampling pattern and the reconstruction are usually designed separately. E2E
acquisition co-design optimises them **jointly against the same reconstruction
loss**, so the budget of acquired lines migrates to wherever — given the recon —
it most reduces error. This is the LOUPE idea (learned under-sampling) made
self-contained over a phase-encode-line sampling problem.

A :class:`LearnableSamplingMask` holds per-line logits; the soft mask is
:math:`\sigma(\text{logits}/\tau)` and a budget penalty
:math:`(\sum_l m_l - B)^2` ties the expected number of acquired lines to the
budget. :func:`codesign_sampling` descends the reconstruction MSE in the soft mask
**and** a per-line recon gain simultaneously, then hardens the soft mask to the
top-:math:`B` lines. Because both halves see the same loss, the learned pattern
adapts to the data + recon — and (the oracle) beats a matched-budget *random* mask,
which a fixed pattern cannot.

A self-contained design primitive (small synthetic problems for its oracles), not
a production training paradigm; it reuses the ``fft``/``ifft`` SSOT for the
line-sampling forward.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class LearnableSamplingMask(nn.Module):
    r"""Per-line learnable sampling logits with a soft, budget-tied mask (A-7.1).

    Args:
        n_lines: number of phase-encode lines (columns).
        budget: target number of acquired lines :math:`B`.
        temperature: sigmoid temperature :math:`\tau` (smaller ⇒ sharper).
    """

    def __init__(self, n_lines: int, budget: int, temperature: float = 0.5) -> None:
        super().__init__()
        if not 0 < budget <= n_lines:
            raise ValueError(f"budget must be in (0, n_lines]; got {budget}/{n_lines}")
        self.n_lines = int(n_lines)
        self.budget = int(budget)
        self.temperature = float(temperature)
        self.logits = nn.Parameter(torch.zeros(n_lines))

    def soft_mask(self) -> Tensor:
        """Continuous mask :math:`\\sigma(\\text{logits}/\\tau)\\in(0,1)^{n}`."""
        return torch.sigmoid(self.logits / self.temperature)

    def budget_penalty(self) -> Tensor:
        """:math:`(\\sum_l m_l - B)^2` — ties the expected acquired-line count to the budget."""
        return (self.soft_mask().sum() - self.budget) ** 2

    def hard_mask(self) -> Tensor:
        """Binary mask of the top-``budget`` lines (the deployed pattern)."""
        idx = torch.topk(self.logits.detach(), self.budget).indices
        m = torch.zeros(self.n_lines, device=self.logits.device)
        m[idx] = 1.0
        return m

    def straight_through_mask(self) -> Tensor:
        """Top-``budget`` binary mask in the forward, soft-mask gradient in the backward.

        Evaluates the recon at the ACTUAL budget (the deployed pattern) while still
        differentiable — so the loss optimises the deployed reconstruction, not a
        budget-violating soft relaxation (the LOUPE soft/hard mismatch).
        """
        soft = self.soft_mask()
        hard = self.hard_mask()
        return hard + soft - soft.detach()


def _line_recon(images: Tensor, mask: Tensor, gain: Tensor) -> Tensor:
    """Recon from column-sampled k-space: ``ifft(mask*gain * fft(x))`` (real part)."""
    x_k = torch.fft.fft(images.to(torch.complex64), dim=-1)
    y = x_k * (mask * gain).to(x_k.dtype)[None, :]
    return torch.fft.ifft(y, dim=-1).real


def codesign_sampling(
    images: Tensor, budget: int, *, n_steps: int = 200, lr: float = 0.1, lam_budget: float = 1.0
) -> dict[str, Any]:
    r"""Co-optimise a sampling mask + a per-line recon gain end-to-end on ``images``.

    Args:
        images: ``[N, W]`` training rows (1-D phase-encode sampling along ``W``).
        budget: number of lines to acquire.
        n_steps / lr / lam_budget: optimisation knobs.

    Returns:
        ``{mask, gain, mse_history, learned_mse, random_mse}`` — ``learned_mse`` is
        the recon MSE of the hardened learned mask, ``random_mse`` that of a
        matched-budget random mask (the baseline the co-design must beat).
    """
    w = images.shape[1]
    mask_module = LearnableSamplingMask(w, budget)
    gain = nn.Parameter(torch.ones(w))
    opt = torch.optim.Adam([mask_module.logits, gain], lr=lr)
    mse = nn.functional.mse_loss
    history: list[float] = []
    for _ in range(n_steps):
        opt.zero_grad(set_to_none=True)
        # Straight-through: the recon is evaluated at the actual (hard) budget, so the
        # loss optimises the DEPLOYED reconstruction (not a budget-violating soft mask).
        recon = _line_recon(images, mask_module.straight_through_mask(), gain)
        loss = mse(recon, images) + lam_budget * mask_module.budget_penalty()
        loss.backward()
        opt.step()
        history.append(float(mse(recon, images).detach()))

    with torch.no_grad():
        hard = mask_module.hard_mask()
        learned_mse = float(mse(_line_recon(images, hard, gain), images))
        # Matched-budget random baseline (deterministic seed for reproducibility).
        g = torch.Generator(device=images.device).manual_seed(0)
        rand_idx = torch.randperm(w, generator=g)[:budget]
        rand = torch.zeros(w, device=images.device)
        rand[rand_idx] = 1.0
        random_mse = float(mse(_line_recon(images, rand, gain), images))

    return {
        "mask": hard,
        "gain": gain.detach(),
        "mse_history": history,
        "learned_mse": learned_mse,
        "random_mse": random_mse,
    }


__all__ = ["LearnableSamplingMask", "codesign_sampling"]
