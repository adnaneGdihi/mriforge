"""Gradient-flow / convergence guard for VF reconstruction generators.

Motivated by a 2026-05-26 investigation into "the VF arms don't seem to
converge". The hypothesis was a disconnected autograd graph. The finding
(this test pins it):

1. **The trainable backbones ARE connected and DO learn.** A real
   forward→backward→step loop reduces an overfit L1 loss and gradient
   reaches the bulk of the parameters. The apparent non-convergence was
   NOT a broken graph.
2. **Zero-init slow-start.** Both ``output_proj`` and each
   ``_SmallUNet.head`` are ``_zero_init_conv``-initialised, so at *step 0*
   the only parameter with gradient is the final projection
   (``d(pred)/d(x_cur) = output_proj.weight = 0`` blocks the cascade). It
   unblocks within ~2 optimiser steps — hence we assert grad-coverage
   *after* a couple of steps, not at init.
3. **Several VF arms are parameter-free by design** (closed-form
   algebraic / physics-identity arms). Their loss curve is essentially
   flat because there is almost nothing to train — that is the
   experiment, not a regression. They are pinned here so a future reader
   does not mistake their flat loss for a bug.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

import mriforge.models  # noqa: F401, E402  triggers @register_model
from mriforge.models.registry import get_model_class  # noqa: E402
from tests.utils.optional_backends import requires_cuda_for_mamba  # noqa: E402

# Substantive trainable backbones used by VF arms (>20k params).
#
# ``hyper_mamba_unet`` is the only one carrying a Mamba backbone, so it alone
# needs the CUDA kernel; marking the whole test would take four unrelated arms
# down with it. The other four must stay CPU-live.
_TRAINABLE_BACKBONES = [
    "modl_sr",
    "wiener_unet",
    "hard_dc_forcing",
    "kalman_pll",
    pytest.param("hyper_mamba_unet", marks=requires_cuda_for_mamba),
]

# Closed-form / algebraic arms: ~0 trainable params *by design*. A flat
# loss curve for these is expected, not a convergence failure.
_PARAMETER_FREE_ARMS = [
    "reciprocity_divisor",
    "dam_trigfit",
    "bloch_siegert_algebraic",
]


@pytest.mark.parametrize("name", _TRAINABLE_BACKBONES)
def test_vf_backbone_graph_is_connected_and_learns(name: str) -> None:
    """Forward→backward→step must reach most params and reduce the loss."""
    torch.manual_seed(0)
    gen = get_model_class(name)(in_channels=2, out_channels=2)
    gen.train()

    x = torch.randn(1, 2, 32, 32)
    y = torch.randn(1, 2, 32, 32)
    opt = torch.optim.Adam(gen.parameters(), lr=1e-3)

    # 60 steps: modl_sr (5 unrolled cascades + identity-DC + zero-init
    # transient) is markedly slower than the plain UNets, so the budget is
    # sized for the slowest backbone. The threshold below still cleanly
    # separates "learns" (≥10% drop) from "disconnected" (~0% drop).
    losses = []
    for _ in range(60):
        opt.zero_grad()
        pred = gen(x)
        loss = F.l1_loss(pred[:, : y.shape[1]], y)
        loss.backward()
        losses.append(loss.item())
        opt.step()

    # Connectivity: after the zero-init transient, gradient must reach a
    # substantial fraction of parameters (catches a real disconnection,
    # tolerates the documented mamba-fallback / unused-conditioning dead
    # branches).
    with_grad = sum(
        1
        for p in gen.parameters()
        if p.grad is not None and p.grad.norm().item() > 1e-12
    )
    total = sum(1 for _ in gen.parameters())
    assert with_grad / total >= 0.4, (
        f"{name}: only {with_grad}/{total} params received gradient — the "
        f"autograd graph is (largely) disconnected from the loss."
    )

    # Convergence: the overfit loss must drop meaningfully. A disconnected
    # graph shows ~0% drop; even the slow modl_sr clears 10% here.
    assert losses[-1] < 0.90 * losses[0], (
        f"{name}: overfit loss did not decrease ({losses[0]:.4f} -> "
        f"{losses[-1]:.4f}); a connected graph should learn a fixed sample."
    )


def test_zero_init_blocks_cascade_at_step0_then_unblocks() -> None:
    """Pin the zero-init slow-start so the behaviour is documented, not feared.

    At step 0 only ``output_proj`` has gradient (the cascade is blocked by
    its zero weights); after one optimiser step the cascade wakes up.
    """
    torch.manual_seed(0)
    gen = get_model_class("modl_sr")(in_channels=2, out_channels=2)
    gen.train()
    x = torch.randn(1, 2, 32, 32)
    opt = torch.optim.Adam(gen.parameters(), lr=1e-3)

    def cascade_alive() -> int:
        return sum(
            1
            for n, p in gen.named_parameters()
            if "regularisers" in n
            and p.grad is not None
            and p.grad.norm().item() > 1e-12
        )

    # Step 0 — cascade blocked by the zero-init output projection.
    opt.zero_grad()
    F.l1_loss(gen(x), torch.randn_like(gen(x))).backward()
    assert cascade_alive() == 0, "expected the cascade to be grad-blocked at init"
    opt.step()

    # Step 2 — cascade fully unblocked.
    for _ in range(2):
        opt.zero_grad()
        F.l1_loss(gen(x), torch.randn_like(gen(x))).backward()
        opt.step()
    assert cascade_alive() > 0, "cascade must receive gradient after the transient"


@pytest.mark.parametrize("name", _PARAMETER_FREE_ARMS)
def test_parameter_free_arms_are_intentionally_untrainable(name: str) -> None:
    """These arms have <2k trainable params by design — flat loss is expected."""
    gen = get_model_class(name)(in_channels=2, out_channels=2)
    n_train = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    assert n_train < 2000, (
        f"{name} now has {n_train} trainable params; if it became a learned "
        f"arm, move it to _TRAINABLE_BACKBONES and assert convergence."
    )
