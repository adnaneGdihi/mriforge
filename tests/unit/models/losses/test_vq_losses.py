"""Tests for VQ-VAE codebook/commitment losses.

Targets ``mriforge.models.losses.vq_losses.VQLoss`` (simple API).

The regression guarded here is the stop-gradient placement (van den Oord
et al. 2017). The *codebook* term ``‖sg[z_e] − e‖²`` must update the codebook
``e`` (stop-grad on the encoder output), and the *commitment* term
``β‖z_e − sg[e]‖²`` must update the encoder (stop-grad on the codebook). A
swapped ``detach`` routes ``codebook_weight``/``commitment_weight`` onto the
wrong tensors, so the codebook learns at the commitment rate (≈4× too slow).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mriforge.models.losses.vq_losses import VQLoss  # noqa: E402


def _pair() -> tuple[torch.Tensor, torch.Tensor]:
    quantized = torch.randn(2, 4, 8, 8, requires_grad=True)  # codebook vectors e
    targets = torch.randn(2, 4, 8, 8, requires_grad=True)  # encoder output z_e
    return quantized, targets


def _grad_mag(t: torch.Tensor) -> float:
    """Gradient magnitude (0 if the leaf received no gradient)."""
    return 0.0 if t.grad is None else float(t.grad.abs().sum())


def test_codebook_term_updates_codebook_only() -> None:
    """With only the codebook term active, gradient flows to ``quantized`` (e),
    not to the encoder output ``targets`` (which carries the stop-gradient).

    The zeroed commitment term keeps ``targets`` in the graph, so its gradient
    is a *zero tensor* — the decisive check is magnitude, not ``is None``.
    """
    quantized, targets = _pair()
    loss_fn = VQLoss(codebook_weight=1.0, commitment_weight=0.0)
    total, _ = loss_fn(quantized, targets)
    total.backward()

    assert _grad_mag(quantized) > 0
    assert _grad_mag(targets) == pytest.approx(0.0, abs=1e-12)


def test_commitment_term_updates_encoder_only() -> None:
    """With only the commitment term active, gradient flows to the encoder
    output ``targets`` (z_e), not to the codebook ``quantized``."""
    quantized, targets = _pair()
    loss_fn = VQLoss(codebook_weight=0.0, commitment_weight=1.0)
    total, _ = loss_fn(quantized, targets)
    total.backward()

    assert _grad_mag(targets) > 0
    assert _grad_mag(quantized) == pytest.approx(0.0, abs=1e-12)


def test_vq_registered() -> None:
    """``vq`` is resolvable from the loss registry."""
    from mriforge.models.losses.registry import list_available

    assert "vq" in list_available()


def test_vq_reconstruction_loss_is_registry_routed() -> None:
    """WS-6 plan #6: the loss_type dispatch matches the replaced functionals
    and raises (never defaults) on unknown names."""
    import torch.nn.functional as F

    from mriforge.models.losses.vq_losses import vq_reconstruction_loss

    torch.manual_seed(0)
    recon = torch.randn(2, 1, 8, 8)
    target = torch.randn(2, 1, 8, 8)

    assert torch.allclose(vq_reconstruction_loss(recon, target, "l1"), F.l1_loss(recon, target))
    assert torch.allclose(vq_reconstruction_loss(recon, target, "l2"), F.mse_loss(recon, target))
    assert torch.allclose(
        vq_reconstruction_loss(recon, target, "huber"), F.smooth_l1_loss(recon, target)
    )
    with pytest.raises(ValueError, match="Unsupported loss type"):
        vq_reconstruction_loss(recon, target, "nope")
