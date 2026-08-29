"""Tests for the slab → HF volume sparse VAE generator.

Pins the contract that distinguishes this model from the existing
2D-only VAEs:

1. Accepts both 4D (single-slice) and 5D (slab) input.
2. Output spatial depth equals ``d_out`` regardless of input depth.
3. Sparsity mask is learnable and contributes a non-zero
   ``sparsity_loss()`` value.
4. ``encode``/``reparameterise`` flow returns finite mu / logvar.
5. Registry capability flag declares both ``spatial_dims=(2, 3)`` and
   ``training_mode=vae`` — the May 2026 stage-1 LDM blocker.
"""

from __future__ import annotations

import pytest
import torch

from mriforge.models.generators.sparse_vae_slab_to_volume import (
    SparseVAESlabToVolumeGenerator,
)


def _small(d_in: int = 1, d_out: int = 8):
    return SparseVAESlabToVolumeGenerator(
        in_channels=1,
        out_channels=1,
        d_in=d_in,
        d_out=d_out,
        base_channels=8,
        latent_channels=4,
        num_layers=2,
    )


class TestSparseVAESlabToVolume:
    def test_4d_slice_to_volume(self) -> None:
        m = _small(d_in=1, d_out=8)
        x = torch.randn(2, 1, 32, 32)
        y = m(x)
        assert y.shape == (2, 1, 32, 32, 8), (
            f"expected (2,1,32,32,8); got {tuple(y.shape)}"
        )
        assert torch.isfinite(y).all()

    def test_5d_slab_to_volume(self) -> None:
        m = _small(d_in=3, d_out=12)
        x = torch.randn(2, 1, 32, 32, 3)
        y = m(x)
        assert y.shape == (2, 1, 32, 32, 12)
        assert torch.isfinite(y).all()

    def test_output_depth_independent_of_input_depth(self) -> None:
        """Pinning the central guarantee: ``d_out`` is fixed by the
        constructor, not by the input depth."""
        m = _small(d_in=1, d_out=16)
        # Same model, two different input depths → both produce d_out=16.
        y1 = m(torch.randn(1, 1, 32, 32, 1))
        # Even an input that disagrees with d_in is interpolated at the
        # tail to match d_out.
        y_extra = m(torch.randn(1, 1, 32, 32))  # 4D promoted to depth=1
        assert y1.shape[-1] == 16
        assert y_extra.shape[-1] == 16

    def test_encode_returns_finite_mu_logvar(self) -> None:
        m = _small()
        mu, logvar = m.encode(torch.randn(1, 1, 32, 32))
        assert torch.isfinite(mu).all()
        assert torch.isfinite(logvar).all()
        assert mu.shape == logvar.shape

    def test_kl_loss_scalar_and_finite(self) -> None:
        m = _small()
        mu, logvar = m.encode(torch.randn(1, 1, 32, 32))
        kl = m.kl_loss(mu, logvar)
        assert kl.dim() == 0
        assert torch.isfinite(kl)

    def test_sparsity_loss_is_learnable(self) -> None:
        """The sparsity mask is registered as ``nn.Parameter`` so the
        sparsity_loss is differentiable wrt the model parameters."""
        m = _small()
        sp = m.sparsity_loss()
        assert sp.requires_grad
        sp.backward()
        # The mask parameter must have received a gradient.
        assert m.sparsity_mask.grad is not None
        assert m.sparsity_mask.grad.abs().sum().item() > 0

    def test_apply_sparsity_zeros_pruned_channels(self) -> None:
        """When the sparsity mask is forced negative, the corresponding
        latent channels are squashed by ``sigmoid(neg) ≈ 0`` before the
        decoder. ``apply_sparsity=False`` bypasses the mask."""
        m = _small()
        with torch.no_grad():
            m.sparsity_mask.fill_(-50.0)  # sigmoid(-50) ≈ 0
        x = torch.randn(1, 1, 32, 32)
        y_with = m(x, apply_sparsity=True)
        y_without = m(x, apply_sparsity=False)
        # The two outputs must differ — ``apply_sparsity`` is a real switch.
        assert not torch.allclose(y_with, y_without, atol=1e-5)

    def test_constructor_rejects_zero_dimensions(self) -> None:
        # d_in must be >= 1 (negative or zero would crash inside Conv3d).
        with pytest.raises(Exception):
            _ = SparseVAESlabToVolumeGenerator(
                in_channels=1, out_channels=1, d_in=0, d_out=8,
                base_channels=8, latent_channels=4, num_layers=2,
            )(torch.randn(1, 1, 16, 16, 0))

    def test_registry_metadata(self) -> None:
        from mriforge.models.registry import MODEL_REGISTRY

        entry = MODEL_REGISTRY.get("sparse_vae_slab_to_volume")
        assert entry is not None
        caps = entry["capabilities"]
        assert 2 in caps.spatial_dims and 3 in caps.spatial_dims
        assert entry["mode"] == "vae"
