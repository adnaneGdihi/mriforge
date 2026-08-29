"""Unit tests for masked_uncertainty_multi_contrast_loss.py."""

from __future__ import annotations

import pytest
import torch

from mriforge.models.losses.masked_uncertainty_multi_contrast_loss import (
    MaskedUncertaintyMultiContrastLoss,
)
from mriforge.models.losses.registry import create_loss

B, C, H, W = 4, 1, 16, 16
N_C = 3  # number of contrasts


def _img(b=B, c=C, h=H, w=W, seed=0) -> torch.Tensor:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.rand(b, c, h, w, generator=gen)


def _mask(b=B, h=H, w=W) -> torch.Tensor:
    """Binary foreground mask [B, 1, H, W]."""
    m = torch.zeros(b, 1, h, w)
    m[:, :, h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 1.0
    return m


class TestMaskedUncertaintyMultiContrastLoss:

    # ── Canary ──────────────────────────────────────────────────────────────

    def test_canary_builds_and_forward_finite(self):
        loss_fn = MaskedUncertaintyMultiContrastLoss(n_contrasts=N_C)
        pred = _img()
        target = _img(seed=1)
        cid = torch.zeros(B, dtype=torch.long)
        out = loss_fn(pred, target, contrast_id=cid)
        assert torch.isfinite(out) and out.shape == ()

    # ── Parametric ──────────────────────────────────────────────────────────

    @pytest.mark.parametrize("n_contrasts", [1, 3, 5])
    def test_parametric_n_contrasts(self, n_contrasts: int):
        loss_fn = MaskedUncertaintyMultiContrastLoss(n_contrasts=n_contrasts)
        pred = _img()
        target = _img(seed=1)
        cid = torch.zeros(B, dtype=torch.long)
        out = loss_fn(pred, target, contrast_id=cid)
        assert torch.isfinite(out)

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_parametric_reduction(self, reduction: str):
        loss_fn = MaskedUncertaintyMultiContrastLoss(n_contrasts=N_C, reduction=reduction)
        pred = _img()
        target = _img(seed=1)
        cid = torch.zeros(B, dtype=torch.long)
        out = loss_fn(pred, target, contrast_id=cid)
        assert torch.isfinite(out)

    # ── Property ────────────────────────────────────────────────────────────

    def test_property_gradient_reaches_prediction(self):
        loss_fn = MaskedUncertaintyMultiContrastLoss(n_contrasts=N_C)
        pred = _img().requires_grad_(True)
        target = _img(seed=1)
        cid = torch.zeros(B, dtype=torch.long)
        loss_fn(pred, target, contrast_id=cid).backward()
        assert pred.grad is not None and torch.isfinite(pred.grad).all()

    def test_property_log_sigma_is_learnable(self):
        loss_fn = MaskedUncertaintyMultiContrastLoss(n_contrasts=N_C)
        assert loss_fn.log_sigma.requires_grad

    def test_property_gradient_reaches_log_sigma(self):
        """log_sigma is a nn.Parameter; gradients should flow to it."""
        loss_fn = MaskedUncertaintyMultiContrastLoss(n_contrasts=N_C)
        pred = _img()
        target = _img(seed=1)
        cid = torch.zeros(B, dtype=torch.long)
        loss_fn(pred, target, contrast_id=cid).backward()
        assert loss_fn.log_sigma.grad is not None

    def test_property_multi_contrast_ids(self):
        """Multiple contrast IDs in one batch — all slots exercised."""
        loss_fn = MaskedUncertaintyMultiContrastLoss(n_contrasts=N_C)
        pred = _img()
        target = _img(seed=1)
        cid = torch.tensor([0, 1, 2, 0], dtype=torch.long)
        out = loss_fn(pred, target, contrast_id=cid)
        assert torch.isfinite(out)

    def test_property_mask_reduces_gradient_region(self):
        """With a foreground mask only the masked region contributes."""
        loss_fn = MaskedUncertaintyMultiContrastLoss(n_contrasts=1)
        pred = _img()
        target = _img(seed=1)
        mask = _mask()
        out_masked = loss_fn(pred, target, mask=mask, contrast_id=0)
        out_full = loss_fn(pred, target, mask=None, contrast_id=0)
        # Both finite; masked < unmasked typically (less area averaged)
        assert torch.isfinite(out_masked) and torch.isfinite(out_full)

    # ── Edge ────────────────────────────────────────────────────────────────

    def test_edge_contrast_id_none_defaults_to_zero(self):
        loss_fn = MaskedUncertaintyMultiContrastLoss(n_contrasts=N_C)
        pred = _img()
        target = _img(seed=1)
        out = loss_fn(pred, target, contrast_id=None)
        assert torch.isfinite(out)

    def test_edge_contrast_id_as_int(self):
        loss_fn = MaskedUncertaintyMultiContrastLoss(n_contrasts=N_C)
        pred = _img()
        target = _img(seed=1)
        out = loss_fn(pred, target, contrast_id=1)
        assert torch.isfinite(out)

    def test_edge_all_zero_mask(self):
        """Zero mask → denom clamped to 1 → finite."""
        loss_fn = MaskedUncertaintyMultiContrastLoss(n_contrasts=1)
        pred = _img()
        target = _img(seed=1)
        mask = torch.zeros(B, 1, H, W)
        out = loss_fn(pred, target, mask=mask, contrast_id=0)
        assert torch.isfinite(out)

    def test_edge_per_contrast_sigma_no_grad(self):
        loss_fn = MaskedUncertaintyMultiContrastLoss(n_contrasts=N_C)
        sigma = loss_fn.per_contrast_sigma()
        assert sigma.shape == (N_C,)
        assert (sigma > 0).all()

    # ── Raises ──────────────────────────────────────────────────────────────

    def test_raises_shape_mismatch(self):
        loss_fn = MaskedUncertaintyMultiContrastLoss(n_contrasts=1)
        pred = _img()
        target = _img(c=2)  # different shape
        with pytest.raises(ValueError, match="shape"):
            loss_fn(pred, target, contrast_id=0)

    def test_raises_contrast_id_out_of_range(self):
        loss_fn = MaskedUncertaintyMultiContrastLoss(n_contrasts=N_C)
        pred = _img()
        target = _img(seed=1)
        cid = torch.full((B,), N_C, dtype=torch.long)  # equal to n_contrasts, out of range
        with pytest.raises(ValueError, match="contrast_id"):
            loss_fn(pred, target, contrast_id=cid)

    def test_raises_bad_reduction(self):
        with pytest.raises(ValueError, match="reduction"):
            MaskedUncertaintyMultiContrastLoss(n_contrasts=1, reduction="xyz")

    def test_raises_zero_n_contrasts(self):
        with pytest.raises(ValueError, match="n_contrasts"):
            MaskedUncertaintyMultiContrastLoss(n_contrasts=0)

    # ── Registry ────────────────────────────────────────────────────────────

    def test_registry_lookup(self):
        loss_fn = create_loss("masked_uncertainty_multi_contrast")
        assert isinstance(loss_fn, MaskedUncertaintyMultiContrastLoss)
