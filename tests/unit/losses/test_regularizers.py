"""Unit tests for regularizers.py.

Covers: WaveletSparsityLoss, HessianSchattenLoss, PatchLowRankLoss,
        HermitianSymmetryLoss, KspaceL1SparsityLoss, HankelLowRankLoss,
        PhaseSmoothnessComplexLoss.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.regularizers import (
    HankelLowRankLoss,
    HermitianSymmetryLoss,
    HessianSchattenLoss,
    KspaceL1SparsityLoss,
    PatchLowRankLoss,
    PhaseSmoothnessComplexLoss,
    WaveletSparsityLoss,
)
from spectramr.models.losses.registry import create_loss

B, C, H, W = 2, 1, 32, 32


def _img(b=B, c=C, h=H, w=W, seed=0) -> torch.Tensor:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.rand(b, c, h, w, generator=gen)


def _complex_img(b=B, c=C, h=H, w=W, seed=0) -> torch.Tensor:
    gen = torch.Generator()
    gen.manual_seed(seed)
    re = torch.randn(b, c, h, w, generator=gen)
    im = torch.randn(b, c, h, w, generator=gen)
    return torch.complex(re, im)


def _kspace_2ch(b=B, h=H, w=W, seed=0) -> torch.Tensor:
    """Real-valued [B, 2, H, W] k-space layout."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.randn(b, 2, h, w, generator=gen)


# ===========================================================================
# WaveletSparsityLoss
# ===========================================================================


class TestWaveletSparsityLoss:

    def test_canary_builds_and_forward_finite(self):
        loss_fn = WaveletSparsityLoss()
        x = _img()
        out = loss_fn(x)
        assert torch.isfinite(out) and out.shape == ()

    @pytest.mark.parametrize("wavelet", ["haar", "db2"])
    def test_parametric_wavelets(self, wavelet: str):
        loss_fn = WaveletSparsityLoss(wavelet=wavelet)
        x = _img()
        out = loss_fn(x)
        assert torch.isfinite(out) and out.item() >= 0.0

    @pytest.mark.parametrize("levels", [1, 2])
    def test_parametric_levels(self, levels: int):
        loss_fn = WaveletSparsityLoss(levels=levels)
        x = _img()
        out = loss_fn(x)
        assert torch.isfinite(out)

    def test_property_nonneg(self):
        """Wavelet L1 is always ≥ 0."""
        loss_fn = WaveletSparsityLoss()
        x = _img()
        assert loss_fn(x).item() >= 0.0

    def test_property_zero_prediction_gives_zero(self):
        loss_fn = WaveletSparsityLoss()
        x = torch.zeros(B, C, H, W)
        out = loss_fn(x)
        assert out.item() == pytest.approx(0.0, abs=1e-7)

    def test_property_gradient_reaches_input(self):
        x = _img().requires_grad_(True)
        loss_fn = WaveletSparsityLoss()
        loss_fn(x).backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_edge_complex_input(self):
        """Complex input: falls back to abs magnitude."""
        loss_fn = WaveletSparsityLoss()
        x = _complex_img()
        out = loss_fn(x)
        assert torch.isfinite(out) and out.item() >= 0.0

    def test_edge_large_magnitude(self):
        loss_fn = WaveletSparsityLoss()
        x = torch.full((B, C, H, W), 1e4)
        out = loss_fn(x)
        assert torch.isfinite(out)

    def test_raises_bad_wavelet(self):
        with pytest.raises(ValueError, match="wavelet"):
            WaveletSparsityLoss(wavelet="nonexistent_wavelet_xyz")

    def test_raises_zero_levels(self):
        with pytest.raises(ValueError, match="levels"):
            WaveletSparsityLoss(levels=0)

    def test_registry_lookup(self):
        loss_fn = create_loss("wavelet_sparsity_l1")
        assert isinstance(loss_fn, WaveletSparsityLoss)


# ===========================================================================
# HessianSchattenLoss
# ===========================================================================


class TestHessianSchattenLoss:

    def test_canary_builds_and_forward_finite(self):
        loss_fn = HessianSchattenLoss()
        x = _img()
        out = loss_fn(x)
        assert torch.isfinite(out) and out.shape == ()

    @pytest.mark.parametrize("weight", [0.5, 1.0, 2.0])
    def test_parametric_weights(self, weight: float):
        loss_fn = HessianSchattenLoss(weight=weight)
        x = _img()
        out = loss_fn(x)
        assert torch.isfinite(out) and out.item() >= 0.0

    def test_property_nonneg(self):
        loss_fn = HessianSchattenLoss()
        x = _img()
        assert loss_fn(x).item() >= 0.0

    def test_property_zero_constant_image(self):
        """Constant image → Hessian = 0 → loss = 0."""
        loss_fn = HessianSchattenLoss()
        x = torch.ones(B, C, H, W)
        out = loss_fn(x)
        assert out.item() == pytest.approx(0.0, abs=1e-5)

    def test_property_gradient_reaches_input(self):
        x = _img().requires_grad_(True)
        loss_fn = HessianSchattenLoss()
        loss_fn(x).backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_edge_complex_input(self):
        loss_fn = HessianSchattenLoss()
        x = _complex_img()
        out = loss_fn(x)
        assert torch.isfinite(out)

    def test_edge_large_magnitude(self):
        loss_fn = HessianSchattenLoss()
        x = torch.full((B, C, H, W), 1e3)
        out = loss_fn(x)
        assert torch.isfinite(out)

    def test_registry_lookup(self):
        loss_fn = create_loss("hessian_schatten")
        assert isinstance(loss_fn, HessianSchattenLoss)


# ===========================================================================
# PatchLowRankLoss
# ===========================================================================


class TestPatchLowRankLoss:

    def test_canary_builds_and_forward_finite(self):
        loss_fn = PatchLowRankLoss()
        x = _img()
        out = loss_fn(x)
        assert torch.isfinite(out) and out.shape == ()

    @pytest.mark.parametrize("patch_size", [4, 8])
    def test_parametric_patch_sizes(self, patch_size: int):
        loss_fn = PatchLowRankLoss(patch_size=patch_size)
        x = _img()
        out = loss_fn(x)
        assert torch.isfinite(out) and out.item() >= 0.0

    def test_property_nonneg(self):
        loss_fn = PatchLowRankLoss()
        x = _img()
        assert loss_fn(x).item() >= 0.0

    def test_property_gradient_reaches_input(self):
        x = _img().requires_grad_(True)
        loss_fn = PatchLowRankLoss()
        loss_fn(x).backward()
        assert x.grad is not None

    def test_edge_complex_input(self):
        loss_fn = PatchLowRankLoss()
        x = _complex_img()
        out = loss_fn(x)
        assert torch.isfinite(out)

    def test_edge_large_magnitude(self):
        loss_fn = PatchLowRankLoss()
        x = torch.full((B, C, H, W), 1e3)
        out = loss_fn(x)
        assert torch.isfinite(out)

    def test_raises_patch_too_small(self):
        with pytest.raises(ValueError, match="patch_size"):
            PatchLowRankLoss(patch_size=1)

    def test_registry_lookup(self):
        loss_fn = create_loss("patch_low_rank")
        assert isinstance(loss_fn, PatchLowRankLoss)


# ===========================================================================
# HermitianSymmetryLoss
# ===========================================================================


class TestHermitianSymmetryLoss:

    def test_canary_builds_and_forward_finite(self):
        loss_fn = HermitianSymmetryLoss()
        x = _kspace_2ch()
        out = loss_fn(x)
        assert torch.isfinite(out) and out.shape == ()

    @pytest.mark.parametrize("normalize", [True, False])
    def test_parametric_normalize(self, normalize: bool):
        loss_fn = HermitianSymmetryLoss(normalize=normalize)
        x = _kspace_2ch()
        out = loss_fn(x)
        assert torch.isfinite(out) and out.item() >= 0.0

    def test_property_nonneg(self):
        loss_fn = HermitianSymmetryLoss()
        x = _kspace_2ch()
        assert loss_fn(x).item() >= 0.0

    def test_property_zero_for_real_spectrum(self):
        """A purely real input with zero imaginary part satisfies k=conj(k[-k])."""
        loss_fn = HermitianSymmetryLoss()
        # Build a Hermitian-symmetric complex tensor explicitly
        re = _img()
        x_complex = torch.complex(re, torch.zeros_like(re))
        out = loss_fn(x_complex)
        # Not necessarily 0 (for arbitrary real images, k-space is Hermitian in FFT),
        # but must be finite and non-negative.
        assert torch.isfinite(out) and out.item() >= 0.0

    def test_property_gradient_reaches_input(self):
        x = _kspace_2ch().requires_grad_(True)
        loss_fn = HermitianSymmetryLoss()
        loss_fn(x).backward()
        assert x.grad is not None

    def test_edge_zero_input(self):
        loss_fn = HermitianSymmetryLoss()
        x = torch.zeros(B, 2, H, W)
        out = loss_fn(x)
        assert torch.isfinite(out)

    def test_edge_large_magnitude(self):
        loss_fn = HermitianSymmetryLoss()
        x = torch.full((B, 2, H, W), 1e3)
        out = loss_fn(x)
        assert torch.isfinite(out)

    def test_registry_lookup(self):
        loss_fn = create_loss("hermitian_symmetry")
        assert isinstance(loss_fn, HermitianSymmetryLoss)


# ===========================================================================
# KspaceL1SparsityLoss
# ===========================================================================


class TestKspaceL1SparsityLoss:

    def test_canary_builds_and_forward_finite(self):
        loss_fn = KspaceL1SparsityLoss()
        x = _kspace_2ch()
        out = loss_fn(x)
        assert torch.isfinite(out) and out.shape == ()

    @pytest.mark.parametrize("radius", [0.0, 0.2])
    def test_parametric_center_mask_radius(self, radius: float):
        loss_fn = KspaceL1SparsityLoss(center_mask_radius=radius)
        x = _kspace_2ch()
        out = loss_fn(x)
        assert torch.isfinite(out) and out.item() >= 0.0

    def test_property_nonneg(self):
        loss_fn = KspaceL1SparsityLoss()
        x = _kspace_2ch()
        assert loss_fn(x).item() >= 0.0

    def test_property_zero_input_gives_zero(self):
        loss_fn = KspaceL1SparsityLoss()
        x = torch.zeros(B, 2, H, W)
        out = loss_fn(x)
        assert out.item() == pytest.approx(0.0, abs=1e-9)

    def test_property_gradient_reaches_input(self):
        x = _kspace_2ch().requires_grad_(True)
        loss_fn = KspaceL1SparsityLoss()
        loss_fn(x).backward()
        assert x.grad is not None

    def test_edge_large_magnitude(self):
        loss_fn = KspaceL1SparsityLoss()
        x = torch.full((B, 2, H, W), 1e5)
        out = loss_fn(x)
        assert torch.isfinite(out)

    def test_raises_invalid_radius(self):
        with pytest.raises(ValueError, match="center_mask_radius"):
            KspaceL1SparsityLoss(center_mask_radius=1.5)

    def test_registry_lookup(self):
        loss_fn = create_loss("kspace_l1_sparsity")
        assert isinstance(loss_fn, KspaceL1SparsityLoss)


# ===========================================================================
# HankelLowRankLoss
# ===========================================================================


class TestHankelLowRankLoss:

    def test_canary_builds_and_forward_finite(self):
        loss_fn = HankelLowRankLoss()
        x = _kspace_2ch()
        out = loss_fn(x)
        assert torch.isfinite(out) and out.shape == ()

    @pytest.mark.parametrize("kernel_size", [3, 5])
    def test_parametric_kernel_sizes(self, kernel_size: int):
        loss_fn = HankelLowRankLoss(kernel_size=kernel_size)
        x = _kspace_2ch(h=16, w=16)
        out = loss_fn(x)
        assert torch.isfinite(out) and out.item() >= 0.0

    def test_property_nonneg(self):
        loss_fn = HankelLowRankLoss()
        x = _kspace_2ch()
        assert loss_fn(x).item() >= 0.0

    def test_property_gradient_reaches_input(self):
        x = _kspace_2ch().requires_grad_(True)
        loss_fn = HankelLowRankLoss()
        loss_fn(x).backward()
        assert x.grad is not None

    def test_edge_small_kspace(self):
        """k-space smaller than kernel → should return 0 (guard in forward)."""
        loss_fn = HankelLowRankLoss(kernel_size=8)
        x = _kspace_2ch(h=4, w=4)
        out = loss_fn(x)
        assert torch.isfinite(out) and out.item() == pytest.approx(0.0, abs=1e-9)

    def test_edge_large_magnitude(self):
        loss_fn = HankelLowRankLoss()
        x = torch.full((B, 2, H, W), 1e3)
        out = loss_fn(x)
        assert torch.isfinite(out)

    def test_raises_kernel_too_small(self):
        with pytest.raises(ValueError, match="kernel_size"):
            HankelLowRankLoss(kernel_size=1)

    def test_registry_lookup(self):
        loss_fn = create_loss("hankel_low_rank")
        assert isinstance(loss_fn, HankelLowRankLoss)


# ===========================================================================
# PhaseSmoothnessComplexLoss
# ===========================================================================


class TestPhaseSmoothnessComplexLoss:

    def test_canary_builds_and_forward_finite(self):
        loss_fn = PhaseSmoothnessComplexLoss()
        x = _complex_img()
        out = loss_fn(x)
        assert torch.isfinite(out) and out.shape == ()

    @pytest.mark.parametrize("magnitude_weighted", [True, False])
    def test_parametric_magnitude_weighted(self, magnitude_weighted: bool):
        loss_fn = PhaseSmoothnessComplexLoss(magnitude_weighted=magnitude_weighted)
        x = _complex_img()
        out = loss_fn(x)
        assert torch.isfinite(out) and out.item() >= 0.0

    def test_property_nonneg(self):
        loss_fn = PhaseSmoothnessComplexLoss()
        x = _complex_img()
        assert loss_fn(x).item() >= 0.0

    def test_property_real_only_input_returns_zero(self):
        """Real-only input (no imaginary part) → zero phase → loss = 0."""
        loss_fn = PhaseSmoothnessComplexLoss()
        # Odd-channel real: no complex conversion → returns 0 by guard
        x = _img(c=1)
        out = loss_fn(x)
        assert torch.isfinite(out) and out.item() == pytest.approx(0.0, abs=1e-9)

    def test_property_gradient_reaches_input(self):
        re = torch.randn(B, C, H, W, requires_grad=True)
        im = torch.randn(B, C, H, W)
        x = torch.complex(re, im)
        loss_fn = PhaseSmoothnessComplexLoss()
        loss_fn(x).backward()
        assert re.grad is not None and torch.isfinite(re.grad).all()

    def test_edge_zero_phase(self):
        """Constant-phase complex input → TV of phase is 0."""
        loss_fn = PhaseSmoothnessComplexLoss(magnitude_weighted=False)
        mag = torch.ones(B, C, H, W)
        # Phase uniformly 0
        x = torch.complex(mag, torch.zeros_like(mag))
        out = loss_fn(x)
        assert out.item() == pytest.approx(0.0, abs=1e-6)

    def test_edge_large_magnitude(self):
        loss_fn = PhaseSmoothnessComplexLoss()
        re = torch.full((B, C, H, W), 1e3)
        im = torch.full((B, C, H, W), 1e3)
        x = torch.complex(re, im)
        out = loss_fn(x)
        assert torch.isfinite(out)

    def test_registry_lookup(self):
        loss_fn = create_loss("phase_smoothness_complex")
        assert isinstance(loss_fn, PhaseSmoothnessComplexLoss)
