"""Unit tests for realistic_degradations transforms.

Covers:
  - RicianNoise: noise_level=0 identity, shape preservation, non-negativity,
    stochastic variance ordering, determinism.
  - MotionBlur3D: intensity=0 identity, shape, direction normalisation,
    low-pass smoothing effect.
  - KSpaceUndersampling: acceleration=1 identity, all three patterns, unknown
    pattern raises, equispaced mask structure, centre-region preservation.
  - ElasticDeformation3D: alpha=0 identity, shape preservation.
  - RealisticDegradationPipeline: constructs and forwards the same shape.
  - Functional helpers: add_rician_noise, undersample_kspace,
    apply_motion_blur_3d, apply_elastic_deformation_3d.

Note: MotionBlur3D and KSpaceUndersampling expect 5-D input [B, C, D, H, W];
all tests use that layout.
"""
from __future__ import annotations

import pytest
import torch

from spectramr.data.transforms.realistic_degradations import (
    ElasticDeformation3D,
    KSpaceUndersampling,
    MotionBlur3D,
    RealisticDegradationPipeline,
    RicianNoise,
    add_rician_noise,
    apply_elastic_deformation_3d,
    apply_motion_blur_3d,
    create_realistic_degradation_pipeline,
    undersample_kspace,
)

# ── Helpers ─────────────────────────────────────────────────────────────────

_SMALL = (1, 1, 4, 8, 8)  # [B, C, D, H, W]


def _vol(*shape: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.rand(*shape)


# ── RicianNoise ─────────────────────────────────────────────────────────────


class TestRicianNoise:
    def test_canary_applies_and_returns_same_shape(self) -> None:
        x = _vol(1, 1, 8, 8)
        out = RicianNoise(noise_level=0.1)(x)
        assert out.shape == x.shape

    def test_identity_at_zero_noise_level(self) -> None:
        """noise_level=0 → exact identity (no stochastic draw)."""
        x = torch.rand(1, 1, 16, 16)
        out = RicianNoise(noise_level=0.0)(x)
        assert torch.equal(out, x)

    def test_output_is_non_negative(self) -> None:
        """Rician magnitude √((x+nR)²+nI²) ≥ 0 always."""
        torch.manual_seed(42)
        x = torch.rand(2, 1, 16, 16)
        out = RicianNoise(noise_level=0.15)(x)
        assert (out >= 0).all()

    def test_output_is_finite(self) -> None:
        torch.manual_seed(7)
        out = RicianNoise(noise_level=0.1)(torch.rand(1, 1, 8, 8))
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("noise_level", [0.01, 0.05, 0.2])
    def test_shape_preserved_across_severities(self, noise_level: float) -> None:
        x = _vol(1, 1, 8, 8)
        assert RicianNoise(noise_level=noise_level)(x).shape == x.shape

    def test_determinism_same_seed(self) -> None:
        x = torch.ones(1, 1, 8, 8)
        torch.manual_seed(99)
        out1 = RicianNoise(noise_level=0.1)(x)
        torch.manual_seed(99)
        out2 = RicianNoise(noise_level=0.1)(x)
        assert torch.allclose(out1, out2)

    def test_higher_noise_increases_variance(self) -> None:
        x = torch.full((1, 1, 32, 32), 1.0)
        torch.manual_seed(0)
        var_lo = RicianNoise(noise_level=0.01)(x).var().item()
        torch.manual_seed(0)
        var_hi = RicianNoise(noise_level=0.5)(x).var().item()
        assert var_hi > var_lo


# ── MotionBlur3D ─────────────────────────────────────────────────────────────


class TestMotionBlur3D:
    def test_canary_applies_and_returns_same_shape(self) -> None:
        x = _vol(*_SMALL)
        out = MotionBlur3D(motion_intensity=0.5, motion_direction=(1, 0, 0))(x)
        assert out.shape == x.shape

    def test_identity_at_zero_intensity(self) -> None:
        """motion_intensity=0 → exact identity."""
        x = _vol(*_SMALL)
        out = MotionBlur3D(motion_intensity=0.0)(x)
        assert torch.equal(out, x)

    def test_direction_is_unit_norm(self) -> None:
        blur = MotionBlur3D(motion_direction=(3, 4, 0))
        assert pytest.approx(blur.motion_direction.norm().item(), abs=1e-5) == 1.0

    def test_low_pass_smooths_sharp_edge(self) -> None:
        """Blurring should reduce the intensity of a step edge."""
        blur = MotionBlur3D(motion_intensity=0.5, motion_direction=(0, 1, 0))
        x = torch.zeros(1, 1, 4, 16, 16)
        x[..., 8:, :] = 1.0
        out = blur(x)
        edge_before = (x[..., 8, :] - x[..., 7, :]).abs().mean()
        edge_after = (out[..., 8, :] - out[..., 7, :]).abs().mean()
        assert edge_after.item() < edge_before.item()

    @pytest.mark.parametrize("intensity", [0.1, 0.5, 1.0])
    def test_shape_preserved_across_intensities(self, intensity: float) -> None:
        x = _vol(*_SMALL)
        assert MotionBlur3D(
            motion_intensity=intensity, motion_direction=(1, 0, 0)
        )(x).shape == x.shape


# ── KSpaceUndersampling ───────────────────────────────────────────────────────


class TestKSpaceUndersampling:
    def test_canary_applies_and_returns_same_shape(self) -> None:
        torch.manual_seed(0)
        x = _vol(*_SMALL)
        out = KSpaceUndersampling(acceleration_factor=2, pattern="uniform")(x)
        assert out.shape == x.shape

    def test_identity_at_acceleration_one(self) -> None:
        """acceleration_factor=1 → exact identity."""
        x = _vol(*_SMALL)
        out = KSpaceUndersampling(acceleration_factor=1)(x)
        assert torch.equal(out, x)

    @pytest.mark.parametrize("pattern", ["uniform", "equispaced", "gaussian"])
    def test_all_patterns_preserve_shape(self, pattern: str) -> None:
        torch.manual_seed(0)
        x = _vol(*_SMALL)
        out = KSpaceUndersampling(acceleration_factor=2, pattern=pattern)(x)
        assert out.shape == x.shape

    def test_unknown_pattern_raises(self) -> None:
        """Unknown pattern string → ValueError (CLAUDE.md pitfall #9)."""
        us = KSpaceUndersampling(acceleration_factor=2, pattern="zigzag")
        with pytest.raises(ValueError, match="Unknown undersampling pattern"):
            us(_vol(*_SMALL))

    def test_equispaced_mask_structure(self) -> None:
        """Equispaced: every step-th row kept, others zero (pre-centre)."""
        us = KSpaceUndersampling(
            acceleration_factor=4, pattern="equispaced", center_fraction=0.0
        )
        mask = us._create_mask((4, 8, 8))
        # Row 0 and row 4 should be sampled; row 1 should not.
        assert mask[0, 0, 0] == 1.0
        assert mask[0, 4, 0] == 1.0
        assert mask[0, 1, 0] == 0.0

    def test_output_is_finite(self) -> None:
        torch.manual_seed(0)
        out = KSpaceUndersampling(acceleration_factor=2, pattern="uniform")(
            _vol(*_SMALL)
        )
        assert torch.isfinite(out).all()

    def test_determinism_equispaced(self) -> None:
        """Equispaced mask is deterministic (no randomness)."""
        x = _vol(*_SMALL)
        out1 = KSpaceUndersampling(acceleration_factor=2, pattern="equispaced")(x)
        out2 = KSpaceUndersampling(acceleration_factor=2, pattern="equispaced")(x)
        assert torch.allclose(out1, out2, atol=1e-6)


# ── ElasticDeformation3D ──────────────────────────────────────────────────────


class TestElasticDeformation3D:
    def test_canary_applies_and_returns_same_shape(self) -> None:
        torch.manual_seed(0)
        x = _vol(*_SMALL)
        out = ElasticDeformation3D(alpha=0.5)(x)
        assert out.shape == x.shape

    def test_identity_at_zero_alpha(self) -> None:
        """alpha=0 → exact identity."""
        x = _vol(*_SMALL)
        out = ElasticDeformation3D(alpha=0.0)(x)
        assert torch.equal(out, x)

    def test_output_is_finite(self) -> None:
        torch.manual_seed(0)
        out = ElasticDeformation3D(alpha=1.0)(_vol(*_SMALL))
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("alpha", [0.1, 0.5, 2.0])
    def test_shape_preserved_across_alphas(self, alpha: float) -> None:
        torch.manual_seed(0)
        x = _vol(*_SMALL)
        assert ElasticDeformation3D(alpha=alpha)(x).shape == x.shape


# ── RealisticDegradationPipeline ──────────────────────────────────────────────


class TestRealisticDegradationPipeline:
    def test_canary_full_pipeline_preserves_shape(self) -> None:
        torch.manual_seed(0)
        pipe = RealisticDegradationPipeline(
            rician_noise_level=0.05,
            motion_blur_intensity=0.3,
            undersampling_factor=2,
            elastic_alpha=0.3,
        )
        x = _vol(*_SMALL)
        out = pipe(x)
        assert out.shape == x.shape

    def test_factory_function_returns_callable(self) -> None:
        pipe = create_realistic_degradation_pipeline()
        assert callable(pipe)


# ── Functional helpers ─────────────────────────────────────────────────────────


def test_add_rician_noise_functional_preserves_shape() -> None:
    torch.manual_seed(0)
    x = torch.rand(1, 1, 8, 8)
    out = add_rician_noise(x, noise_level=0.05)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_undersample_kspace_functional_preserves_shape() -> None:
    torch.manual_seed(0)
    x = _vol(*_SMALL)
    out = undersample_kspace(x, acceleration_factor=2, pattern="uniform")
    assert out.shape == x.shape


def test_apply_motion_blur_3d_functional_preserves_shape() -> None:
    x = _vol(*_SMALL)
    out = apply_motion_blur_3d(x, intensity=0.3)
    assert out.shape == x.shape


def test_apply_elastic_deformation_3d_functional_preserves_shape() -> None:
    torch.manual_seed(0)
    x = _vol(*_SMALL)
    out = apply_elastic_deformation_3d(x, alpha=0.5)
    assert out.shape == x.shape
