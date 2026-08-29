"""Unit tests for precision-aware casters and managers.

Targets ``mriforge.infrastructure.training.precision_annotations``:
- ``PrecisionAwareCaster.cast_to_fp32``
- ``PrecisionAwareCaster.cast_to_fp16``
- ``PrecisionAwareCaster.validate_precision``
- ``VAEPrecisionManager.prepare_latent_for_kl_computation``
- ``VAEPrecisionManager.compute_kl_divergence``
- ``VQVAEPrecisionManager.validate_vq_loss_precision``
- ``VQVAEPrecisionManager.validate_codebook_precision``
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------


def test_canary_caster_fp32() -> None:
    from mriforge.infrastructure.training.precision_annotations import PrecisionAwareCaster

    caster = PrecisionAwareCaster("test")
    t = torch.tensor([1.0], dtype=torch.float16)
    result = caster.cast_to_fp32(t)
    assert result.dtype == torch.float32


# ---------------------------------------------------------------------------
# PrecisionAwareCaster.cast_to_fp32 / cast_to_fp16
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_dtype,target_method,expected_dtype",
    [
        (torch.float16, "cast_to_fp32", torch.float32),
        (torch.float32, "cast_to_fp32", torch.float32),
        (torch.float32, "cast_to_fp16", torch.float16),
        (torch.float16, "cast_to_fp16", torch.float16),
    ],
)
def test_cast_returns_expected_dtype(input_dtype, target_method, expected_dtype) -> None:
    from mriforge.infrastructure.training.precision_annotations import PrecisionAwareCaster

    caster = PrecisionAwareCaster("model_x")
    t = torch.randn(4, 4).to(input_dtype)
    result = getattr(caster, target_method)(t)
    assert result.dtype == expected_dtype


def test_cast_preserves_values_numerically() -> None:
    from mriforge.infrastructure.training.precision_annotations import PrecisionAwareCaster

    caster = PrecisionAwareCaster("model_v")
    # Create fp32 → cast to fp16 → cast back to fp32
    original = torch.tensor([1.5, -2.0, 0.25], dtype=torch.float32)
    fp16 = caster.cast_to_fp16(original)
    fp32_back = caster.cast_to_fp32(fp16)
    # fp16 precision; allow tolerance
    assert torch.allclose(original, fp32_back, atol=1e-3)


# ---------------------------------------------------------------------------
# PrecisionAwareCaster.validate_precision
# ---------------------------------------------------------------------------


def test_validate_precision_passes_correct_dtype() -> None:
    from mriforge.infrastructure.training.precision_annotations import PrecisionAwareCaster

    caster = PrecisionAwareCaster("v")
    t = torch.randn(2, 2, dtype=torch.float32)
    # Should NOT raise.
    caster.validate_precision(t, torch.float32, "test_op")


def test_validate_precision_raises_on_mismatch() -> None:
    from mriforge.infrastructure.training.precision_annotations import PrecisionAwareCaster

    caster = PrecisionAwareCaster("v")
    t = torch.randn(2, dtype=torch.float16)
    with pytest.raises(ValueError, match="Precision mismatch"):
        caster.validate_precision(t, torch.float32, "bad_op")


# ---------------------------------------------------------------------------
# VAEPrecisionManager.prepare_latent_for_kl_computation
# ---------------------------------------------------------------------------


def test_vae_prepare_latent_casts_to_fp32() -> None:
    from mriforge.infrastructure.training.precision_annotations import VAEPrecisionManager

    mgr = VAEPrecisionManager(device=torch.device("cpu"))
    mu_fp16 = torch.randn(2, 4, dtype=torch.float16)
    logvar_fp16 = torch.randn(2, 4, dtype=torch.float16)
    mu_fp32, logvar_fp32 = mgr.prepare_latent_for_kl_computation(mu_fp16, logvar_fp16)
    assert mu_fp32.dtype == torch.float32
    assert logvar_fp32.dtype == torch.float32


def test_vae_compute_kl_divergence_is_scalar() -> None:
    from mriforge.infrastructure.training.precision_annotations import VAEPrecisionManager

    mgr = VAEPrecisionManager(device=torch.device("cpu"))
    mu = torch.zeros(4, 8, dtype=torch.float32)
    logvar = torch.zeros(4, 8, dtype=torch.float32)
    kl = mgr.compute_kl_divergence(mu, logvar)
    # KL(N(0,1)||N(0,1)) = 0
    assert kl.item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# VQVAEPrecisionManager
# ---------------------------------------------------------------------------


def test_vqvae_validate_vq_loss_fp32_passthrough() -> None:
    from mriforge.infrastructure.training.precision_annotations import VQVAEPrecisionManager

    mgr = VQVAEPrecisionManager(device=torch.device("cpu"))
    loss = torch.tensor(0.5, dtype=torch.float32)
    out = mgr.validate_vq_loss_precision(loss)
    assert out.dtype == torch.float32


def test_vqvae_validate_vq_loss_upcasts_fp16() -> None:
    from mriforge.infrastructure.training.precision_annotations import VQVAEPrecisionManager

    mgr = VQVAEPrecisionManager(device=torch.device("cpu"))
    loss = torch.tensor(0.5, dtype=torch.float16)
    out = mgr.validate_vq_loss_precision(loss)
    assert out.dtype == torch.float32


def test_vqvae_validate_codebook_fp32_passes() -> None:
    from mriforge.infrastructure.training.precision_annotations import VQVAEPrecisionManager

    mgr = VQVAEPrecisionManager(device=torch.device("cpu"))
    codebook = torch.nn.Parameter(torch.randn(16, 8, dtype=torch.float32))
    # Should NOT raise.
    mgr.validate_codebook_precision(codebook)


def test_vqvae_validate_codebook_fp16_raises() -> None:
    from mriforge.infrastructure.training.precision_annotations import VQVAEPrecisionManager

    mgr = VQVAEPrecisionManager(device=torch.device("cpu"))
    codebook = torch.nn.Parameter(torch.randn(16, 8, dtype=torch.float16))
    with pytest.raises(ValueError, match="FP32"):
        mgr.validate_codebook_precision(codebook)
