"""Tests for the conditioning encoder registry + encoders (2026-05-26).

Each encoder reads ONE source from a ConditioningContext and emits a
``[B, embed_dim]`` embedding; the bank fuses several (the joint co-embedding).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mriforge.models.conditioning import ConditioningContext  # noqa: E402
from mriforge.models.conditioning.encoders import (  # noqa: E402
    AcquisitionTimeEncoder,
    ConditioningEncoderBank,
    SeverityTokenEncoder,
    SinusoidalTimeEncoder,
    get_conditioner,
    register_conditioner,
)


def test_registry_dispatch_returns_right_class() -> None:
    assert isinstance(get_conditioner("diffusion_t", embed_dim=16), SinusoidalTimeEncoder)
    assert isinstance(get_conditioner("severity_vec", embed_dim=16), SeverityTokenEncoder)


def test_registry_unknown_source_raises() -> None:
    with pytest.raises(KeyError, match="banana"):
        get_conditioner("banana", embed_dim=8)


def test_duplicate_registration_raises() -> None:
    with pytest.raises(ValueError, match="already registered"):

        @register_conditioner("diffusion_t")
        class _Dup(SinusoidalTimeEncoder):
            pass


def test_time_encoder_shape() -> None:
    out = SinusoidalTimeEncoder(embed_dim=16)(ConditioningContext(diffusion_t=torch.rand(4)))
    assert out.shape == (4, 16)


def test_time_encoder_accepts_b1_shape() -> None:
    out = SinusoidalTimeEncoder(embed_dim=8)(ConditioningContext(diffusion_t=torch.rand(4, 1)))
    assert out.shape == (4, 8)


def test_severity_encoder_shape() -> None:
    enc = SeverityTokenEncoder(embed_dim=16, num_severities=5)
    out = enc(ConditioningContext(severity_vec=torch.rand(4, 5)))
    assert out.shape == (4, 16)


def test_acq_time_encoder_pools_over_lines() -> None:
    enc = AcquisitionTimeEncoder(embed_dim=12)
    out = enc(ConditioningContext(acq_time_tau=torch.rand(4, 32)))
    assert out.shape == (4, 12)


def test_encoder_missing_source_raises_clear_error() -> None:
    enc = SeverityTokenEncoder(embed_dim=8, num_severities=5)
    with pytest.raises(ValueError, match="severity_vec"):
        enc(ConditioningContext(diffusion_t=torch.rand(2)))


def test_bank_concat_fusion_shape() -> None:
    bank = ConditioningEncoderBank(
        ["diffusion_t", "severity_vec"], embed_dim=16, fusion="concat"
    )
    ctx = ConditioningContext(diffusion_t=torch.rand(4), severity_vec=torch.rand(4, 5))
    assert bank(ctx).shape == (4, 16)


def test_bank_sum_fusion_shape() -> None:
    bank = ConditioningEncoderBank(
        ["diffusion_t", "severity_vec"], embed_dim=16, fusion="sum"
    )
    ctx = ConditioningContext(diffusion_t=torch.rand(4), severity_vec=torch.rand(4, 5))
    assert bank(ctx).shape == (4, 16)


def test_bank_rejects_unknown_source() -> None:
    with pytest.raises(KeyError):
        ConditioningEncoderBank(["banana"], embed_dim=8)


def test_bank_requires_at_least_one_source() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ConditioningEncoderBank([], embed_dim=8)


def test_bank_propagates_missing_source_at_forward() -> None:
    bank = ConditioningEncoderBank(["severity_vec"], embed_dim=8)
    with pytest.raises(ValueError, match="severity_vec"):
        bank(ConditioningContext(diffusion_t=torch.rand(2)))


def test_bank_is_differentiable() -> None:
    bank = ConditioningEncoderBank(["diffusion_t", "severity_vec"], embed_dim=8)
    ctx = ConditioningContext(
        diffusion_t=torch.rand(2, requires_grad=True),
        severity_vec=torch.rand(2, 5, requires_grad=True),
    )
    bank(ctx).sum().backward()
    assert ctx.severity_vec.grad is not None
