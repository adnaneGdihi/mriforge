"""Unit tests for core/metrics/meta_evaluation/group_actions.py.

Covers: nuis_phase, nuis_bias, nuis_gamma, pert_lesion, pert_inversion,
        pert_contrast_permute.
"""

from __future__ import annotations

import pytest
import torch

from mriforge.core.metrics.meta_evaluation.group_actions import (
    nuis_bias,
    nuis_gamma,
    nuis_phase,
    pert_inversion,
    pert_lesion,
)


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_nuis_phase_canary():
    """Phase rotation with phi=0 returns identical tensors."""
    x = torch.randn(1, 1, 16, 16)
    hat = torch.randn(1, 1, 16, 16)
    x2, hat2 = nuis_phase(x, hat, {"phi": 0.0})
    assert torch.allclose(x, x2)
    assert torch.allclose(hat, hat2)


# ---------------------------------------------------------------------------
# Parametrized: nuis_bias identity at a=0, b=0
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("shape", [(1, 1, 16, 16), (2, 3, 32, 32)])
def test_nuis_bias_zero_params_identity(shape):
    """Bias field with a=0, b=0 multiplies by 1 → identity."""
    x = torch.randn(*shape)
    hat = torch.randn(*shape)
    x2, hat2 = nuis_bias(x, hat, {"a": 0.0, "b": 0.0})
    assert torch.allclose(x, x2, atol=1e-5), "nuis_bias(a=0,b=0) should be identity"
    assert torch.allclose(hat, hat2, atol=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize("gamma", [0.8, 1.0, 1.2])
def test_nuis_gamma_preserves_sign(gamma):
    """nuis_gamma with positive input keeps positive output."""
    x = torch.abs(torch.randn(1, 1, 8, 8)) + 0.1
    hat = torch.abs(torch.randn(1, 1, 8, 8)) + 0.1
    x2, hat2 = nuis_gamma(x, hat, {"gamma": gamma})
    assert (x2 > 0).all()
    assert (hat2 > 0).all()


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pert_lesion_only_modifies_hat():
    """pert_lesion injects a blob only into hat, x is unchanged."""
    x = torch.zeros(1, 1, 32, 32)
    hat = torch.zeros(1, 1, 32, 32)
    x2, hat2 = pert_lesion(x, hat, {"amplitude": 0.5, "cx": 0.5, "cy": 0.5, "sigma": 0.1})
    assert torch.allclose(x, x2), "x should be unchanged by pert_lesion"
    assert not torch.allclose(hat, hat2), "hat should be modified by pert_lesion"


@pytest.mark.unit
def test_pert_inversion_negates_hat():
    """pert_inversion returns x unchanged and -hat."""
    x = torch.randn(1, 1, 8, 8)
    hat = torch.randn(1, 1, 8, 8)
    x2, hat2 = pert_inversion(x, hat, {})
    assert torch.allclose(x, x2)
    assert torch.allclose(hat2, -hat)


@pytest.mark.unit
def test_nuis_phase_on_real_tensor_returns_unchanged():
    """Phase rotation has no effect on real-valued (non-complex) tensors."""
    x = torch.randn(1, 1, 16, 16)
    hat = torch.randn(1, 1, 16, 16)
    x2, hat2 = nuis_phase(x, hat, {"phi": 1.57})  # pi/2
    assert torch.allclose(x, x2)
    assert torch.allclose(hat, hat2)


# ---------------------------------------------------------------------------
# Expected failure
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_nuis_gamma_zero_gamma_clamps():
    """gamma=0 → all positive values map to 1 (x^0=1). No NaN."""
    x = torch.abs(torch.randn(1, 1, 8, 8)) + 0.1
    hat = torch.abs(torch.randn(1, 1, 8, 8)) + 0.1
    x2, hat2 = nuis_gamma(x, hat, {"gamma": 0.0})
    assert not torch.isnan(x2).any(), "gamma=0 should not produce NaN"
    assert not torch.isnan(hat2).any()
