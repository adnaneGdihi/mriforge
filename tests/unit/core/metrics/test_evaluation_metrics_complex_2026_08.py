"""D15#4: PSNR / SNR / SSIM on complex (k-space) tensors.

Measured before the fix, all three raised rather than returning a wrong number:

    PSNR(domain="kspace")  NotImplementedError: "mse_cpu"  not implemented for 'ComplexFloat'
    SNR                    NotImplementedError: "lt_cpu"   not implemented for 'ComplexFloat'
    SSIM                   NotImplementedError: "min_all"  not implemented for 'ComplexFloat'

PSNR's ``domain="kspace"`` branch exists specifically to grade k-space (it
derives its data_range from ``torch.abs(target).max()``), so crashing on complex
input made the branch unreachable for the data it was written for.

The plan row proposed ``mse_loss(view_as_real(a), view_as_real(b))``. That
form averages over 2x the elements and therefore returns HALF the squared error
-- a +3.01 dB PSNR bias, pinned below. E[|a-b|^2] is the physical residual
energy and matches MSE/NMSE/NRMSE elsewhere in this module.
"""

import math

import pytest
import torch
from torch.nn import functional

from spectramr.core.metrics.evaluation_metrics import PSNR, SNR, SSIMMetric


@pytest.fixture()
def complex_pair():
    torch.manual_seed(0)
    target = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
    preds = target + 0.1 * torch.randn(1, 1, 16, 16, dtype=torch.complex64)
    return preds, target


@pytest.fixture()
def real_pair():
    torch.manual_seed(0)
    target = torch.rand(1, 1, 16, 16)
    preds = (target + 0.05 * torch.randn(1, 1, 16, 16)).clamp(0, 1)
    return preds, target


def test_psnr_kspace_accepts_complex(complex_pair):
    preds, target = complex_pair
    value = PSNR(domain="kspace")(preds, target)
    assert torch.isfinite(value)
    assert not value.is_complex()


def test_psnr_kspace_matches_the_magnitude_squared_definition(complex_pair):
    preds, target = complex_pair
    dr = max(float(target.abs().max()), 0.01)
    mse = float((preds - target).abs().pow(2).mean())
    expected = 20 * math.log10(dr / (math.sqrt(mse) + 1e-10))
    assert float(PSNR(domain="kspace")(preds, target)) == pytest.approx(expected, abs=1e-4)


def test_view_as_real_form_would_inflate_psnr_by_exactly_3_01_db(complex_pair):
    """Pins WHY the rejected form is rejected, so a future "simplification"
    back to ``view_as_real`` fails here with the number in hand."""
    preds, target = complex_pair
    true_mse = float((preds - target).abs().pow(2).mean())
    halved = float(functional.mse_loss(torch.view_as_real(preds), torch.view_as_real(target)))
    assert halved == pytest.approx(true_mse / 2, rel=1e-6)
    # PSNR is 20*log10(dr / sqrt(mse)), so halving the MSE ADDS 10*log10(2) dB.
    inflation_db = 20 * math.log10(math.sqrt(true_mse / halved))
    assert inflation_db == pytest.approx(10 * math.log10(2), abs=1e-6)
    assert inflation_db == pytest.approx(3.0103, abs=1e-4)


def test_snr_accepts_complex_and_is_real_valued(complex_pair):
    preds, target = complex_pair
    value = SNR()(preds, target)
    assert not value.is_complex()
    expected = 10 * math.log10(
        float(target.abs().pow(2).mean()) / float((preds - target).abs().pow(2).mean())
    )
    assert float(value) == pytest.approx(expected, abs=1e-4)


def test_ssim_refuses_complex_with_an_actionable_message(complex_pair):
    """SSIM has no accepted complex form. Silently taking ``.abs()`` would grade
    a different quantity than the caller passed; the error names the fix."""
    preds, target = complex_pair
    with pytest.raises(TypeError, match="undefined for complex"):
        SSIMMetric()(preds, target)
    # ... and the suggested workaround actually works (k-space magnitudes are
    # not [0, 1], so the range has to be declared -- the same contract any
    # unnormalized real input faces).
    dr = float(target.abs().max())
    assert torch.isfinite(SSIMMetric()(preds.abs(), target.abs(), data_range=dr))


@pytest.mark.parametrize("metric", [PSNR, SNR, SSIMMetric])
def test_real_input_is_unchanged(metric, real_pair):
    """The complex branches must not perturb the real path they guard."""
    preds, target = real_pair
    value = metric()(preds, target)
    assert torch.isfinite(value) and not value.is_complex()


def test_real_psnr_still_uses_mse_loss(real_pair):
    preds, target = real_pair
    dr = 1.0
    mse = float(functional.mse_loss(preds, target))
    expected = 20 * math.log10(dr / (math.sqrt(mse) + 1e-10))
    assert float(PSNR(data_range=dr)(preds, target)) == pytest.approx(expected, abs=1e-4)
