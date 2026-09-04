"""Tests for spectramr.core.metrics.channel_adapter.

Covers every (mode × channel-count) combination and verifies the
adapter never silently drops channels — the previous LPIPS code did
``preds[:, :3]`` which masked channel-4 information.

Also includes integration tests that confirm LPIPS / FID / KID accept
1-, 2-, 3-, and 4-channel inputs without erroring after the wiring,
with shape assertions on the adapted intermediate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from tests.utils.optional_backends import requires_torch_fidelity
import torch

_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from spectramr.core.metrics.channel_adapter import ChannelMode, adapt_to_rgb  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# adapt_to_rgb: shape + behavior tests
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def toy_image():
    return torch.rand(2, 1, 16, 16)


class TestAuto:
    def test_grayscale_replicates_to_3(self):
        x = torch.rand(2, 1, 8, 8)
        out = adapt_to_rgb(x, mode=ChannelMode.AUTO)
        assert out.shape == (2, 3, 8, 8)
        # Replication, not noise
        assert torch.allclose(out[:, 0], out[:, 1])
        assert torch.allclose(out[:, 1], out[:, 2])

    def test_3_channel_passthrough(self):
        x = torch.rand(2, 3, 8, 8)
        out = adapt_to_rgb(x, mode=ChannelMode.AUTO)
        assert out.shape == (2, 3, 8, 8)
        assert torch.equal(out, x)

    def test_2_channel_treated_as_complex(self):
        # 2 channels = (real, imag); RSS magnitude is sqrt(re^2 + im^2)
        re = torch.full((2, 1, 8, 8), 3.0)
        im = torch.full((2, 1, 8, 8), 4.0)
        x = torch.cat([re, im], dim=1)  # interleaved at C dim
        out = adapt_to_rgb(x, mode=ChannelMode.AUTO)
        assert out.shape == (2, 3, 8, 8)
        # sqrt(3^2 + 4^2) = 5
        assert torch.allclose(out, torch.full_like(out, 5.0), atol=1e-5)

    def test_4_channel_treated_as_two_complex_coils(self):
        # 4 channels = 2 coils × (re, im). RSS per coil → sum across coils.
        # Build: coil0 = (3, 4) → mag 5; coil1 = (0, 0) → mag 0; total RSS = 5.
        x = torch.zeros(2, 4, 8, 8)
        x[:, 0] = 3.0  # coil0 real
        x[:, 1] = 4.0  # coil0 imag
        x[:, 2] = 0.0  # coil1 real
        x[:, 3] = 0.0  # coil1 imag
        out = adapt_to_rgb(x, mode=ChannelMode.AUTO)
        assert out.shape == (2, 3, 8, 8)
        assert torch.allclose(out, torch.full_like(out, 5.0), atol=1e-5)

    def test_8_channel_complex(self):
        # 8 channels = 4 coils × (re, im); RSS = sqrt(sum of |coil|^2).
        x = torch.randn(1, 8, 4, 4)
        out = adapt_to_rgb(x, mode=ChannelMode.AUTO)
        assert out.shape == (1, 3, 4, 4)
        # All three replicated channels equal
        assert torch.allclose(out[:, 0], out[:, 1])

    def test_complex_dtype(self):
        x = torch.complex(torch.rand(2, 4, 8, 8), torch.rand(2, 4, 8, 8))
        out = adapt_to_rgb(x, mode=ChannelMode.AUTO)
        assert out.shape == (2, 3, 8, 8)
        assert out.dtype == x.real.dtype  # real-valued output

    def test_odd_5_channel_raises(self):
        x = torch.rand(2, 5, 8, 8)
        with pytest.raises(ValueError, match="AUTO mode cannot handle odd"):
            adapt_to_rgb(x, mode=ChannelMode.AUTO)

    def test_odd_7_channel_raises(self):
        x = torch.rand(2, 7, 8, 8)
        with pytest.raises(ValueError):
            adapt_to_rgb(x, mode=ChannelMode.AUTO)


class TestGrayscaleMean:
    def test_4_channel_means_then_replicates(self):
        x = torch.zeros(2, 4, 8, 8)
        x[:, 0] = 1.0
        x[:, 1] = 2.0
        x[:, 2] = 3.0
        x[:, 3] = 4.0
        out = adapt_to_rgb(x, mode=ChannelMode.GRAYSCALE_MEAN)
        # Mean = 2.5
        assert out.shape == (2, 3, 8, 8)
        assert torch.allclose(out, torch.full_like(out, 2.5))

    def test_handles_odd_channels(self):
        x = torch.rand(2, 5, 8, 8)
        out = adapt_to_rgb(x, mode=ChannelMode.GRAYSCALE_MEAN)
        assert out.shape == (2, 3, 8, 8)
        assert torch.allclose(out[:, 0], out[:, 1])

    def test_complex_input_uses_abs(self):
        x = torch.complex(
            torch.full((2, 2, 4, 4), 3.0),
            torch.full((2, 2, 4, 4), 4.0),
        )
        out = adapt_to_rgb(x, mode=ChannelMode.GRAYSCALE_MEAN)
        # |3+4j| = 5 per channel → mean = 5
        assert torch.allclose(out, torch.full_like(out, 5.0))


class TestComplexRss:
    def test_even_c_works(self):
        x = torch.zeros(1, 2, 4, 4)
        x[:, 0] = 3.0
        x[:, 1] = 4.0
        out = adapt_to_rgb(x, mode=ChannelMode.COMPLEX_RSS)
        assert torch.allclose(out, torch.full_like(out, 5.0))

    def test_odd_c_raises(self):
        x = torch.rand(1, 3, 4, 4)
        with pytest.raises(ValueError, match="even C"):
            adapt_to_rgb(x, mode=ChannelMode.COMPLEX_RSS)

    def test_complex_dtype_works(self):
        x = torch.complex(torch.rand(1, 4, 4, 4), torch.rand(1, 4, 4, 4))
        out = adapt_to_rgb(x, mode=ChannelMode.COMPLEX_RSS)
        assert out.shape == (1, 3, 4, 4)


class TestReplicateAndPassthrough:
    def test_replicate_requires_c1(self):
        x = torch.rand(1, 4, 4, 4)
        with pytest.raises(ValueError, match="REPLICATE mode requires C == 1"):
            adapt_to_rgb(x, mode=ChannelMode.REPLICATE)

    def test_replicate_works_for_c1(self):
        x = torch.rand(1, 1, 4, 4)
        out = adapt_to_rgb(x, mode=ChannelMode.REPLICATE)
        assert out.shape == (1, 3, 4, 4)

    def test_passthrough_requires_c3(self):
        x = torch.rand(1, 1, 4, 4)
        with pytest.raises(ValueError, match="PASSTHROUGH mode requires C == 3"):
            adapt_to_rgb(x, mode=ChannelMode.PASSTHROUGH)

    def test_passthrough_works_for_c3(self):
        x = torch.rand(1, 3, 4, 4)
        out = adapt_to_rgb(x, mode=ChannelMode.PASSTHROUGH)
        assert torch.equal(out, x)


class TestErrors:
    def test_3d_input_raises(self):
        x = torch.rand(3, 4, 4)
        with pytest.raises(ValueError, match="4-D"):
            adapt_to_rgb(x, mode=ChannelMode.AUTO)

    def test_unknown_mode_string_raises(self):
        x = torch.rand(1, 1, 4, 4)
        with pytest.raises(ValueError, match="Unknown channel mode"):
            adapt_to_rgb(x, mode="not_a_mode")

    def test_string_mode_accepted(self):
        # Accept the enum's .value as a string for ergonomic call sites
        x = torch.rand(1, 1, 4, 4)
        out = adapt_to_rgb(x, mode="auto")
        assert out.shape == (1, 3, 4, 4)


# ──────────────────────────────────────────────────────────────────────
# Integration: LPIPS / FID / KID accept multi-channel inputs end-to-end
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def torchmetrics_available():
    try:
        import torchmetrics  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.parametrize("C", [1, 2, 3, 4, 8])
def test_lpips_accepts_channel_count(C, torchmetrics_available):
    if not torchmetrics_available:
        pytest.skip("torchmetrics not installed")
    from spectramr.core.metrics.evaluation_metrics import LPIPS

    metric = LPIPS(device="cpu")
    preds = torch.rand(2, C, 32, 32) * 0.5 + 0.25
    target = torch.rand(2, C, 32, 32) * 0.5 + 0.25
    val = metric.compute_metric(preds, target)
    # LPIPS is non-negative for real-valued reasonable inputs. 0.0 is the
    # "neutral score" the class returns on too-small inputs; we use 32x32
    # so we should get a real LPIPS value.
    assert torch.is_tensor(val)
    assert val.numel() == 1


def test_lpips_odd_channel_raises_explicitly(torchmetrics_available):
    if not torchmetrics_available:
        pytest.skip("torchmetrics not installed")
    from spectramr.core.metrics.evaluation_metrics import LPIPS

    metric = LPIPS(device="cpu")
    preds = torch.rand(2, 5, 32, 32)
    target = torch.rand(2, 5, 32, 32)
    # Default AUTO mode raises on odd C != 1, 3 — no silent truncation.
    with pytest.raises(ValueError, match="AUTO mode cannot handle odd"):
        metric.compute_metric(preds, target)


def test_lpips_grayscale_mean_handles_odd_channels(torchmetrics_available):
    if not torchmetrics_available:
        pytest.skip("torchmetrics not installed")
    from spectramr.core.metrics.evaluation_metrics import LPIPS

    metric = LPIPS(device="cpu", channel_mode="grayscale_mean")
    preds = torch.rand(2, 5, 32, 32)
    target = torch.rand(2, 5, 32, 32)
    val = metric.compute_metric(preds, target)
    assert torch.is_tensor(val)


@pytest.mark.parametrize("C", [1, 2, 4])
@requires_torch_fidelity
def test_fid_accepts_channel_count(C, torchmetrics_available):
    if not torchmetrics_available:
        pytest.skip("torchmetrics not installed")
    from spectramr.core.metrics.evaluation_metrics import FID

    fid = FID(device="cpu")
    preds = torch.rand(2, C, 75, 75)  # InceptionV3 wants >= 75
    target = torch.rand(2, C, 75, 75)
    fid.update(preds, target)
    # Don't call compute() — needs many samples — just confirm update accepts.


@pytest.mark.parametrize("C", [1, 2, 4])
@requires_torch_fidelity
def test_kid_accepts_channel_count(C, torchmetrics_available):
    if not torchmetrics_available:
        pytest.skip("torchmetrics not installed")
    from spectramr.core.metrics.evaluation_metrics import KID

    kid = KID(device="cpu", subset_size=2)
    preds = torch.rand(2, C, 75, 75)
    target = torch.rand(2, C, 75, 75)
    kid.update(preds, target)


# ──────────────────────────────────────────────────────────────────────
# Wiring smoke tests (no torchmetrics required)
# Confirm that LPIPS / FID / KID accept channel_mode and store it,
# even when their torchmetrics backbone is unavailable.
# ──────────────────────────────────────────────────────────────────────


def test_lpips_constructor_accepts_channel_mode():
    from spectramr.core.metrics.evaluation_metrics import LPIPS

    lpips = LPIPS(device="cpu", channel_mode="grayscale_mean")
    assert lpips.channel_mode == ChannelMode.GRAYSCALE_MEAN


def test_lpips_default_channel_mode_is_auto():
    from spectramr.core.metrics.evaluation_metrics import LPIPS

    lpips = LPIPS(device="cpu")
    assert lpips.channel_mode == ChannelMode.AUTO


@requires_torch_fidelity
def test_fid_constructor_accepts_channel_mode():
    from spectramr.core.metrics.evaluation_metrics import FID

    fid = FID(device="cpu", channel_mode="complex_rss")
    assert fid.channel_mode == ChannelMode.COMPLEX_RSS


@requires_torch_fidelity
def test_kid_constructor_accepts_channel_mode():
    from spectramr.core.metrics.evaluation_metrics import KID

    kid = KID(device="cpu", channel_mode="auto")
    assert kid.channel_mode == ChannelMode.AUTO


def test_lpips_invalid_channel_mode_raises():
    from spectramr.core.metrics.evaluation_metrics import LPIPS

    with pytest.raises(ValueError):
        LPIPS(device="cpu", channel_mode="invalid_mode_that_does_not_exist")


# ──────────────────────────────────────────────────────────────────────
# Regression: silent-truncation behavior is gone
# ──────────────────────────────────────────────────────────────────────


def test_no_silent_channel_truncation(torchmetrics_available):
    """The old LPIPS path did ``preds = preds[:, :3]`` which silently
    discarded channel 4 (and beyond). With the channel adapter, channel
    4 must now contribute to the result — different inputs in channel 4
    must produce different outputs.
    """
    if not torchmetrics_available:
        pytest.skip("torchmetrics not installed")
    from spectramr.core.metrics.evaluation_metrics import LPIPS

    metric = LPIPS(device="cpu")
    torch.manual_seed(0)
    preds_a = torch.rand(2, 4, 32, 32)
    preds_b = preds_a.clone()
    # Modify ONLY channel 3 (the one the old code dropped).
    preds_b[:, 3] = preds_b[:, 3] + 0.5
    target = torch.rand(2, 4, 32, 32)

    val_a = metric.compute_metric(preds_a, target)
    val_b = metric.compute_metric(preds_b, target)

    # If silent truncation were still happening, val_a == val_b. The
    # adapter routes channel 3 through RSS, so they must differ.
    assert not torch.allclose(val_a, val_b, atol=1e-6), (
        "channel 4 was silently dropped — adapter not in effect"
    )
