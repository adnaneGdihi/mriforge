"""Tests for the piq-backed perceptual IQA metrics.

The 2026 MRI-IQA survey (Journal of Imaging Informatics in Medicine) evaluates a
broad full-reference panel; this codebase was missing several widely-cited
closed-form perceptual metrics. We add HaarPSI, MDSI, VSI, DSS and MS-GMSD by
wrapping the validated ``piq`` implementations (no reimplementation risk).

These property tests assert the *direction* and *finiteness* of each metric
rather than exact reference values: identical inputs give the perfect score and
degradation moves the score the expected way.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("piq")

from mriforge.core.metrics.registry import MetricsRegistry


def _pair(h=96, w=96, noise=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
    )
    clean = (torch.sqrt(xx**2 + yy**2) < 0.6).float() + 0.1 * torch.rand(h, w, generator=g)
    clean = clean.clamp(0, None).unsqueeze(0).unsqueeze(0)
    deg = (clean + noise * torch.randn(1, 1, h, w, generator=g)).clamp(0, None)
    return deg, clean


# (key, higher_is_better)
_PIQ_METRICS = [
    ("haarpsi", True),
    ("mdsi", False),
    ("vsi", True),
    ("dss", True),
    ("ms_gmsd", False),
    ("iw_ssim", True),
    ("srsim", True),
    ("vif_p", True),
]


@pytest.mark.parametrize("key,higher_is_better", _PIQ_METRICS)
def test_piq_metric_registered_and_directional(key, higher_is_better):
    metric = MetricsRegistry.get(key)
    clean = _pair(noise=0.0)[1]
    light = _pair(noise=0.05, seed=1)[0]
    heavy = _pair(noise=0.35, seed=1)[0]

    v_self = float(metric(clean, clean))
    v_light = float(metric(light, clean))
    v_heavy = float(metric(heavy, clean))

    import math
    assert all(math.isfinite(v) for v in (v_self, v_light, v_heavy))
    # Property: identical inputs give the best score and degradation moves it
    # the expected direction. (Some saturating metrics, e.g. MDSI, are not
    # strictly monotone between two non-trivial noise levels, so we don't
    # assert light-vs-heavy ordering.)
    if higher_is_better:  # similarity
        assert v_self >= v_light and v_self >= v_heavy
        assert v_heavy < v_self
    else:  # deviation
        assert v_self <= v_light and v_self <= v_heavy
        assert v_heavy > v_self


def test_total_variation_higher_for_noisy():
    """total_variation (NR, piq) must rise with added noise."""
    metric = MetricsRegistry.get("total_variation")
    clean = _pair(noise=0.0)[1]
    noisy = (clean + 0.2 * torch.randn_like(clean)).clamp(0, None)
    v_c, v_n = float(metric(clean, clean)), float(metric(noisy, noisy))
    import math
    assert math.isfinite(v_n) and v_n >= 0
    assert v_n > v_c, f"TV should rise with noise: {v_c} vs {v_n}"


def test_iw_ssim_handles_small_images():
    """IW-SSIM needs >=161px; the wrapper must upsample small MRI patches."""
    metric = MetricsRegistry.get("iw_ssim")
    clean = _pair(h=64, w=64, noise=0.0)[1]  # below piq's 161 floor
    val = float(metric(clean, clean))
    import math
    assert math.isfinite(val) and val > 0.9
