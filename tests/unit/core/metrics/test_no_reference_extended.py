"""Tests for the extended no-reference artifact-detection metrics.

These grayscale NR metrics detect specific degradations *without* a clean
reference — the use case the 2026-05-25 metric expansion targeted (BRISQUE /
NIQE-style blind IQA plus classical computer-vision focus/noise/artifact
measures). Property tests assert each metric moves the expected way under the
artifact it is designed to detect.

References: Brenner (1971) focus measure; Immerkaer (1996) fast noise variance
estimation; Bahrami & Kot (2014) Maximum Local Variation; Wang et al. (2000)
blocking-artifact measure; Gibbs ringing from Fourier truncation.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from mriforge.core.metrics.registry import MetricsRegistry


def _disk(h=96, w=96):
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
    )
    return (torch.sqrt(xx**2 + yy**2) < 0.5).float().unsqueeze(0).unsqueeze(0)


def _blur(img, k=7):
    import torch.nn.functional as F

    kernel = torch.ones(1, 1, k, k) / (k * k)
    return F.conv2d(img, kernel, padding=k // 2)


def _gibbs(img):
    """Induce Gibbs ringing by truncating the image's Fourier spectrum."""
    f = torch.fft.fftshift(torch.fft.fft2(img))
    h, w = img.shape[-2:]
    mask = torch.zeros(h, w)
    cy, cx = h // 2, w // 2
    r = h // 6
    mask[cy - r : cy + r, cx - r : cx + r] = 1.0
    return torch.fft.ifft2(torch.fft.ifftshift(f * mask)).abs()


def _blocky(img, block=8):
    """Quantize into block-mean tiles to create a blocking grid artifact."""
    out = img.clone()
    h, w = img.shape[-2:]
    for i in range(0, h, block):
        for j in range(0, w, block):
            out[..., i : i + block, j : j + block] = img[
                ..., i : i + block, j : j + block
            ].mean()
    return out


def test_brenner_focus_higher_for_sharp():
    m = MetricsRegistry.get("brenner_focus")
    sharp, blurred = _disk(), _blur(_disk())
    v_s, v_b = float(m(sharp, sharp)), float(m(blurred, blurred))
    assert math.isfinite(v_s) and v_s >= 0
    assert v_s > v_b, f"Brenner should be higher for sharp: {v_s} vs {v_b}"


def test_immerkaer_noise_higher_for_noisy():
    m = MetricsRegistry.get("immerkaer_noise")
    clean = _disk()
    noisy = (clean + 0.1 * torch.randn_like(clean)).clamp(0, None)
    v_c, v_n = float(m(clean, clean)), float(m(noisy, noisy))
    assert math.isfinite(v_n) and v_n >= 0
    assert v_n > v_c, f"Immerkaer noise should rise with noise: {v_c} vs {v_n}"


def test_mlv_higher_for_sharp():
    m = MetricsRegistry.get("mlv")
    sharp, blurred = _disk(), _blur(_disk())
    assert float(m(sharp, sharp)) > float(m(blurred, blurred))


def test_intensity_entropy_higher_for_textured():
    m = MetricsRegistry.get("intensity_entropy")
    uniform = torch.full((1, 1, 64, 64), 0.5)
    textured = torch.rand(1, 1, 64, 64)
    v_u, v_t = float(m(uniform, uniform)), float(m(textured, textured))
    assert math.isfinite(v_t) and v_t >= 0
    assert v_t > v_u


def test_blockiness_higher_for_blocky():
    m = MetricsRegistry.get("blockiness")
    smooth = _blur(_disk())
    blocky = _blocky(_disk(), block=8)
    assert float(m(blocky, blocky)) > float(m(smooth, smooth))


def test_high_freq_energy_ratio_higher_for_sharp():
    m = MetricsRegistry.get("high_freq_energy_ratio")
    sharp, blurred = _disk(), _blur(_disk())
    v_s, v_b = float(m(sharp, sharp)), float(m(blurred, blurred))
    assert math.isfinite(v_s) and 0 <= v_s <= 1
    assert v_s > v_b


def test_gibbs_ringing_higher_for_truncated():
    m = MetricsRegistry.get("gibbs_ringing")
    clean = _blur(_disk(), k=5)  # smooth, no ringing
    gibbs = _gibbs(_disk())  # Fourier-truncated → ringing
    v_c, v_g = float(m(clean, clean)), float(m(gibbs, gibbs))
    assert math.isfinite(v_g) and v_g >= 0
    assert v_g > v_c, f"Gibbs ringing metric should flag truncation: {v_c} vs {v_g}"


def test_nr_metrics_handle_complex_and_multichannel():
    """All new NR metrics collapse complex / multi-channel input to grayscale."""
    for key in [
        "brenner_focus", "immerkaer_noise", "mlv", "intensity_entropy",
        "blockiness", "high_freq_energy_ratio", "gibbs_ringing",
    ]:
        m = MetricsRegistry.get(key)
        cplx = torch.complex(torch.rand(1, 1, 48, 48), torch.rand(1, 1, 48, 48))
        multi = torch.rand(1, 4, 48, 48)
        assert math.isfinite(float(m(cplx, cplx))), f"{key} failed on complex"
        assert math.isfinite(float(m(multi, multi))), f"{key} failed on multichannel"
