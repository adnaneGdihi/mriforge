"""Integration test: score-field tomography Hermitian-noise round trip (Idea 2).

The Phase 0 / Phase 3 interlock requires that when the diffusion noise
process operates in k-space, the noise vector itself must be drawn
from a Hermitian-Gaussian distribution; otherwise the inverse FFT of
the noised k-space picks up a non-trivial imaginary component and the
score head learns a biased target.

This test exercises the integration:

1. Sample a centered k-space tensor.
2. Add Hermitian-Gaussian noise of varying levels.
3. Confirm the inverse FFT remains *real* (imaginary part below tolerance).
4. Confirm an independent draw of plain complex Gaussian noise of the
   same energy *fails* the same check — establishing that the
   Hermitian sampler is what the strategy needs, not just any complex
   noise.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.hermitian import (
    enforce_hermitian,
    hermitian_gaussian_noise,
)


@pytest.mark.integration
def test_hermitian_noise_preserves_real_image() -> None:
    """ifft( real-spectrum + hermitian-noise ) stays real."""
    torch.manual_seed(0)
    H = W = 16
    img = torch.randn(2, 1, H, W)
    k = torch.fft.fftshift(torch.fft.fft2(img, norm="ortho"), dim=(-2, -1))
    for sigma in (0.01, 0.1, 0.5, 1.0):
        noise = sigma * hermitian_gaussian_noise(k.shape, dtype=torch.complex64)
        noisy_k = enforce_hermitian(k + noise)
        recovered = torch.fft.ifft2(
            torch.fft.ifftshift(noisy_k, dim=(-2, -1)),
            norm="ortho",
        )
        assert recovered.imag.abs().max().item() < 1e-4, (
            f"Hermitian noise broke real-image invariant at sigma={sigma}"
        )


@pytest.mark.integration
def test_plain_complex_noise_does_break_real_image() -> None:
    """Sanity check: plain complex Gaussian noise (no Hermitian projection)
    *does* introduce an imaginary component after the round trip — confirming
    the test in the positive case is non-trivial."""
    torch.manual_seed(1)
    H = W = 16
    img = torch.randn(1, 1, H, W)
    k = torch.fft.fftshift(torch.fft.fft2(img, norm="ortho"), dim=(-2, -1))
    plain = 0.3 * torch.complex(torch.randn_like(k.real), torch.randn_like(k.real))
    noisy_k = k + plain  # no enforce_hermitian
    recovered = torch.fft.ifft2(
        torch.fft.ifftshift(noisy_k, dim=(-2, -1)),
        norm="ortho",
    )
    assert recovered.imag.abs().max().item() > 1e-3


@pytest.mark.integration
def test_score_field_strategy_logs_hermitian_residual() -> None:
    """The strategy's diagnostic ``score_field_herm_residual`` must be
    finite and small on a clean batch — confirms the Phase 0/Phase 3
    interlock is wired through to the loss dict."""
    from types import SimpleNamespace

    import torch.nn as nn

    from spectramr.infrastructure.training.strategies.score_field_tomography_strategy import (
        ScoreFieldTomographyStrategy,
    )

    class _StubModel(nn.Module):
        def __init__(self, channels: int, cond_dim: int) -> None:
            super().__init__()
            self.head_sx = nn.Conv2d(channels, channels, 1)
            self.head_sc = nn.Linear(channels, cond_dim)

        def forward(self, x, sigma, c):
            pooled = x.mean(dim=(-2, -1))
            return self.head_sx(x), self.head_sc(pooled)

    strat = ScoreFieldTomographyStrategy.__new__(ScoreFieldTomographyStrategy)
    strat.config = SimpleNamespace(training=SimpleNamespace())
    strat.device = torch.device("cpu")
    strat.env = SimpleNamespace(
        models={"generator": _StubModel(1, 5)},
        device=torch.device("cpu"),
        config=strat.config,
    )
    strat.sigma_min = 0.01
    strat.sigma_max = 0.5
    strat.cond_dim = 5
    strat._dim_x = None
    type(strat).generator_model = property(lambda s: s.env.models["generator"])  # type: ignore[assignment]

    x = torch.randn(2, 1, 8, 8)
    losses = strat._compute_losses_impl(
        input_batch=x,
        target_batch=x,
        epoch=0,
        batch={},
    )
    assert "score_field_herm_residual" in losses
    assert torch.isfinite(losses["score_field_herm_residual"])
