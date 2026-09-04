"""Unit tests for the ``nr_prior_geometry`` learned-prior NR metric group.

CPU-only, deterministic, fast. Each metric is exercised with a trivial
*analytic* Gaussian prior whose score is :math:`s(x)=-(x-\\mu)/(\\sigma^2+t^2)`
so the real formula path runs, plus the graceful no-asset ``nan`` path.

Anchors checked:
* PMMD — monotone increase under mean-shift degradation; nan without a fitted
  Gaussian; nan without an encoder.
* DPST — small near the prior mean, rises with atypicality (monotone).
* FITC — Hutchinson trace estimator unbiased on a known matrix; positive-large
  (~D/var) at a clean image for the Gaussian-prior score.
* KSDP — near-zero when the patch distribution matches the prior; rises with
  mean-shift.
* PFLD — lower for clean than degraded via both the ``log_prob`` short-circuit
  and the score-only short-integration ODE path.
* HallucinationIndex — rises when structure is injected into the forward-model
  null space; nan without the prior or the measurement context.
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.core.metrics import get_metric
from spectramr.core.metrics.context import MetricContext
from spectramr.infrastructure.physics.fft_ops import sense_adjoint, sense_forward


# --------------------------------------------------------------------------- #
# Stub assets
# --------------------------------------------------------------------------- #
class GaussianPrior:
    """Analytic isotropic-Gaussian prior with an exact score + log-density.

    score(x, t) = -(x - mu) / (sigma^2 + t^2)  (perturbed-prior convention)
    log_prob(x) = log N(x; mu, sigma^2 I)      (per-sample, shape [B])
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None:
        self.mu = mu
        self.sigma = sigma

    def score(self, x: torch.Tensor, t: float = 0.0) -> torch.Tensor:
        var = self.sigma**2 + float(t) ** 2
        return -(x - self.mu) / var

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        var = self.sigma**2
        b = x.shape[0]
        xf = x.reshape(b, -1)
        d = xf.shape[1]
        const = 0.5 * d * math.log(2.0 * math.pi * var)
        return -0.5 * ((xf - self.mu) ** 2).sum(dim=1) / var - const


class ScoreOnlyGaussian:
    """Gaussian prior that exposes ONLY a score (no log_prob) — exercises the
    PFLD short-integration ODE path rather than the log_prob short-circuit."""

    def score(self, x: torch.Tensor, t: float = 0.0) -> torch.Tensor:
        var = 1.0 + float(t) ** 2
        return -x / var


class DenoiserPrior:
    """Tweedie denoiser prior: D(x, sigma) shrinks toward 0 — the adapter must
    recover the score s = (D - x)/sigma^2 = -x/(1+sigma^2)·... (we just check it
    runs and is monotone)."""

    def denoise(self, x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
        # Optimal Gaussian-prior (mu=0, sigma_x=1) denoiser: D = x / (1 + sigma^2)
        return x / (1.0 + float(sigma) ** 2)


class StubEncoder:
    """Deterministic feature encoder: adaptive-avg-pool the magnitude to D=16."""

    def __init__(self) -> None:
        self.d = 16

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        f = torch.nn.functional.adaptive_avg_pool2d(x.mean(1, keepdim=True), (4, 4))
        return f.reshape(b, -1)


def _spearman_sign(values: list[float]) -> float:
    """Sign of Spearman rho between [0,1,2,...] severities and ``values``."""
    n = len(values)
    ranks_x = list(range(n))
    order = sorted(range(n), key=lambda i: values[i])
    ranks_y = [0] * n
    for rank, i in enumerate(order):
        ranks_y[i] = rank
    dx = [r - (n - 1) / 2 for r in ranks_x]
    dy = [r - (n - 1) / 2 for r in ranks_y]
    num = sum(a * b for a, b in zip(dx, dy, strict=False))
    den = math.sqrt(sum(a * a for a in dx) * sum(b * b for b in dy)) + 1e-12
    return num / den


# =========================================================================== #
# Registration / contract
# =========================================================================== #
@pytest.mark.parametrize(
    "key,hib",
    [
        ("pmmd", False),
        ("dpst", False),
        ("fitc", True),
        ("ksdp", False),
        ("pfld", False),
        ("hallucination_index", False),
    ],
)
def test_registered_with_direction(key: str, hib: bool) -> None:
    m = get_metric(key)
    assert m.higher_is_better is hib
    assert m.name == key


def test_import_fires_all_registrations() -> None:
    import spectramr.core.metrics.nr_prior_geometry as mod  # noqa: F401
    from spectramr.core.metrics.registry import MetricsRegistry

    for key in ("pmmd", "dpst", "fitc", "ksdp", "pfld", "hallucination_index"):
        assert MetricsRegistry.is_registered(key)
        assert MetricsRegistry.requires_reference(key) is False


# =========================================================================== #
# Graceful no-asset path (must NOT raise — return nan)
# =========================================================================== #
@pytest.mark.parametrize(
    "key", ["pmmd", "dpst", "fitc", "ksdp", "pfld", "hallucination_index"]
)
def test_no_asset_returns_nan(key: str) -> None:
    x = torch.randn(1, 1, 8, 8)
    # Empty context: no encoder, no prior, no measurement.
    val = get_metric(key)(x, context=MetricContext())
    assert math.isnan(val)


def test_prior_without_supported_hook_returns_nan() -> None:
    class Useless:  # exposes none of score/denoise/log_prob
        pass

    x = torch.randn(1, 1, 8, 8)
    val = get_metric("dpst")(x, context=MetricContext(prior_model=Useless()))
    assert math.isnan(val)


# =========================================================================== #
# pmmd
# =========================================================================== #
def test_pmmd_monotone_increase() -> None:
    enc = StubEncoder()
    ctx_params = {"mu": torch.zeros(16), "cov": torch.eye(16)}
    vals = []
    for shift in (0.0, 0.5, 1.0, 1.5):
        torch.manual_seed(0)
        x = shift * torch.ones(1, 1, 8, 8) + 0.01 * torch.randn(1, 1, 8, 8)
        vals.append(
            get_metric("pmmd")(
                x, context=MetricContext(encoder=enc, nr_params=ctx_params)
            )
        )
    assert _spearman_sign(vals) >= 0.6  # ↓ metric, degradation raises it
    assert vals[0] < vals[-1]


def test_pmmd_clean_near_zero() -> None:
    enc = StubEncoder()
    x = 0.001 * torch.randn(1, 1, 8, 8)
    val = get_metric("pmmd")(
        x, context=MetricContext(encoder=enc, nr_params={"mu": torch.zeros(16), "cov": torch.eye(16)})
    )
    assert val == pytest.approx(0.0, abs=0.5)


def test_pmmd_no_fitted_stats_returns_nan() -> None:
    enc = StubEncoder()
    val = get_metric("pmmd")(torch.zeros(1, 1, 8, 8), context=MetricContext(encoder=enc))
    assert math.isnan(val)


# =========================================================================== #
# dpst
# =========================================================================== #
def test_dpst_monotone_and_clean_small() -> None:
    prior = GaussianPrior(0.0, 1.0)
    vals = []
    for shift in (0.0, 1.0, 2.0, 3.0):
        torch.manual_seed(1)
        x = shift * torch.ones(1, 1, 8, 8) + 0.01 * torch.randn(1, 1, 8, 8)
        vals.append(get_metric("dpst")(x, context=MetricContext(prior_model=prior)))
    assert _spearman_sign(vals) >= 0.6
    assert vals[0] == pytest.approx(0.0, abs=1e-2)  # typical ⇒ ~0


def test_dpst_denoiser_adapter_runs() -> None:
    prior = DenoiserPrior()
    x_typ = 0.01 * torch.randn(1, 1, 8, 8)
    x_atyp = 3.0 * torch.ones(1, 1, 8, 8)
    v_typ = get_metric("dpst")(x_typ, context=MetricContext(prior_model=prior))
    v_atyp = get_metric("dpst")(x_atyp, context=MetricContext(prior_model=prior))
    assert v_atyp > v_typ


# =========================================================================== #
# fitc
# =========================================================================== #
def test_fitc_hutchinson_unbiased_on_known_matrix() -> None:
    # E[eta^T B eta] = tr(B) for eta ~ N(0, I).
    torch.manual_seed(0)
    d = 12
    a = torch.randn(d, d)
    b = a @ a.T  # SPD
    eta = torch.randn(20000, d)
    est = ((eta @ b) * eta).sum(dim=1).mean()
    assert est.item() == pytest.approx(torch.trace(b).item(), rel=0.05)


def test_fitc_positive_large_at_clean() -> None:
    # For s(x) = -x/var, d s/dx = -(1/var) I, tr = -D/var, FITC = -tr = +D/var.
    prior = GaussianPrior(0.0, 1.0)
    gen = torch.Generator().manual_seed(0)
    x = torch.zeros(1, 1, 8, 8)
    var = 1.0 + 0.1**2  # default t=0.1
    val = get_metric("fitc")(x, context=MetricContext(prior_model=prior, generator=gen))
    expected = 64.0 / var
    assert val > 0
    assert val == pytest.approx(expected, rel=0.15)


# =========================================================================== #
# ksdp
# =========================================================================== #
def test_ksdp_zero_when_matched_rises_with_shift() -> None:
    prior = GaussianPrior(0.0, 1.0)
    vals = []
    for shift in (0.0, 2.0, 5.0):
        g = torch.Generator().manual_seed(7)
        x = shift + torch.randn(1, 1, 16, 16, generator=g)
        vals.append(
            get_metric("ksdp")(
                x,
                context=MetricContext(
                    prior_model=prior, generator=torch.Generator().manual_seed(3)
                ),
            )
        )
    assert _spearman_sign(vals) >= 0.6
    # Matched (samples ~ N(0,1)) ⇒ KSD^2 small relative to the shifted values.
    assert vals[0] < 0.1 * vals[-1]


# =========================================================================== #
# pfld
# =========================================================================== #
def test_pfld_logprob_shortcircuit_monotone() -> None:
    prior = GaussianPrior(0.0, 1.0)
    vals = []
    for shift in (0.0, 2.0, 4.0):
        x = shift * torch.ones(1, 1, 8, 8)
        vals.append(
            get_metric("pfld")(
                x,
                context=MetricContext(
                    prior_model=prior, generator=torch.Generator().manual_seed(0)
                ),
            )
        )
    assert _spearman_sign(vals) >= 0.6
    assert vals[0] < vals[-1]  # clean ⇒ lower deficit


def test_pfld_score_only_ode_path_monotone() -> None:
    prior = ScoreOnlyGaussian()
    v_clean = get_metric("pfld")(
        torch.zeros(1, 1, 8, 8),
        context=MetricContext(prior_model=prior, generator=torch.Generator().manual_seed(0)),
    )
    v_deg = get_metric("pfld")(
        3.0 * torch.ones(1, 1, 8, 8),
        context=MetricContext(prior_model=prior, generator=torch.Generator().manual_seed(0)),
    )
    assert v_deg > v_clean


# =========================================================================== #
# hallucination_index
# =========================================================================== #
def _hi_context(prior, smaps, mask, y):
    return MetricContext(prior_model=prior, y_kspace=y, mask=mask, coil_maps=smaps)


def test_hallucination_index_rises_with_null_space_injection() -> None:
    torch.manual_seed(3)
    h = w = 16
    smaps = torch.randn(1, 4, h, w, dtype=torch.complex64)
    mask = torch.zeros(1, 1, h, w)
    mask[..., ::2] = 1.0  # 2x column undersampling ⇒ a real null space
    clean = torch.randn(1, 1, h, w)
    y = sense_forward(clean, smaps, mask)
    prior = ScoreOnlyGaussian()

    hi = get_metric("hallucination_index")
    base = hi(clean, context=_hi_context(prior, smaps, mask, y))

    # Inject structure purely into the null space (orthogonal to the range of A).
    noise = torch.randn(1, 1, h, w, dtype=torch.complex64)
    a_noise = sense_forward(noise, smaps, mask)
    aha_noise = sense_adjoint(a_noise, smaps, mask)
    noise_null = noise - aha_noise
    injected = clean.to(torch.complex64) + 3.0 * noise_null
    injected_val = hi(injected, context=_hi_context(prior, smaps, mask, y))

    assert injected_val > base


def test_hallucination_index_missing_measurement_returns_nan() -> None:
    prior = ScoreOnlyGaussian()
    x = torch.randn(1, 1, 16, 16)
    # Prior present but no y_kspace/mask/coil_maps ⇒ asset-blocked ⇒ nan.
    val = get_metric("hallucination_index")(x, context=MetricContext(prior_model=prior))
    assert math.isnan(val)


def test_hallucination_index_missing_prior_returns_nan() -> None:
    torch.manual_seed(0)
    h = w = 16
    smaps = torch.randn(1, 4, h, w, dtype=torch.complex64)
    mask = torch.ones(1, 1, h, w)
    y = sense_forward(torch.randn(1, 1, h, w), smaps, mask)
    val = get_metric("hallucination_index")(
        torch.randn(1, 1, h, w),
        context=MetricContext(y_kspace=y, mask=mask, coil_maps=smaps),
    )
    assert math.isnan(val)
