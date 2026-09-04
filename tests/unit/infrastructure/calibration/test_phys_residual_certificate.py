"""Unit tests for the PR-CC physics-residual conformal certificate.

Targets ``spectramr.infrastructure.calibration.phys_residual_certificate``.

PR-CC calibrates a distribution-free hallucination certificate on the
*physics-consistency residual* (re-render the observed contrast through the
Bloch forward, score = ``|render - observed|``). The properties checked here are
the scientific contract:

- **Marginal coverage.** On exchangeable calibration/test residuals the
  empirical coverage of the flagged-consistent set is ``>= 1 - alpha``.
- **Hallucination flagging.** An injected high-residual perturbation lands in
  the flagged (``s > q``) set while clean tissue stays covered.
- **Mondrian (field-stratified) calibration.** When the residual scale differs
  by field (low SNR at ultra-low field), pooled calibration under-covers the
  noisy field; per-``B0`` calibration restores per-group coverage. This is the
  PR-CC contribution: anchoring the score to a field-invariant forward model and
  fixing the residual-shift case with Mondrian stratification.
- **Guards.** Unfit certify and unseen-field certify raise; ``alpha`` is validated.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit

_ACQ = {"TR": 3000.0, "TE": 90.0, "TI": 0.0}


def _params(h: int = 16, w: int = 16) -> torch.Tensor:
    """A ``[1, 3, h, w]`` ``(rho, T1_ms, T2_ms)`` map with smooth per-voxel contrast."""
    yy, xx = torch.meshgrid(
        torch.linspace(0.4, 1.0, h), torch.linspace(0.5, 1.0, w), indexing="ij"
    )
    rho = (0.6 + 0.4 * xx).unsqueeze(0).unsqueeze(0)
    t1 = (600.0 + 800.0 * yy).unsqueeze(0).unsqueeze(0)
    t2 = (60.0 + 80.0 * xx).unsqueeze(0).unsqueeze(0)
    return torch.cat([rho, t1, t2], dim=1)


def _clean_render(params: torch.Tensor) -> torch.Tensor:
    from spectramr.infrastructure.physics.multi_physics_bloch import (
        MultiPhysicsBlochLayer,
    )

    layer = MultiPhysicsBlochLayer(learnable_flip_angle=False)
    return layer(
        {"rho": params[:, 0:1], "t1": params[:, 1:2], "t2": params[:, 2:3]},
        {"TR": 3000.0, "TE": 90.0, "TI": 0.0, "contrast_type": 0},
    )


def test_marginal_coverage_meets_target() -> None:
    from spectramr.infrastructure.calibration.phys_residual_certificate import (
        PhysResidualConformalCertificate,
    )

    torch.manual_seed(0)
    params = _params()
    clean = _clean_render(params)
    sigma = 0.05

    cal_units = [
        (params, clean + sigma * torch.randn_like(clean), None) for _ in range(40)
    ]
    cert = PhysResidualConformalCertificate(
        alpha=0.1, acquisition=_ACQ, contrast_type="spin_echo"
    )
    cert.calibrate(cal_units)

    covs = []
    for _ in range(40):
        res = cert.certify(params, clean + sigma * torch.randn_like(clean))
        covs.append(res.coverage)
    mean_cov = sum(covs) / len(covs)
    assert mean_cov >= 0.88, f"marginal coverage {mean_cov:.3f} < 1 - alpha"


def test_flags_injected_hallucination() -> None:
    from spectramr.infrastructure.calibration.phys_residual_certificate import (
        PhysResidualConformalCertificate,
    )

    torch.manual_seed(1)
    params = _params()
    clean = _clean_render(params)
    sigma = 0.03

    cal_units = [
        (params, clean + sigma * torch.randn_like(clean), None) for _ in range(40)
    ]
    cert = PhysResidualConformalCertificate(
        alpha=0.1, acquisition=_ACQ, contrast_type="spin_echo"
    )
    cert.calibrate(cal_units)

    corrupted = clean.clone()
    corrupted[..., :4, :4] += 0.6  # injected "hallucination" block
    res = cert.certify(params, corrupted)

    assert res.flagged[..., :4, :4].all(), "injected perturbation must be flagged"
    clean_region = res.flagged[..., 8:, 8:].float().mean().item()
    assert clean_region < 0.2, f"clean region over-flagged: {clean_region:.3f}"


def test_mondrian_restores_per_field_coverage() -> None:
    from spectramr.infrastructure.calibration.phys_residual_certificate import (
        PhysResidualConformalCertificate,
    )

    torch.manual_seed(2)
    params = _params()
    clean = _clean_render(params)
    # Ultra-low field: low SNR -> large residual. High field: small residual.
    b0_ulf, sigma_ulf = 0.064, 0.16
    b0_hf, sigma_hf = 3.0, 0.02

    def units(b0: float, sigma: float, n: int) -> list:
        return [(params, clean + sigma * torch.randn_like(clean), b0) for _ in range(n)]

    cal = units(b0_ulf, sigma_ulf, 40) + units(b0_hf, sigma_hf, 40)

    marginal = PhysResidualConformalCertificate(
        alpha=0.1, acquisition=_ACQ, contrast_type="spin_echo", stratify_by=None
    )
    marginal.calibrate(cal)

    mondrian = PhysResidualConformalCertificate(
        alpha=0.1, acquisition=_ACQ, contrast_type="spin_echo", stratify_by="b0"
    )
    mondrian.calibrate(cal)

    # The pooled quantile is dominated by the large ULF residuals, so it is too
    # wide for the high field (over-covers, ~1.0) and too narrow for the noisy
    # ultra-low field (UNDER-covers -- the dangerous case, hallucinations slip
    # through). Mondrian gives each field its own quantile.
    ulf_test = units(b0_ulf, sigma_ulf, 40)
    marg_cov_ulf = sum(
        marginal.certify(p, t).coverage for p, t, _ in ulf_test
    ) / len(ulf_test)
    mond_cov_ulf = sum(
        mondrian.certify(p, t, b0=b0_ulf).coverage for p, t, _ in ulf_test
    ) / len(ulf_test)

    assert mond_cov_ulf >= 0.88, (
        f"Mondrian must restore ULF coverage to >= 1 - alpha: {mond_cov_ulf:.3f}"
    )
    assert marg_cov_ulf < 0.86, (
        f"pooled calibration should under-cover the noisy ultra-low field; "
        f"got marginal {marg_cov_ulf:.3f} vs Mondrian {mond_cov_ulf:.3f}"
    )


def test_certify_before_calibrate_raises() -> None:
    from spectramr.infrastructure.calibration.phys_residual_certificate import (
        PhysResidualConformalCertificate,
    )

    params = _params()
    clean = _clean_render(params)
    cert = PhysResidualConformalCertificate(alpha=0.1, acquisition=_ACQ)
    with pytest.raises((RuntimeError, KeyError)):
        cert.certify(params, clean)


def test_unseen_field_raises_under_mondrian() -> None:
    from spectramr.infrastructure.calibration.phys_residual_certificate import (
        PhysResidualConformalCertificate,
    )

    torch.manual_seed(3)
    params = _params()
    clean = _clean_render(params)
    cert = PhysResidualConformalCertificate(
        alpha=0.1, acquisition=_ACQ, stratify_by="b0"
    )
    cert.calibrate([(params, clean + 0.05 * torch.randn_like(clean), 3.0) for _ in range(20)])
    with pytest.raises(KeyError):
        cert.certify(params, clean, b0=7.0)  # never calibrated at 7T


def test_invalid_alpha_raises() -> None:
    from spectramr.infrastructure.calibration.phys_residual_certificate import (
        PhysResidualConformalCertificate,
    )

    with pytest.raises(ValueError):
        PhysResidualConformalCertificate(alpha=1.5, acquisition=_ACQ)
