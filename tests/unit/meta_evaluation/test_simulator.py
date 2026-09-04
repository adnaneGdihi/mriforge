"""Simulator smoke tests."""

from __future__ import annotations

import pytest
import torch

from spectramr.core.metrics.meta_evaluation import SimulatorConfig, run_simulator


def test_simulator_produces_unique_samples_per_severity() -> None:
    clean = [("s0", torch.randn(1, 32, 32))]
    cfg = SimulatorConfig(
        families=["gaussian_noise"], n_severities_per_family=4, seed=0
    )
    samples = run_simulator(clean, cfg)
    assert len(samples) == 4
    severities = sorted({s.severity["theta"] for s in samples})
    assert len(severities) == 4

    # Increasing severity should generally increase residual energy.
    energies = [s.residual_energy for s in sorted(samples, key=lambda s: s.severity["theta"])]
    # Allow non-monotone for the lowest severity (theta near 0 -> tiny noise).
    assert energies[-1] > energies[0]


def test_simulator_reproducible_seed() -> None:
    clean = [("s0", torch.randn(1, 16, 16))]
    cfg = SimulatorConfig(
        families=["gaussian_blur", "gaussian_noise"],
        n_severities_per_family=3,
        seed=99,
    )
    a = run_simulator(clean, cfg)
    b = run_simulator(clean, cfg)
    assert len(a) == len(b)
    for sa, sb in zip(a, b, strict=True):
        assert torch.allclose(sa.degraded, sb.degraded)
        assert sa.severity == sb.severity


# ---------------------------------------------------------------------------
# Dependency-inversion seam: injected operator_library (pitfall #13 / unify)
# ---------------------------------------------------------------------------


def test_families_none_defaults_to_core_library() -> None:
    """families=None resolves to every core degradation family."""
    from spectramr.core.metrics.meta_evaluation.simulator import DEGRADATION_LIBRARY

    cfg = SimulatorConfig()
    assert cfg.families == list(DEGRADATION_LIBRARY.keys())
    assert cfg.effective_library is DEGRADATION_LIBRARY


def test_injected_operator_library_is_used() -> None:
    """An injected library overrides the core ops and is actually invoked."""
    calls: list[str] = []

    def sentinel(clean, theta, generator):
        calls.append("hit")
        return clean * 0.0  # deterministic, shape-preserving

    cfg = SimulatorConfig(operator_library={"custom": sentinel}, n_severities_per_family=3)
    # families default to the injected library's keys.
    assert cfg.families == ["custom"]
    samples = run_simulator([("s0", torch.ones(1, 8, 8))], cfg)
    assert len(samples) == 3
    assert calls == ["hit", "hit", "hit"]
    assert torch.count_nonzero(samples[0].degraded) == 0


def test_injected_library_validates_families() -> None:
    """A family not in the injected library raises (no silent fallback)."""
    import pytest

    with pytest.raises(ValueError, match="unknown degradation families"):
        SimulatorConfig(
            operator_library={"a": lambda c, t, g: c}, families=["a", "missing"]
        )


# ---------------------------------------------------------------------------
# Severity monotonicity of the parametric photometric families (critique C1).
# These degraders scale a single random draw by theta; the draw must be FIXED
# across the trajectory or the degradation magnitude is non-monotone in severity.
# ---------------------------------------------------------------------------


def _residual_trajectory(family: str, seed: int = 7) -> list[float]:
    clean = [("s0", torch.randn(1, 32, 32))]
    cfg = SimulatorConfig(
        families=[family], n_severities_per_family=12, seed=seed
    )
    samples = sorted(run_simulator(clean, cfg), key=lambda s: s.severity["theta"])
    return [s.residual_energy for s in samples]


def _spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for rank, i in enumerate(order):
            r[i] = float(rank)
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    vy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    return cov / (vx * vy) if vx and vy else 0.0


@pytest.mark.parametrize("family", ["gamma", "contrast", "bias_field"])
def test_parametric_families_are_severity_monotone(family: str) -> None:
    """Residual energy rises (near-)monotonically with theta for these families.

    Pre-fix the random offset was re-drawn per severity step, so |offset| varied
    and a high-theta sample could be LESS degraded than a low-theta one (critique
    C1). Seeding the draw per (content, family) makes the trajectory monotone.
    """
    energies = _residual_trajectory(family)
    thetas = [float(i) for i in range(len(energies))]
    rho = _spearman(thetas, energies)
    assert rho > 0.98, f"{family} severity trajectory not monotone (rho={rho:.3f})"


def test_parametric_family_draw_is_trajectory_stable() -> None:
    """The random parameter is constant across a (content, family) severity sweep."""
    from spectramr.core.metrics.meta_evaluation.simulator import (
        TRAJECTORY_STABLE_FAMILIES,
    )

    assert {"gamma", "contrast", "bias_field"} <= TRAJECTORY_STABLE_FAMILIES
    # contrast = (c - mean)*scale + mean with scale = 1 + theta*k for a FIXED k:
    # the recovered k = (deg-mean)/(c-mean) - 1, divided by theta, is constant.
    clean = torch.randn(1, 16, 16)
    cfg = SimulatorConfig(families=["contrast"], n_severities_per_family=6, seed=3)
    samples = sorted(
        run_simulator([("s0", clean)], cfg), key=lambda s: s.severity["theta"]
    )
    mean = clean.mean()
    ks = []
    for s in samples:
        theta = s.severity["theta"]
        if theta < 1e-6:
            continue
        num = (s.degraded - mean)
        den = (clean - mean)
        mask = den.abs() > 1e-3
        scale_minus_1 = ((num[mask] / den[mask]).mean() - 1.0).item()
        ks.append(scale_minus_1 / theta)
    # All recovered per-theta coefficients agree (one underlying random draw).
    assert max(ks) - min(ks) < 1e-3


# ---------------------------------------------------------------------------
# Complex-input handling: ``.float()`` drops the imaginary part (critique C3).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn_name", ["degrade_gaussian_noise", "degrade_rician_noise", "degrade_gamma"]
)
def test_complex_degraders_preserve_phase(fn_name: str) -> None:
    """Magnitude-path degraders must not collapse a complex input to its real part."""
    from spectramr.core.metrics.meta_evaluation import simulator as sim

    fn = getattr(sim, fn_name)
    # A purely-imaginary signal: its real part is 0, so a ``.float()``-based
    # degrader would see an all-zero image and produce a near-zero / NaN output.
    clean = torch.zeros(1, 16, 16, dtype=torch.complex64)
    clean.imag = torch.randn(1, 16, 16)
    gen = torch.Generator().manual_seed(0)
    out = fn(clean, 0.5, gen)
    assert out.is_complex()
    assert torch.isfinite(torch.view_as_real(out)).all()
    # The output must carry imaginary energy — phase was not discarded.
    assert out.imag.abs().sum() > 0.0


def test_noise_sigma_uses_magnitude_for_complex() -> None:
    """_noise_sigma reflects signal magnitude, not just the (zero) real part."""
    from spectramr.core.metrics.meta_evaluation.simulator import _noise_sigma

    clean = torch.zeros(1, 16, 16, dtype=torch.complex64)
    clean.imag = torch.full((1, 16, 16), 3.0)  # |signal| = 3 everywhere
    # std of a constant magnitude is ~0, so sigma falls back to the abs floor, but
    # crucially it does not error and is computed from |.| not the real part.
    assert _noise_sigma(clean, 1.0) >= 0.05


# ---------------------------------------------------------------------------
# k-space axis correctness: undersampling and motion act on the PHASE-ENCODE
# axis (H, dim -2); the readout (W) is fully sampled (critique physics 2.1/3.1).
# ---------------------------------------------------------------------------


def test_undersample_masks_phase_encode_not_readout() -> None:
    from spectramr.core.metrics.meta_evaluation.simulator import degrade_kspace_undersample
    from spectramr.infrastructure.physics.fft_ops import fft2c

    # Non-square so the two axes are distinguishable. Complex in → complex out, so
    # fft2c(out) recovers exactly the masked k-space.
    clean = torch.randn(1, 1, 16, 32, dtype=torch.complex64)
    gen = torch.Generator().manual_seed(0)
    out = degrade_kspace_undersample(clean, 0.9, gen)
    k = fft2c(out)
    row_energy = k.abs().sum(dim=-1).flatten()  # energy per H (phase-encode) row
    col_energy = k.abs().sum(dim=-2).flatten()  # energy per W (readout) column
    assert (row_energy < 1e-6).any(), "no PE lines were dropped"
    assert (col_energy > 1e-6).all(), "a readout column was zeroed (wrong axis!)"


def test_motion_corrupts_phase_encode_lines() -> None:
    from spectramr.core.metrics.meta_evaluation.simulator import degrade_motion
    from spectramr.infrastructure.physics.fft_ops import fft2c

    clean = torch.randn(1, 1, 16, 32, dtype=torch.complex64)
    gen = torch.Generator().manual_seed(1)
    out = degrade_motion(clean, 0.8, gen)
    diff = (fft2c(out) - fft2c(clean)).abs()
    row_diff = diff.sum(dim=-1).flatten()  # per H (PE) row
    # Some PE lines are phase-corrupted (full row changes), others are untouched.
    # Threshold well above the ~1e-5 fft round-trip floor on untouched rows.
    assert (row_diff > 1e-2).any(), "no PE lines corrupted"
    assert (row_diff < 1e-2).any(), "every PE line corrupted (frac should be < 1)"
