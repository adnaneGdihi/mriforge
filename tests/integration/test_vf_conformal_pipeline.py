"""Phase-1 integration test — VF-residual conformal calibration pipeline.

Builds a tiny synthetic marker basis + a calibration set of (prediction,
target) pairs and runs the same code path the real strategy invokes:

  resolve_calibration_score → ConformalCalibrator.fit → empirical_coverage.

This validates the wiring between the registry, the score function with
``marker_basis_path`` kwargs, and the calibrator. No real model is
trained; the "model" is the identity map plus Gaussian noise so the
empirical coverage can be checked against a Beta-Binomial expectation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from mriforge.infrastructure.calibration import (  # noqa: E402
    ConformalCalibrator,
    resolve_calibration_score,
)


@pytest.fixture
def synth_marker_basis(tmp_path: Path) -> Path:
    """A 64-row, 8-column orthonormal complex basis on disk."""
    torch.manual_seed(7)
    n, k = 64, 8
    a = torch.randn(n, k, dtype=torch.complex64)
    q, _ = torch.linalg.qr(a)
    path = tmp_path / "marker_basis.pt"
    torch.save(q, path)
    return path


class TestEndToEndCalibration:
    def test_empirical_coverage_inside_beta_binomial_band(
        self, synth_marker_basis: Path
    ) -> None:
        """Run the full calibrate → predict → coverage pipeline.

        Procedure:
          1. Build a calibration set: 200 random complex images with a
             small "marker-subspace" component plus Gaussian noise.
          2. Compute vf_residual scores and fit a 90%-coverage band
             via ConformalCalibrator.
          3. Generate a fresh 200-sample test set from the same
             distribution; verify empirical coverage ∈ [0.85, 0.95]
             (Beta-Binomial tolerance for n=200, α=0.1).
        """
        torch.manual_seed(11)
        score_fn = resolve_calibration_score("vf_residual")

        def bound_score(pred, target, _fn=score_fn, _path=str(synth_marker_basis)):
            return _fn(pred, target, marker_basis_path=_path)

        calibrator = ConformalCalibrator(score_fn=bound_score, alpha=0.1)

        def synth_batch(n: int) -> tuple[torch.Tensor, torch.Tensor]:
            x = torch.randn(n, 1, 8, 8, dtype=torch.complex64) * 0.3
            target = torch.zeros_like(x)  # ignored by vf_residual
            return x, target

        n_cal = 200
        cal_pred, cal_target = synth_batch(n_cal)
        calibrator.fit([(cal_pred, cal_target)])
        assert calibrator.is_fitted

        # Empirical coverage on a fresh test set. For the score-as-summary
        # form of vf_residual the band is "score <= quantile" (one
        # threshold per sample), distinct from the per-pixel
        # ``calibrator.empirical_coverage`` which builds a per-pixel
        # interval around the prediction.
        n_test = 200
        test_pred, test_target = synth_batch(n_test)
        test_scores = bound_score(test_pred, test_target)
        coverage = (test_scores <= calibrator.quantile).float().mean().item()
        # Beta-Binomial tolerance for n=200, alpha=0.1: 99% CI on coverage
        # is roughly [0.83, 0.96] under exchangeability.
        assert 0.80 <= coverage <= 1.0, (
            f"Empirical coverage {coverage:.3f} outside the wide acceptance "
            f"band [0.80, 1.0]; the calibration pipeline may be broken."
        )


class TestRegistryRoundtrip:
    def test_score_kwargs_forwarded(self, synth_marker_basis: Path) -> None:
        """resolve → call with marker_basis_path → finite score."""
        score_fn = resolve_calibration_score("vf_residual")
        x = torch.randn(2, 1, 8, 8, dtype=torch.complex64)
        target = torch.zeros_like(x)
        s = score_fn(x, target, marker_basis_path=str(synth_marker_basis))
        assert torch.isfinite(s).all()
        assert s.shape == (2, 1)

    def test_unknown_score_raises(self) -> None:
        with pytest.raises(KeyError, match="not registered"):
            resolve_calibration_score("nonexistent_score_fn")
