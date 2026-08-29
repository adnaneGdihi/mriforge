"""C7 trust calibration: scores, artifact, sizing guard, fabrication flag.

The conformal expectations are planted on known grids so the correct order
statistic is a hand-computable number, and coverage is asserted against the
finite-sample guarantee, not a snapshot.
"""

from __future__ import annotations

import json
import math

import pytest
import torch

from mriforge.infrastructure.calibration.chd import dkw_slack
from mriforge.infrastructure.calibration.score_registry import resolve_calibration_score
from mriforge.infrastructure.calibration.scores import kappa_residual, null_energy
from mriforge.infrastructure.calibration.trust_certificate import (
    TrustCalibrationArtifact,
    confident_fabrication_flag,
    fit_trust_calibration,
    write_trust_certificate_json,
)


def _kspace_and_mask() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    full = torch.randn(2, 2, 16, 16, generator=generator)
    mask = torch.zeros(1, 1, 16, 16)
    mask[..., ::2] = 1.0
    return full * mask, mask


class TestKappaResidualScore:
    def test_zero_for_data_consistent_prediction(self):
        y, mask = _kspace_and_mask()
        pred = y + torch.randn_like(y) * (1.0 - mask)  # differs only off-support
        score = kappa_residual(pred, y, mask=mask)
        assert score.shape == (2,)
        assert torch.allclose(score, torch.zeros(2))

    def test_planted_value_is_exact(self):
        """pred = 3y on the support ⇒ κ = ‖2y·M‖/‖y·M‖ = 2 per image."""
        y, mask = _kspace_and_mask()
        score = kappa_residual(3.0 * y, y, mask=mask)
        assert torch.allclose(score, torch.full((2,), 2.0), rtol=1e-5)

    def test_scale_invariant(self):
        y, mask = _kspace_and_mask()
        pred = 1.7 * y
        assert torch.allclose(
            kappa_residual(pred, y, mask=mask),
            kappa_residual(10 * pred, 10 * y, mask=mask),
            rtol=1e-6,
        )

    def test_shape_mismatch_raises(self):
        y, mask = _kspace_and_mask()
        with pytest.raises(ValueError, match="Shape mismatch"):
            kappa_residual(y[:, :1], y, mask=mask)

    def test_registered_and_dispatches(self):
        y, mask = _kspace_and_mask()
        fn = resolve_calibration_score("kappa_residual")
        assert torch.allclose(fn(3.0 * y, y, mask=mask), torch.full((2,), 2.0), rtol=1e-5)


class TestNullEnergyScore:
    def test_extremes(self):
        _, mask = _kspace_and_mask()
        on_support = torch.randn(2, 2, 16, 16) * mask
        off_support = torch.randn(2, 2, 16, 16) * (1.0 - mask)
        assert torch.allclose(null_energy(on_support, mask=mask), torch.zeros(2))
        assert torch.allclose(
            null_energy(off_support, mask=mask), torch.ones(2), atol=1e-6
        )

    def test_mask_is_mandatory(self):
        pred = torch.randn(1, 2, 8, 8)
        with pytest.raises(TypeError):
            null_energy(pred)  # a null band is undefined without a mask

    def test_registered_and_ignores_target(self):
        _, mask = _kspace_and_mask()
        pred = torch.randn(2, 2, 16, 16) * mask
        fn = resolve_calibration_score("null_energy")
        score = fn(pred, torch.zeros_like(pred), mask=mask)
        assert torch.allclose(score, torch.zeros(2))


class TestFitTrustCalibration:
    def test_quantile_is_the_conformal_order_statistic(self):
        """n=99 scores 0.01..0.99, α=0.1 ⇒ rank ⌈100·0.9⌉=90 ⇒ q̂=0.90."""
        scores = [i / 100 for i in range(1, 100)]
        artifact = fit_trust_calibration(scores, scores, alpha=0.1)
        assert artifact.q_hat_alpha == pytest.approx(0.90)
        assert artifact.n == 99
        assert artifact.dkw_eps == pytest.approx(dkw_slack(99, 0.1))

    def test_coverage_on_exchangeable_scores(self):
        """Fresh draws land inside the band ≥ 1−α−dkw_eps.

        The conformal guarantee is marginal over the JOINT calibration+test
        draw; for a fixed calibration realisation the honest finite-sample
        floor is 1−α minus the DKW band — which is exactly what the artifact
        records, so the test asserts the bound the artifact advertises.
        """
        generator = torch.Generator().manual_seed(7)
        calibration = torch.rand(200, generator=generator)
        fresh = torch.rand(4000, generator=generator)
        artifact = fit_trust_calibration(
            calibration.tolist(), calibration.tolist(), alpha=0.1
        )
        coverage = float((fresh <= artifact.q_hat_alpha).float().mean())
        assert coverage >= 1.0 - artifact.alpha - artifact.dkw_eps

    def test_undersized_calibration_raises(self):
        """n=5 < ⌈1/0.1⌉−1 = 9: the honest band is infinite ⇒ hard error."""
        with pytest.raises(ValueError, match="too small"):
            fit_trust_calibration([0.1] * 5, [0.1] * 5, alpha=0.1)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            fit_trust_calibration([], [], alpha=0.1)

    def test_eta_quantile_law_is_recorded(self):
        scores = [i / 100 for i in range(1, 100)]
        eta = [i / 1000 for i in range(1, 100)]
        artifact = fit_trust_calibration(scores, eta, alpha=0.1)
        # nearest-rank: q50 -> 50th of 99, q99 -> ceil(0.99*99)=99th
        assert artifact.eta_null_clean_quantiles["q50"] == pytest.approx(0.050)
        assert artifact.eta_null_clean_quantiles["q99"] == pytest.approx(0.099)

    def test_provenance_is_recorded(self):
        scores = [i / 100 for i in range(1, 100)]
        artifact = fit_trust_calibration(
            scores, scores, alpha=0.1, score_name="null_energy", sigma=0.05, seed=42
        )
        assert artifact.score_name == "null_energy"
        assert artifact.sigma == 0.05
        assert artifact.seed == 42


class TestConfidentFabricationFlag:
    _ARTIFACT = TrustCalibrationArtifact(
        score_name="kappa_residual",
        alpha=0.1,
        n=100,
        q_hat_alpha=0.2,
        eta_null_clean_quantiles={"q50": 0.1, "q99": 0.3},
        dkw_eps=0.05,
    )

    def test_fires_on_low_kappa_high_eta(self):
        assert confident_fabrication_flag(0.1, 0.5, self._ARTIFACT) is True

    def test_silent_on_clean(self):
        assert confident_fabrication_flag(0.1, 0.1, self._ARTIFACT) is False

    def test_silent_when_kappa_is_out_of_band(self):
        """High κ + high η is VISIBLE inconsistency, not confident fabrication."""
        assert confident_fabrication_flag(0.5, 0.5, self._ARTIFACT) is False

    def test_unknown_quantile_key_raises(self):
        with pytest.raises(KeyError, match="q77"):
            confident_fabrication_flag(0.1, 0.5, self._ARTIFACT, eta_quantile="q77")


class TestSerialisation:
    def test_json_round_trip(self, tmp_path):
        scores = [i / 100 for i in range(1, 100)]
        artifact = fit_trust_calibration(scores, scores, alpha=0.1, sigma=0.0, seed=42)
        path = write_trust_certificate_json(artifact, tmp_path / "certs" / "trust.json")
        assert path.exists()
        restored = TrustCalibrationArtifact.from_dict(json.loads(path.read_text()))
        assert restored == artifact

    def test_dkw_eps_matches_the_ssot(self):
        scores = [i / 100 for i in range(1, 100)]
        artifact = fit_trust_calibration(scores, scores, alpha=0.05)
        assert artifact.dkw_eps == pytest.approx(
            math.sqrt(math.log(2 / 0.05) / (2 * 99))
        )
