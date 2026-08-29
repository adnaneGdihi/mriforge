"""Unit tests for the Robust SSDU audit checks (Phase B step 4).

``check_ssdu_selection_density_range`` warns when the held-out fraction is
outside the useful [0.2, 0.5] band; ``check_robust_ssdu_key_matches_correction``
refuses the contradiction of the ``robust_ssdu`` key with the Noisier2Noise
correction off (pitfall #16).
"""

from __future__ import annotations

from mriforge.infrastructure.validation.config_health_checker import ConfigHealthChecker


class _SSDU:
    def __init__(self, theta_fraction=0.4, noisier2noise_correction=False):
        self.theta_fraction = theta_fraction
        self.noisier2noise_correction = noisier2noise_correction


class _Training:
    def __init__(self, training_mode="ssdu", ssdu_block=True, **ssdu_kw):
        self.training_mode = training_mode
        self.strategy_class = (
            "mriforge.infrastructure.training.strategies."
            "ssdu_strategy.SSDUReconstructionStrategy"
        )
        self.ssdu = _SSDU(**ssdu_kw) if ssdu_block else None


class _Config:
    def __init__(self, training_mode="ssdu", ssdu_block=True, **ssdu_kw):
        self.training = _Training(training_mode=training_mode, ssdu_block=ssdu_block, **ssdu_kw)


def _c():
    return ConfigHealthChecker()


# --- selection density -------------------------------------------------------
def test_density_not_applicable_for_non_ssdu_arm():
    cfg = _Config(training_mode="reconstruction", ssdu_block=False)
    cfg.training.strategy_class = "mriforge.x.Y"
    r = _c().check_ssdu_selection_density_range(cfg)
    assert r.passed and r.severity == "info"


def test_density_passes_in_range():
    r = _c().check_ssdu_selection_density_range(_Config(theta_fraction=0.4))
    assert r.passed


def test_density_warns_out_of_range():
    r = _c().check_ssdu_selection_density_range(_Config(theta_fraction=0.7))
    assert not r.passed and r.severity == "warning"


# --- robust_ssdu key match ---------------------------------------------------
def test_robust_key_not_applicable_for_plain_ssdu():
    r = _c().check_robust_ssdu_key_matches_correction(_Config(training_mode="ssdu"))
    assert r.passed and r.severity == "info"


def test_robust_key_passes_when_correction_on():
    r = _c().check_robust_ssdu_key_matches_correction(
        _Config(training_mode="robust_ssdu", noisier2noise_correction=True)
    )
    assert r.passed


def test_robust_key_errors_when_correction_off():
    r = _c().check_robust_ssdu_key_matches_correction(
        _Config(training_mode="robust_ssdu", noisier2noise_correction=False)
    )
    assert not r.passed and r.severity == "error"


def test_ssdu_checks_wired_into_run_all_checks():
    import inspect

    src = inspect.getsource(ConfigHealthChecker.run_all_checks)
    assert "check_ssdu_selection_density_range" in src
    assert "check_robust_ssdu_key_matches_correction" in src
