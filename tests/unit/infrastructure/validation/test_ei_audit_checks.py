"""Unit tests for the Equivariant Imaging audit checks (Phase A step 3).

``check_ei_sensing_margin`` is the static identifiability gate (EI needs
undersampling so the operator breaks the group symmetry — Tachella sensing
theorems). ``check_ei_robust_key_matches_correction`` refuses the contradiction
of selecting the ``robust_ei`` key with the nc-χ correction turned off
(pitfall #16: a registry key that silently does nothing).
"""

from __future__ import annotations

from mriforge.infrastructure.validation.config_health_checker import ConfigHealthChecker


class _CS:
    def __init__(self, acceleration_factor=4.0):
        self.acceleration_factor = acceleration_factor


class _Physics:
    def __init__(self, acceleration_factor=4.0):
        self.compressed_sensing = _CS(acceleration_factor=acceleration_factor)


class _Accel:
    def __init__(self, base_acceleration=4.0):
        self.base_acceleration = base_acceleration


class _EI:
    def __init__(self, robust_correction=False):
        self.robust_correction = robust_correction


class _Training:
    def __init__(self, training_mode="equivariant_imaging", ei_block=True, robust_correction=False):
        self.training_mode = training_mode
        self.strategy_class = (
            "mriforge.infrastructure.training.strategies."
            "equivariant_imaging_strategy.EquivariantImagingStrategy"
        )
        self.equivariant_imaging = (
            _EI(robust_correction=robust_correction) if ei_block else None
        )


class _Config:
    def __init__(
        self,
        training_mode="equivariant_imaging",
        ei_block=True,
        robust_correction=False,
        acceleration_factor=4.0,
    ):
        self.training = _Training(
            training_mode=training_mode,
            ei_block=ei_block,
            robust_correction=robust_correction,
        )
        self.physics = _Physics(acceleration_factor=acceleration_factor)
        self.acceleration = _Accel(base_acceleration=acceleration_factor)


def _checker():
    return ConfigHealthChecker()


# --- check_ei_sensing_margin -------------------------------------------------
def test_sensing_margin_not_applicable_for_non_ei_arm():
    cfg = _Config(training_mode="reconstruction", ei_block=False)
    r = _checker().check_ei_sensing_margin(cfg)
    assert r.passed and r.severity == "info"


def test_sensing_margin_passes_when_undersampled():
    r = _checker().check_ei_sensing_margin(_Config(acceleration_factor=4.0))
    assert r.passed


def test_sensing_margin_errors_when_fully_sampled():
    r = _checker().check_ei_sensing_margin(_Config(acceleration_factor=1.0))
    assert not r.passed and r.severity == "error"


# --- check_ei_robust_key_matches_correction ----------------------------------
def test_robust_key_not_applicable_when_mode_is_plain_ei():
    r = _checker().check_ei_robust_key_matches_correction(
        _Config(training_mode="equivariant_imaging")
    )
    assert r.passed and r.severity == "info"


def test_robust_key_passes_when_correction_on():
    r = _checker().check_ei_robust_key_matches_correction(
        _Config(training_mode="robust_ei", robust_correction=True)
    )
    assert r.passed


def test_robust_key_errors_when_correction_off():
    r = _checker().check_ei_robust_key_matches_correction(
        _Config(training_mode="robust_ei", robust_correction=False)
    )
    assert not r.passed and r.severity == "error"


# --- wiring ------------------------------------------------------------------
def test_ei_checks_wired_into_run_all_checks():
    import inspect

    src = inspect.getsource(ConfigHealthChecker.run_all_checks)
    assert "check_ei_sensing_margin" in src
    assert "check_ei_robust_key_matches_correction" in src
