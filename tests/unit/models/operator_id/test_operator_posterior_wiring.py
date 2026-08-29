"""Wiring tests for the T3 operator posterior: config knob + audit gate + strategy."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mriforge.config.schemas.training.operator_id import OperatorIDConfig
from mriforge.infrastructure.validation.config_health_checker import ConfigHealthChecker


# --- config knob -------------------------------------------------------------
def test_operator_posterior_defaults_to_point_estimate():
    cfg = OperatorIDConfig()
    assert cfg.operator_posterior == "none"
    assert cfg.posterior_prior_precision == 0.0


def test_operator_posterior_accepts_laplace():
    cfg = OperatorIDConfig(operator_posterior="laplace", posterior_prior_precision=1.0)
    assert cfg.operator_posterior == "laplace"
    assert cfg.posterior_prior_precision == 1.0


def test_operator_posterior_rejects_unknown_and_negative_precision():
    with pytest.raises(ValidationError):
        OperatorIDConfig(operator_posterior="swag_typo")
    with pytest.raises(ValidationError):
        OperatorIDConfig(posterior_prior_precision=-1.0)


# --- audit gate (identifiability) -------------------------------------------
class _Op:
    def __init__(self, posterior="laplace"):
        self.operator_posterior = posterior


class _Training:
    def __init__(self, posterior="laplace", input_domain="kspace"):
        self.operator_id = _Op(posterior)
        self.input_domain = input_domain


class _Data:
    def __init__(self, dataset_type="kspace"):
        self.dataset_type = dataset_type


class _Config:
    def __init__(self, posterior="laplace", input_domain="kspace", dataset_type="kspace"):
        self.training = _Training(posterior, input_domain)
        self.data = _Data(dataset_type)


def test_audit_not_applicable_for_point_estimate():
    r = ConfigHealthChecker().check_operator_posterior_requires_complex(_Config(posterior="none"))
    assert r.passed and r.severity == "info"


def test_audit_passes_on_complex_kspace():
    r = ConfigHealthChecker().check_operator_posterior_requires_complex(_Config())
    assert r.passed


def test_audit_errors_on_magnitude_only():
    r = ConfigHealthChecker().check_operator_posterior_requires_complex(
        _Config(input_domain="image", dataset_type="image")
    )
    assert not r.passed and r.severity == "error"


def test_audit_wired_into_run_all_checks():
    import inspect

    src = inspect.getsource(ConfigHealthChecker.run_all_checks)
    assert "check_operator_posterior_requires_complex" in src


# --- strategy consumption ----------------------------------------------------
def test_strategy_consumes_laplace_posterior():
    """The operator_id strategy must read the knob and compute the posterior
    (not a dead knob, pitfall #15) — asserted from source."""
    import inspect

    from mriforge.infrastructure.training.strategies.operator_id_bch_strategy import (
        OperatorIdBCHTrainingStrategy,
    )

    src = inspect.getsource(OperatorIdBCHTrainingStrategy)
    assert "operator_posterior" in src
    assert "laplace_posterior" in src
    assert "operator_posterior_std" in src
