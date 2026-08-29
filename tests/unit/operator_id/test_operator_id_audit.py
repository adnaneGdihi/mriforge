"""Tests for the operator-ID Tier-1 audit checks (Proposal 1).

Covers the four checks (``operator_basis_registered``,
``covariance_rank_bound``, ``bch_order_supported``, ``krylov_dim_valid``) in
isolation, plus an end-to-end clean audit of the smoke + in-progress YAMLs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from mriforge.infrastructure.validation.config_health_checker import ConfigHealthChecker


def _cfg(operator_id: dict | None, patch=(64, 64, 1), in_channels=1):
    return SimpleNamespace(
        training=SimpleNamespace(operator_id=operator_id),
        # `patch_size` folded to `data.sampling.patch_size`, which is what the
        # checker reads. The flat spelling made that read raise -- and the
        # checker swallows it in a bare `except Exception` and substitutes
        # `n_pixels = 64*64`, so both krylov tests were graded against a
        # 4096-pixel image they never asked for instead of their own (8, 8).
        data=SimpleNamespace(sampling=SimpleNamespace(patch_size=list(patch))),
        model=SimpleNamespace(in_channels=in_channels),
    )


def _by_name(results):
    return {r.check_name: r for r in results}


def test_no_checks_when_operator_id_absent():
    checker = ConfigHealthChecker()
    assert checker.check_operator_id_config(_cfg(None)) == []


def test_all_checks_pass_for_valid_config():
    checker = ConfigHealthChecker()
    op = {
        "mode_dictionary": ["motion_rigid", "b0_phase", "rician_noise"],
        "covariance_rank": 8,
        "bch_order": 2,
        "krylov_dim": 30,
    }
    res = _by_name(checker.check_operator_id_config(_cfg(op)))
    assert res["operator_basis_registered"].passed
    assert res["covariance_rank_bound"].passed
    assert res["bch_order_supported"].passed
    assert res["krylov_dim_valid"].passed


def test_unregistered_mode_errors():
    checker = ConfigHealthChecker()
    op = {
        "mode_dictionary": ["not_a_real_mode"],
        "covariance_rank": 4,
        "bch_order": 2,
        "krylov_dim": 8,
    }
    res = _by_name(checker.check_operator_id_config(_cfg(op)))
    r = res["operator_basis_registered"]
    assert not r.passed and r.severity == "error"


def test_bad_bch_order_errors():
    checker = ConfigHealthChecker()
    op = {
        "mode_dictionary": ["motion_rigid"],
        "covariance_rank": 4,
        "bch_order": 5,
        "krylov_dim": 8,
    }
    res = _by_name(checker.check_operator_id_config(_cfg(op)))
    assert not res["bch_order_supported"].passed
    assert res["bch_order_supported"].severity == "error"


def test_covariance_rank_out_of_bounds_errors():
    checker = ConfigHealthChecker()
    # rank >= N_pixels (here 64*64=4096)
    op = {
        "mode_dictionary": ["motion_rigid"],
        "covariance_rank": 99999,
        "bch_order": 2,
        "krylov_dim": 8,
    }
    res = _by_name(checker.check_operator_id_config(_cfg(op)))
    assert not res["covariance_rank_bound"].passed
    assert res["covariance_rank_bound"].severity == "error"


def test_krylov_dim_too_large_warns():
    checker = ConfigHealthChecker()
    # Within [1, N] but large relative to N (N=8*8=64) ⇒ warning.
    op = {
        "mode_dictionary": ["motion_rigid"],
        "covariance_rank": 4,
        "bch_order": 2,
        "krylov_dim": 60,
    }
    res = _by_name(checker.check_operator_id_config(_cfg(op, patch=(8, 8, 1))))
    r = res["krylov_dim_valid"]
    assert not r.passed and r.severity == "warning"


def test_krylov_dim_above_n_errors():
    checker = ConfigHealthChecker()
    op = {
        "mode_dictionary": ["motion_rigid"],
        "covariance_rank": 4,
        "bch_order": 2,
        "krylov_dim": 999,
    }
    res = _by_name(checker.check_operator_id_config(_cfg(op, patch=(8, 8, 1))))
    assert not res["krylov_dim_valid"].passed
    assert res["krylov_dim_valid"].severity == "error"


@pytest.mark.parametrize(
    "yaml_path",
    [
        "tests/smoke/operator_id_smoke.yaml",
        # The in-progress arm lives under the maintainer-private (git-ignored)
        # experiments/ tree; audited when present, skipped otherwise.
        "experiments/inprogress/operator_id/bch_m4raw.yaml",
    ],
)
def test_yaml_audits_clean(yaml_path):
    import os

    if not os.path.exists(yaml_path):
        pytest.skip(f"{yaml_path} not present (git-ignored experiments tree).")

    from mriforge.config.settings import TrainingSettings
    from mriforge.infrastructure.validation.config_health_checker import (
        validate_config_health,
    )

    settings = TrainingSettings.from_yaml(yaml_path)
    report = validate_config_health(settings)
    assert report.passed, [str(e) for e in report.errors]
    assert len(report.warnings) == 0, [str(w) for w in report.warnings]
