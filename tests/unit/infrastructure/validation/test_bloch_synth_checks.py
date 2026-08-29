"""Tier-1 audit guard tests for the bloch_synth arm (MICCAI MRIxFields2026, 2.1)."""

from __future__ import annotations

from types import SimpleNamespace

from mriforge.infrastructure.validation.config_health_checker import ConfigHealthChecker


def _checker() -> ConfigHealthChecker:
    return object.__new__(ConfigHealthChecker)


def _cfg(
    *,
    contrasts=("T1w", "T2w", "FLAIR"),
    in_channels=3,
    bounds=(0.3, 0.4),
    mode="bloch_synth",
):
    return SimpleNamespace(
        training=SimpleNamespace(
            training_mode=mode,
            strategy_class=mode,
            bloch_synth=SimpleNamespace(
                source_contrasts=list(contrasts),
                dispersion_beta_bounds=bounds,
            ),
        ),
        model=SimpleNamespace(in_channels=in_channels),
    )


def _by_name(results):
    return {r.check_name: r for r in results}


def test_valid_arm_passes() -> None:
    by = _by_name(_checker().check_bloch_synth_arm(_cfg()))
    assert by["bloch_synth_source_contrast_count"].passed
    assert by["bloch_synth_dispersion_bounds"].passed


def test_not_applicable_for_other_mode() -> None:
    r = _checker().check_bloch_synth_arm(_cfg(mode="reconstruction"))
    assert len(r) == 1 and r[0].passed


def test_under_determined_fires() -> None:
    r = _by_name(
        _checker().check_bloch_synth_arm(_cfg(contrasts=("T1w",), in_channels=1))
    )
    c = r["bloch_synth_source_contrast_count"]
    assert not c.passed and c.severity == "error"


def test_in_channels_mismatch_fires() -> None:
    r = _by_name(
        _checker().check_bloch_synth_arm(_cfg(in_channels=1))
    )  # 3 contrasts, 1 ch
    c = r["bloch_synth_source_contrast_count"]
    assert not c.passed and c.severity == "error"


def test_dispersion_bounds_warns_outside_envelope() -> None:
    r = _by_name(_checker().check_bloch_synth_arm(_cfg(bounds=(0.05, 0.9))))
    c = r["bloch_synth_dispersion_bounds"]
    assert not c.passed and c.severity == "warning"
