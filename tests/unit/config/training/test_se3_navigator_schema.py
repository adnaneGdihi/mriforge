"""Unit tests for the SE(3)-navigator training schema (B-3 of the VF campaign).

Direct-import per the source<->test pairing rule.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mriforge.config.schemas.training.se3_navigator import (
    SE3NavigatorConfig,
    TrainingConfigSE3Navigator,
)


def test_defaults_valid() -> None:
    c = SE3NavigatorConfig()
    assert c.group == "SE3"
    assert c.n_mamba_layers == 4
    assert c.d_state == 16
    assert c.bio_harmonic_freqs_Hz == (0.3, 0.6, 1.2, 2.4)
    assert c.dc_navigator_sample_rate_Hz == 100.0


def test_frozen() -> None:
    c = SE3NavigatorConfig()
    with pytest.raises(ValidationError):
        c.n_mamba_layers = 8  # type: ignore[misc]


def test_extra_forbidden() -> None:
    with pytest.raises(ValidationError):
        SE3NavigatorConfig(unexpected_field=1)


def test_nyquist_violation_raises() -> None:
    with pytest.raises(ValidationError, match="Nyquist"):
        SE3NavigatorConfig(
            dc_navigator_sample_rate_Hz=4.0, bio_harmonic_freqs_Hz=(2.4,)
        )


def test_physiological_band_low_raises() -> None:
    with pytest.raises(ValidationError, match="physiological"):
        SE3NavigatorConfig(bio_harmonic_freqs_Hz=(0.05,))


def test_physiological_band_high_raises() -> None:
    with pytest.raises(ValidationError, match="physiological"):
        SE3NavigatorConfig(bio_harmonic_freqs_Hz=(7.0,))


def test_nonpositive_frequency_raises() -> None:
    with pytest.raises(ValidationError):
        SE3NavigatorConfig(bio_harmonic_freqs_Hz=(0.0, 1.2))


def test_empty_frequencies_raises() -> None:
    with pytest.raises(ValidationError):
        SE3NavigatorConfig(bio_harmonic_freqs_Hz=())


def test_layer_bounds() -> None:
    with pytest.raises(ValidationError):
        SE3NavigatorConfig(n_mamba_layers=0)
    with pytest.raises(ValidationError):
        SE3NavigatorConfig(n_mamba_layers=99)


def test_group_literal_enforced() -> None:
    with pytest.raises(ValidationError):
        SE3NavigatorConfig(group="SU2")


def test_top_level_config_mounts_block() -> None:
    t = TrainingConfigSE3Navigator()
    assert isinstance(t.se3_equivariant_navigator, SE3NavigatorConfig)
    assert t.se3_equivariant_navigator.group == "SE3"


def test_top_level_accepts_custom_block() -> None:
    t = TrainingConfigSE3Navigator(
        se3_equivariant_navigator={
            "n_mamba_layers": 6,
            "group": "SO3",
            "dc_navigator_sample_rate_Hz": 50.0,
        }
    )
    assert t.se3_equivariant_navigator.n_mamba_layers == 6
    assert t.se3_equivariant_navigator.group == "SO3"
