"""Spec E2 — BartConfigSchema (BART / non-Cartesian k-space ingestion) validation.

The BART dim vector is not self-describing, so each non-singleton dim's role MUST
be declared explicitly via ``bart_dim_map`` — the loader never guesses
(CLAUDE.md pitfalls #9 / #15). These tests pin the raise-on-unknown contract.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.data import BartConfigSchema, DataConfigSchema


def test_bart_defaults_are_disabled() -> None:
    cfg = BartConfigSchema()
    assert cfg.enabled is False
    assert cfg.bart_dim_map == {}
    assert cfg.sampling == "cartesian"
    assert cfg.trajectory_source == "none"
    assert cfg.density_compensation == "none"


def test_enabled_requires_non_empty_dim_map() -> None:
    with pytest.raises(ValidationError, match="bart_dim_map"):
        BartConfigSchema(enabled=True, bart_dim_map={})


def test_unknown_role_raises() -> None:
    with pytest.raises(ValidationError, match="unknown role"):
        BartConfigSchema(enabled=True, bart_dim_map={"banana": 0})


def test_dim_index_out_of_range_raises() -> None:
    with pytest.raises(ValidationError, match="range"):
        BartConfigSchema(enabled=True, bart_dim_map={"readout": 99})


def test_duplicate_dim_index_raises() -> None:
    with pytest.raises(ValidationError, match="same BART dim"):
        BartConfigSchema(enabled=True, bart_dim_map={"readout": 1, "coil": 1})


def test_non_cartesian_requires_trajectory_source() -> None:
    with pytest.raises(ValidationError, match="trajectory_source"):
        BartConfigSchema(
            enabled=True,
            bart_dim_map={"readout": 1, "coil": 3, "spoke": 2},
            sampling="radial",
            trajectory_source="none",
        )


def test_valid_radial_multiecho_config() -> None:
    # mirrors multiecho_radial_b0_r2star dims [1,408,33,20,1,1,7,...]
    cfg = BartConfigSchema(
        enabled=True,
        bart_dim_map={"readout": 1, "spoke": 2, "coil": 3, "echo": 6},
        sampling="radial",
        trajectory_source="golden_angle",
        density_compensation="radial",
    )
    assert cfg.bart_dim_map["echo"] == 6
    assert cfg.sampling == "radial"


def test_valid_cartesian_config() -> None:
    # mirrors phase_pole_correction dims [1,256,208,20,...]
    cfg = BartConfigSchema(
        enabled=True,
        bart_dim_map={"readout": 1, "phase": 2, "coil": 3},
        sampling="cartesian",
    )
    assert cfg.trajectory_source == "none"  # Cartesian needs no measured traj


def test_mounts_on_data_config_and_defaults_disabled() -> None:
    cfg = DataConfigSchema()
    assert cfg.bart.enabled is False


def test_new_dataset_types_pass_the_allow_list_validator() -> None:
    # regression: the dataset_type field-validator allow-list must include the
    # new types, else a config using them is rejected at load BEFORE dispatch.
    for dt in ("bart_kspace", "bids_paired", "png_paired"):
        assert DataConfigSchema(dataset_type=dt).dataset_type == dt


def test_b0_from_echo_requires_echo_role_and_delta_te() -> None:
    # b0_from_echo with no echo role → raise
    with pytest.raises(ValidationError, match="echo"):
        BartConfigSchema(
            enabled=True,
            bart_dim_map={"readout": 1, "coil": 3},
            b0_from_echo=True,
            delta_te=0.0025,
        )
    # b0_from_echo with no delta_te → raise (no absolute Hz without ΔTE)
    with pytest.raises(ValidationError, match="delta_te"):
        BartConfigSchema(
            enabled=True,
            bart_dim_map={"readout": 1, "spoke": 2, "coil": 3, "echo": 6},
            sampling="radial",
            trajectory_source="golden_angle",
            b0_from_echo=True,
        )


def test_b1_from_dam_requires_flip_role_and_nominal_flip() -> None:
    with pytest.raises(ValidationError, match="flip"):
        BartConfigSchema(
            enabled=True,
            bart_dim_map={"readout": 1, "phase": 2, "coil": 3},
            b1_from_dam=True,
            b1_nominal_flip_deg=60.0,
        )
    with pytest.raises(ValidationError, match="b1_nominal_flip"):
        BartConfigSchema(
            enabled=True,
            bart_dim_map={"readout": 1, "phase": 2, "coil": 3, "flip": 11},
            b1_from_dam=True,
        )


def test_b0_from_echo_valid() -> None:
    cfg = BartConfigSchema(
        enabled=True,
        bart_dim_map={"readout": 1, "spoke": 2, "coil": 3, "echo": 6},
        sampling="radial",
        trajectory_source="golden_angle",
        density_compensation="radial",
        b0_from_echo=True,
        delta_te=0.00246,
    )
    assert cfg.b0_from_echo is True
    assert cfg.delta_te == 0.00246
