"""Phase 4c — TemporalConfigSchema validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mriforge.config.schemas.data import DataConfigSchema, TemporalConfigSchema


def test_temporal_defaults_are_disabled() -> None:
    cfg = TemporalConfigSchema()
    assert cfg.enabled is False
    assert cfg.frames_per_window == 8
    assert cfg.frame_axis == 3
    assert cfg.temporal_overlap == 2


def test_temporal_overlap_must_be_below_window() -> None:
    """overlap >= window means non-advancing windows."""
    with pytest.raises(ValidationError, match="temporal_overlap"):
        TemporalConfigSchema(
            enabled=True,
            frames_per_window=8,
            temporal_overlap=8,
        )


def test_temporal_overlap_zero_is_valid() -> None:
    cfg = TemporalConfigSchema(
        enabled=True, frames_per_window=8, temporal_overlap=0, target_source="self"
    )
    assert cfg.temporal_overlap == 0


def test_temporal_frame_axis_enum_is_closed() -> None:
    with pytest.raises(ValidationError):
        TemporalConfigSchema(frame_axis=5)  # type: ignore[arg-type]


def test_data_config_exposes_temporal_block() -> None:
    cfg = DataConfigSchema()
    assert hasattr(cfg, "temporal")
    assert cfg.temporal.enabled is False


def test_data_config_accepts_cine_dataset_type() -> None:
    cfg = DataConfigSchema(dataset_type="cine")
    assert cfg.dataset_type == "cine"


def test_data_config_threads_temporal_block_end_to_end() -> None:
    cfg = DataConfigSchema(
        dataset_type="cine",
        temporal={
            "enabled": True,
            "frames_per_window": 4,
            "frame_axis": 0,
            "temporal_overlap": 1,
            "target_source": "self",
        },
    )
    assert cfg.temporal.enabled is True
    assert cfg.temporal.frames_per_window == 4
    assert cfg.temporal.frame_axis == 0
    assert cfg.temporal.temporal_overlap == 1


def test_temporal_total_frames_optional() -> None:
    cfg = TemporalConfigSchema()
    assert cfg.total_frames is None
    cfg2 = TemporalConfigSchema(enabled=True, total_frames=24, target_source="self")
    assert cfg2.total_frames == 24
