"""Phase 4b schemas — quantitative + acquisition_metadata validation.

These tests pin the contract of the two new config blocks
(``data.quantitative.*`` and ``data.acquisition_metadata.*``) without
exercising any file IO. The transform and dataset have their own tests.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from mriforge.config.schemas.data import (
    AcquisitionMetadataConfigSchema,
    DataConfigSchema,
    QuantitativeConfigSchema,
)


# ── QuantitativeConfigSchema ─────────────────────────────────────────────────


def test_quantitative_defaults_are_disabled() -> None:
    """Pre-Phase-4b YAMLs see a fully no-op default block."""
    cfg = QuantitativeConfigSchema()
    assert cfg.enabled is False
    assert cfg.target_maps == []
    assert cfg.units == "physical"


def test_quantitative_enabled_requires_target_maps() -> None:
    """Enabling without declaring any maps must fail loudly (CLAUDE.md #9)."""
    with pytest.raises(ValidationError, match="target_maps"):
        QuantitativeConfigSchema(enabled=True, target_maps=[])


def test_quantitative_accepts_t1_t2_pd() -> None:
    cfg = QuantitativeConfigSchema(
        enabled=True, target_maps=["t1", "t2", "pd"], input_source="contrasts"
    )
    assert cfg.target_maps == ["t1", "t2", "pd"]


def test_quantitative_rejects_unknown_map() -> None:
    """The map enum is closed — typos surface at load time."""
    with pytest.raises(ValidationError):
        QuantitativeConfigSchema(enabled=True, target_maps=["t1", "magnetization"])


def test_quantitative_units_enum_is_closed() -> None:
    with pytest.raises(ValidationError):
        QuantitativeConfigSchema(units="raw")  # type: ignore[arg-type]


# ── AcquisitionMetadataConfigSchema ──────────────────────────────────────────


def test_acquisition_metadata_defaults_are_disabled() -> None:
    cfg = AcquisitionMetadataConfigSchema()
    assert cfg.enabled is False
    assert cfg.source == "bids_json"
    assert "TE" in cfg.fields
    assert "TR" in cfg.fields


def test_acquisition_metadata_strict_defaults_to_false() -> None:
    """``strict=False`` is the user-friendly default; flipping to True is opt-in."""
    cfg = AcquisitionMetadataConfigSchema()
    assert cfg.strict is False


def test_acquisition_metadata_source_enum_is_closed() -> None:
    with pytest.raises(ValidationError):
        AcquisitionMetadataConfigSchema(source="parquet")  # type: ignore[arg-type]


def test_acquisition_metadata_fields_enum_is_closed() -> None:
    with pytest.raises(ValidationError):
        AcquisitionMetadataConfigSchema(fields=["TE", "SNR"])  # type: ignore[list-item]


def test_acquisition_metadata_allows_inline_defaults() -> None:
    """``yaml_inline`` source uses the ``defaults`` map as the truth."""
    cfg = AcquisitionMetadataConfigSchema(
        enabled=True,
        source="yaml_inline",
        fields=["TE", "TR", "FA"],
        defaults={"TE": 4.5, "TR": 8.0, "FA": 15.0},
    )
    assert cfg.source == "yaml_inline"
    assert cfg.defaults["TE"] == 4.5


# ── DataConfigSchema wiring ──────────────────────────────────────────────────


def test_data_config_exposes_quantitative_block() -> None:
    cfg = DataConfigSchema()
    assert hasattr(cfg, "quantitative")
    assert cfg.quantitative.enabled is False


def test_data_config_exposes_acquisition_metadata_block() -> None:
    cfg = DataConfigSchema()
    assert hasattr(cfg, "acquisition_metadata")
    assert cfg.acquisition_metadata.enabled is False


def test_data_config_accepts_quantitative_dataset_type() -> None:
    """``dataset_type='quantitative'`` must validate (Phase 4b)."""
    cfg = DataConfigSchema(dataset_type="quantitative")
    assert cfg.dataset_type == "quantitative"


def test_data_config_threads_quantitative_block_end_to_end() -> None:
    cfg = DataConfigSchema(
        dataset_type="quantitative",
        quantitative={
                "enabled": True,
                "target_maps": ["t1", "t2"],
                "input_source": "contrasts",
            },
    )
    assert cfg.quantitative.enabled is True
    assert cfg.quantitative.target_maps == ["t1", "t2"]
