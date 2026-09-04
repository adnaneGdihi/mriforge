# tests/unit/config/test_coil_processing_schema.py
import pytest
from pydantic import ValidationError

from spectramr.config.schemas.physics import (
    CoilProcessingConfig,
    PhysicsConfigSchema,
)


def test_defaults():
    # A bare block is a no-op (legacy 'none' mode): no compression, no combine,
    # complex k-space output. combine defaults to 'none' so the block is valid
    # and inert by default.
    cp = CoilProcessingConfig()
    assert cp.compression.method == "none"
    assert cp.estimation.method == "power_iter"
    assert cp.combine.method == "none"
    assert cp.output.domain == "kspace"
    assert cp.output.channels == "complex"


def test_calibration_lines_defaults_to_none_full_fov():
    # The schema default MUST be None (full FoV) to agree with the runtime: the
    # loader sync only propagates calibration_lines when set explicitly, so a
    # non-None default would advertise an ACS crop the runtime never applies
    # (CLAUDE.md non-negotiable #8 — no unread knobs).
    cp = CoilProcessingConfig()
    assert cp.compression.calibration_lines is None


def test_calibration_lines_schema_runtime_agree_for_default_svd():
    # A svd config without an explicit calibration_lines must resolve to the same
    # value on both sides (schema None == runtime None == full FoV).
    from spectramr.config.schemas.loader import _sync_coil_processing_to_legacy

    raw = {
        "physics": {
            "coil_processing": {
                "compression": {"method": "svd", "num_virtual_coils": 4}
            }
        }
    }
    out = _sync_coil_processing_to_legacy(raw)
    schema_default = CoilProcessingConfig(
        compression={"method": "svd", "num_virtual_coils": 4}
    ).compression.calibration_lines
    assert schema_default is None
    assert out["data"].get("svd_calibration_lines") is None


def test_full_block():
    cp = CoilProcessingConfig(
        compression={"method": "svd", "num_virtual_coils": 4},
        estimation={"method": "espirit", "enabled": True},
        combine={"method": "sense"},
    )
    assert cp.compression.num_virtual_coils == 4
    assert cp.combine.method == "sense"


@pytest.mark.parametrize("block,bad", [
    ("compression", {"method": "bogus"}),
    ("estimation", {"method": "bogus"}),
    ("combine", {"method": "bogus"}),
])
def test_unknown_method_rejected(block, bad):
    with pytest.raises(ValidationError):
        CoilProcessingConfig(**{block: bad})


def test_mounted_on_physics():
    p = PhysicsConfigSchema()
    assert hasattr(p, "coil_processing")
    assert p.coil_processing.combine.method == "none"
    assert hasattr(p.coil_processing, "output")


def test_coil_processing_subschemas_are_frozen():
    """Regression WS1-physics-02: the five coil sub-schemas (compression /
    estimation / combine / output / processing) carried ``extra="forbid"`` but
    NOT ``frozen=True``. Pydantic v2 does not propagate the parent's frozen-ness,
    so a held reference could be mutated post-load, breaking NN#1. Every sibling
    physics sub-schema is frozen; these must be too.
    """
    cp = CoilProcessingConfig()
    # the bundle itself
    with pytest.raises(ValidationError):
        cp.combine = cp.combine  # type: ignore[misc]
    # and each nested sub-schema
    with pytest.raises(ValidationError):
        cp.compression.method = "svd"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        cp.estimation.method = "espirit"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        cp.combine.method = "rss"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        cp.output.domain = "image"  # type: ignore[misc]
