"""Tests for ``CheckpointConfigSchema``.

Targets ``spectramr.config.schemas.checkpoint``. Validates: defaults, ge
constraints, ``frozen=True``, ``extra='ignore'`` (lenient), validation
alias ``save_dir → checkpoint_dir``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.checkpoint import CheckpointConfigSchema


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_construction() -> None:
    """Default checkpoint config has expected values."""
    cfg = CheckpointConfigSchema()
    assert cfg.enabled is True
    assert cfg.checkpoint_dir == "./checkpoints"
    assert cfg.save_interval == 1000
    assert cfg.keep_last_n == 5
    assert cfg.keep_best_n == 3
    assert cfg.format == "safetensors"
    assert cfg.pretrained_path is None
    assert cfg.resume_training is False
    assert cfg.produced_by_arm is None


# ---------------------------------------------------------------------------
# Campaign-dependency declaration: produced_by_arm
# ---------------------------------------------------------------------------


def test_produced_by_arm_accepts_nonempty_name() -> None:
    """``produced_by_arm`` records the upstream campaign arm that builds the ckpt."""
    cfg = CheckpointConfigSchema(produced_by_arm="baseline_m4raw_unet_4x")
    assert cfg.produced_by_arm == "baseline_m4raw_unet_4x"


def test_produced_by_arm_rejects_empty_string() -> None:
    """An empty producer name is a meaningless declaration (min_length=1)."""
    with pytest.raises(ValidationError):
        CheckpointConfigSchema(produced_by_arm="")


def test_explicit_construction() -> None:
    """Explicit values are preserved."""
    cfg = CheckpointConfigSchema(
        enabled=False,
        save_interval=500,
        keep_last_n=10,
        format="pth",
    )
    assert cfg.enabled is False
    assert cfg.save_interval == 500
    assert cfg.keep_last_n == 10
    assert cfg.format == "pth"


# ---------------------------------------------------------------------------
# Validation alias: save_dir → checkpoint_dir
# ---------------------------------------------------------------------------


def test_validation_alias_save_dir_resolves() -> None:
    """``save_dir`` (alias) resolves to ``checkpoint_dir``."""
    cfg = CheckpointConfigSchema(save_dir="/scratch/checkpoints")
    assert cfg.checkpoint_dir == "/scratch/checkpoints"


def test_canonical_field_name_still_works() -> None:
    """``checkpoint_dir`` (canonical name) also accepted (``populate_by_name=True``)."""
    cfg = CheckpointConfigSchema(checkpoint_dir="/scratch/x")
    assert cfg.checkpoint_dir == "/scratch/x"


# ---------------------------------------------------------------------------
# Numeric constraints
# ---------------------------------------------------------------------------


def test_save_interval_ge_one() -> None:
    """``save_interval >= 1``."""
    with pytest.raises(ValidationError):
        CheckpointConfigSchema(save_interval=0)


def test_keep_last_n_ge_one() -> None:
    """``keep_last_n >= 1``."""
    with pytest.raises(ValidationError):
        CheckpointConfigSchema(keep_last_n=0)


def test_keep_best_n_ge_one() -> None:
    """``keep_best_n >= 1``."""
    with pytest.raises(ValidationError):
        CheckpointConfigSchema(keep_best_n=0)


# ---------------------------------------------------------------------------
# Frozen + extra='ignore'
# ---------------------------------------------------------------------------


def test_schema_is_frozen() -> None:
    """Cannot mutate after construction."""
    cfg = CheckpointConfigSchema()
    with pytest.raises(ValidationError):
        cfg.save_interval = 99


def test_extra_fields_silently_ignored() -> None:
    """``extra='ignore'`` accepts unknown fields without raising."""
    # Should NOT raise — extra='ignore' is lenient.
    cfg = CheckpointConfigSchema(unknown_legacy_field=42)
    assert not hasattr(cfg, "unknown_legacy_field")
