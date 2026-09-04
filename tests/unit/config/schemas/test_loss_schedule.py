"""Tests for the ``loss_schedule:`` config schema.

Targets ``spectramr.config.schemas.loss_schedule``. The schema declares dynamic
loss-term scheduling rules (clock curriculum, plateau triggers, post-change
monitoring). Validation discipline:

- valid rules of every trigger/action kind load
- an unknown ``trigger.type`` / ``action.type`` is rejected (#9 enum coercion)
- a field that the declared type would never read RAISES (#15 no silent no-op)
- duplicate rule names raise
- ``extra='forbid'`` rejects stray keys
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.enums import (
    ActionType,
    MetricMode,
    OnFailureKind,
    RampShape,
    TriggerType,
)
from spectramr.config.schemas.loss_schedule import (
    ActionConfig,
    LossScheduleConfigSchema,
    LossScheduleRule,
    MonitorAfterConfig,
    TriggerConfig,
)

# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_iteration_ramp_rule_loads() -> None:
    rule = LossScheduleRule(
        name="perceptual_warmin",
        target="perceptual",
        trigger={"type": "iteration", "at": 5000},
        action={"type": "ramp", "to": 0.1, "over": 1000},
    )
    assert rule.trigger.type is TriggerType.ITERATION
    assert rule.trigger.at == 5000
    assert rule.action.type is ActionType.RAMP
    assert rule.action.shape is RampShape.LINEAR  # default


def test_metric_plateau_with_monitor_after_loads() -> None:
    rule = LossScheduleRule(
        name="relax_l1_on_ssim_plateau",
        target="l1",
        trigger={
            "type": "metric_plateau",
            "metric": "val_ssim",
            "mode": "max",
            "patience": 3,
            "min_delta": 0.001,
            "cooldown": 2,
        },
        action={"type": "scale", "factor": 0.5, "min_weight": 0.1},
        monitor_after={"metric": "val_ssim", "mode": "max", "window": 3, "on_failure": "rollback"},
    )
    assert rule.trigger.mode is MetricMode.MAX
    assert rule.action.factor == 0.5
    assert isinstance(rule.monitor_after, MonitorAfterConfig)
    assert rule.monitor_after.on_failure is OnFailureKind.ROLLBACK


def test_top_level_schema_holds_rules() -> None:
    cfg = LossScheduleConfigSchema(
        enabled=True,
        rules=[
            {"name": "r1", "target": "l1", "trigger": {"type": "epoch", "at": 10},
             "action": {"type": "disable"}},
        ],
    )
    assert cfg.enabled is True
    assert len(cfg.rules) == 1


@pytest.mark.parametrize(
    "trigger,action",
    [
        ({"type": "iteration", "every": 1000}, {"type": "set_weight", "to": 1.0}),
        ({"type": "loss_plateau", "loss_key": "g_total_loss", "patience": 5}, {"type": "enable"}),
        # metric triggers require an explicit mode (C3 — no safe default).
        (
            {"type": "metric_threshold", "metric": "val_psnr", "value": 30.0, "mode": "max"},
            {"type": "disable"},
        ),
    ],
)
def test_each_trigger_action_combo_loads(trigger: dict, action: dict) -> None:
    LossScheduleRule(name="x", target="l1", trigger=trigger, action=action)


# ---------------------------------------------------------------------------
# #9 — unknown enum value rejected
# ---------------------------------------------------------------------------


def test_unknown_trigger_type_raises() -> None:
    with pytest.raises(ValidationError):
        TriggerConfig(type="on_full_moon", at=10)


def test_unknown_action_type_raises() -> None:
    with pytest.raises(ValidationError):
        ActionConfig(type="teleport", to=1.0)


# ---------------------------------------------------------------------------
# #15 — a field the declared type never reads is a silent no-op -> RAISE
# ---------------------------------------------------------------------------


def test_iteration_trigger_rejects_plateau_field() -> None:
    with pytest.raises(ValidationError, match="silent no-op"):
        TriggerConfig(type="iteration", at=10, patience=3)


def test_metric_plateau_requires_metric_and_patience() -> None:
    with pytest.raises(ValidationError, match="requires"):
        TriggerConfig(type="metric_plateau", metric="val_ssim")  # missing patience


def test_iteration_trigger_requires_at_or_every() -> None:
    with pytest.raises(ValidationError, match="'at' or 'every'"):
        TriggerConfig(type="iteration")


def test_scale_action_rejects_to_field() -> None:
    with pytest.raises(ValidationError, match="silent no-op"):
        ActionConfig(type="scale", factor=0.5, to=1.0)


def test_ramp_action_requires_to_and_over() -> None:
    with pytest.raises(ValidationError, match="requires"):
        ActionConfig(type="ramp", to=0.1)  # missing 'over'


def test_disable_action_rejects_weight_fields() -> None:
    with pytest.raises(ValidationError, match="silent no-op"):
        ActionConfig(type="disable", to=0.0)


# ---------------------------------------------------------------------------
# Structural strictness
# ---------------------------------------------------------------------------


def test_duplicate_rule_names_raise() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        LossScheduleConfigSchema(
            rules=[
                {"name": "dup", "target": "l1", "trigger": {"type": "epoch", "at": 1},
                 "action": {"type": "disable"}},
                {"name": "dup", "target": "l2", "trigger": {"type": "epoch", "at": 2},
                 "action": {"type": "disable"}},
            ]
        )


def test_extra_key_is_forbidden() -> None:
    with pytest.raises(ValidationError):
        TriggerConfig(type="iteration", at=10, bogus=True)


def test_schema_is_frozen() -> None:
    cfg = LossScheduleConfigSchema(enabled=True)
    with pytest.raises(ValidationError):
        cfg.enabled = False


# ---------------------------------------------------------------------------
# Checkpoint-on-change: interval must be user-set when enabled (#15, no
# hardcoded fallback)
# ---------------------------------------------------------------------------


def test_checkpoint_defaults_off_with_no_interval() -> None:
    cfg = LossScheduleConfigSchema(enabled=True)
    assert cfg.checkpoint_on_change is False
    assert cfg.checkpoint_min_interval is None


def test_checkpoint_on_change_requires_interval() -> None:
    with pytest.raises(ValidationError, match="checkpoint_min_interval"):
        LossScheduleConfigSchema(enabled=True, checkpoint_on_change=True)  # no interval


def test_checkpoint_on_change_with_interval_loads() -> None:
    cfg = LossScheduleConfigSchema(
        enabled=True, checkpoint_on_change=True, checkpoint_min_interval=500
    )
    assert cfg.checkpoint_on_change is True
    assert cfg.checkpoint_min_interval == 500


def test_checkpoint_interval_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        LossScheduleConfigSchema(
            enabled=True, checkpoint_on_change=True, checkpoint_min_interval=0
        )


# ---------------------------------------------------------------------------
# C3 — metric triggers require an explicit mode (no silent MIN default)
# ---------------------------------------------------------------------------


def test_metric_plateau_requires_mode() -> None:
    with pytest.raises(ValidationError, match="requires"):
        TriggerConfig(type="metric_plateau", metric="val_ssim", patience=2)  # no mode


def test_metric_threshold_requires_mode() -> None:
    with pytest.raises(ValidationError, match="requires"):
        TriggerConfig(type="metric_threshold", metric="val_psnr", value=30.0)  # no mode


def test_loss_plateau_does_not_require_mode() -> None:
    # A loss is naturally MIN; mode stays optional for loss_plateau.
    t = TriggerConfig(type="loss_plateau", loss_key="g_total_loss", patience=2)
    assert t.mode is None


def test_loss_plateau_allows_mode_max() -> None:
    # The controller reads t.mode for loss_plateau, so the schema must allow it (L3).
    t = TriggerConfig(type="loss_plateau", loss_key="margin", patience=2, mode="max")
    assert t.mode is MetricMode.MAX


# ---------------------------------------------------------------------------
# L3 — cooldown/min_delta forbidden where unread; min_weight/max_weight only on scale
# ---------------------------------------------------------------------------


def test_cooldown_forbidden_on_iteration_trigger() -> None:
    with pytest.raises(ValidationError, match="silent no-op"):
        TriggerConfig(type="iteration", at=10, cooldown=5)


def test_min_delta_forbidden_on_threshold_trigger() -> None:
    with pytest.raises(ValidationError, match="silent no-op"):
        TriggerConfig(type="metric_threshold", metric="val_psnr", value=30.0, mode="max", min_delta=0.1)


def test_cooldown_allowed_on_metric_plateau() -> None:
    t = TriggerConfig(type="metric_plateau", metric="val_ssim", mode="max", patience=2, cooldown=3)
    assert t.cooldown == 3


@pytest.mark.parametrize("atype", ["set_weight", "ramp", "enable"])
def test_min_weight_forbidden_on_non_scale_actions(atype: str) -> None:
    kwargs: dict = {"set_weight": {"to": 1.0}, "ramp": {"to": 1.0, "over": 10}, "enable": {}}[atype]
    with pytest.raises(ValidationError, match="silent no-op"):
        ActionConfig(type=atype, min_weight=0.1, **kwargs)


def test_scale_still_accepts_clamps() -> None:
    a = ActionConfig(type="scale", factor=0.5, min_weight=0.1, max_weight=2.0)
    assert a.min_weight == 0.1 and a.max_weight == 2.0


# ---------------------------------------------------------------------------
# L1 — a plateau trigger driving a compounding action needs a guard
# ---------------------------------------------------------------------------


def test_plateau_scale_without_cooldown_or_monitor_raises() -> None:
    with pytest.raises(ValidationError, match="compounds"):
        LossScheduleRule(
            name="r", target="l1",
            trigger={"type": "metric_plateau", "metric": "val_ssim", "mode": "max", "patience": 2},
            action={"type": "scale", "factor": 0.5},
        )


def test_plateau_scale_with_cooldown_loads() -> None:
    rule = LossScheduleRule(
        name="r", target="l1",
        trigger={"type": "metric_plateau", "metric": "val_ssim", "mode": "max",
                 "patience": 2, "cooldown": 2},
        action={"type": "scale", "factor": 0.5},
    )
    assert rule.trigger.cooldown == 2


def test_plateau_scale_with_monitor_after_loads() -> None:
    rule = LossScheduleRule(
        name="r", target="l1",
        trigger={"type": "metric_plateau", "metric": "val_ssim", "mode": "max", "patience": 2},
        action={"type": "scale", "factor": 0.5},
        monitor_after={"metric": "val_ssim", "mode": "max", "window": 2, "on_failure": "rollback"},
    )
    assert rule.monitor_after is not None


def test_plateau_disable_needs_no_cooldown() -> None:
    # disable is idempotent (not compounding) -> no guard required.
    rule = LossScheduleRule(
        name="r", target="l1",
        trigger={"type": "loss_plateau", "loss_key": "g_total_loss", "patience": 2},
        action={"type": "disable"},
    )
    assert rule.action.type is ActionType.DISABLE
