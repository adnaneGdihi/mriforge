"""Declarative loss-schedule / curriculum configuration.

A ``loss_schedule:`` block lets a YAML drive *dynamic* loss-term weights during
training, three ways:

1. **Clock curriculum** — ramp/enable/disable a term at a fixed iteration or
   epoch (a configurable generalisation of the hardcoded spatial-loss warmup
   gate).
2. **Stagnation trigger** — change a term's weight when a validation metric or a
   training loss *plateaus* (reusing the same plateau detection as early
   stopping).
3. **Post-change monitoring** — after a change fires, watch the metric for a
   window of checks and roll back / hold / continue based on whether it reacted.

The block is consumed by
:class:`~mriforge.infrastructure.training.loss_schedule_controller.LossScheduleController`
and is mounted on :class:`~mriforge.config.settings.TrainingSettings` as
``loss_schedule`` (parallel to ``early_stopping``).

Validation discipline
----------------------
Schemas are ``StrictSchema`` (``extra='forbid'``, frozen). An unknown
``trigger.type`` / ``action.type`` is rejected natively by the str-Enum
coercion (#9, no silent fallback). In addition, each block carries a
``model_validator`` that **raises** when a field is present that the declared
type would never read — an unread knob is a silent no-op (#15), so we fail at
load time rather than degrade quietly.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from .enums import (
    ActionType,
    ExpectKind,
    MetricMode,
    OnFailureKind,
    RampShape,
    TriggerType,
)
from .strictness import StrictSchema

# Fields each trigger type MUST declare, and fields it must NOT declare (because
# they belong to a different type and would otherwise be a silent no-op).
_TRIGGER_REQUIRED: dict[TriggerType, tuple[str, ...]] = {
    TriggerType.ITERATION: (),  # 'at' or 'every' enforced separately
    TriggerType.EPOCH: (),
    # 'mode' is required for metric triggers: no safe default exists because the
    # metric's natural direction is unknown (SSIM/PSNR are MAX, loss is MIN), and
    # a silent MIN default fires backwards on MAX metrics (C3).
    TriggerType.METRIC_PLATEAU: ("metric", "patience", "mode"),
    TriggerType.LOSS_PLATEAU: ("loss_key", "patience"),  # mode optional, MIN default ok for a loss
    TriggerType.METRIC_THRESHOLD: ("metric", "value", "mode"),
}
_TRIGGER_FORBIDDEN: dict[TriggerType, tuple[str, ...]] = {
    # cooldown/min_delta are only read for plateau triggers; forbidding them on
    # clock/threshold triggers closes the silent-no-op hole (L3).
    TriggerType.ITERATION: (
        "metric",
        "patience",
        "loss_key",
        "value",
        "mode",
        "min_delta",
        "cooldown",
    ),
    TriggerType.EPOCH: ("metric", "patience", "loss_key", "value", "mode", "min_delta", "cooldown"),
    TriggerType.METRIC_PLATEAU: ("at", "every", "loss_key", "value"),
    # 'mode' is read for loss_plateau (the controller builds its PlateauMonitor
    # with t.mode), so it must NOT be forbidden (L3): a MAX-direction loss
    # plateau is a legitimate config.
    TriggerType.LOSS_PLATEAU: ("at", "every", "metric", "value"),
    TriggerType.METRIC_THRESHOLD: ("at", "every", "patience", "loss_key", "min_delta", "cooldown"),
}

_ACTION_REQUIRED: dict[ActionType, tuple[str, ...]] = {
    ActionType.SET_WEIGHT: ("to",),
    ActionType.SCALE: ("factor",),
    ActionType.RAMP: ("to", "over"),
    ActionType.ENABLE: (),
    ActionType.DISABLE: (),
}
# Only SCALE reads min_weight/max_weight; every other action forbids them so an
# assumed-but-ignored clamp fails at load instead of silently no-op'ing (L3).
_ACTION_FORBIDDEN: dict[ActionType, tuple[str, ...]] = {
    ActionType.SET_WEIGHT: ("factor", "over", "min_weight", "max_weight"),
    ActionType.SCALE: ("to", "over"),
    ActionType.RAMP: ("factor", "min_weight", "max_weight"),
    ActionType.ENABLE: ("factor", "over", "min_weight", "max_weight"),
    ActionType.DISABLE: ("to", "factor", "over", "min_weight", "max_weight"),
}


class TriggerConfig(StrictSchema):
    """When a rule fires. Fields are type-specific; mismatches raise at load."""

    type: TriggerType = Field(description="Trigger kind (see TriggerType).")
    # clock (iteration / epoch)
    at: int | None = Field(default=None, ge=0, description="Fire once at this iteration/epoch.")
    every: int | None = Field(default=None, ge=1, description="Fire every N iterations/epochs.")
    # metric / loss plateau + threshold
    metric: str | None = Field(default=None, description="val_metrics key (plateau/threshold).")
    loss_key: str | None = Field(default=None, description="Training-loss key (loss_plateau).")
    mode: MetricMode | None = Field(
        default=None, description="Improvement direction; defaults MIN."
    )
    patience: int | None = Field(
        default=None, ge=1, description="Non-improving checks before firing."
    )
    # Optional (None) so the forbidden-field check can reject them on trigger
    # types that never read them (L3). Controller defaults None -> 0.0 / 0.
    min_delta: float | None = Field(
        default=None, ge=0.0, description="Strict improvement margin (plateau)."
    )
    cooldown: int | None = Field(
        default=None, ge=0, description="Checks suppressed after a fire (plateau)."
    )
    value: float | None = Field(default=None, description="Threshold value (metric_threshold).")

    @model_validator(mode="after")
    def _fields_match_type(self) -> TriggerConfig:
        t = self.type
        for f in _TRIGGER_REQUIRED[t]:
            if getattr(self, f) is None:
                raise ValueError(f"loss_schedule trigger type={t.value!r} requires '{f}'.")
        if (
            t in (TriggerType.ITERATION, TriggerType.EPOCH)
            and self.at is None
            and self.every is None
        ):
            raise ValueError(f"loss_schedule trigger type={t.value!r} requires 'at' or 'every'.")
        for f in _TRIGGER_FORBIDDEN[t]:
            if getattr(self, f) is not None:
                raise ValueError(
                    f"loss_schedule trigger type={t.value!r} does not read '{f}' "
                    f"(would be a silent no-op; CLAUDE.md #15). Remove it."
                )
        return self


class ActionConfig(StrictSchema):
    """What a rule does to its target term's weight."""

    type: ActionType = Field(description="Action kind (see ActionType).")
    to: float | None = Field(
        default=None, ge=0.0, description="Target weight (set_weight/ramp/enable)."
    )
    factor: float | None = Field(default=None, ge=0.0, description="Multiplier (scale).")
    over: int | None = Field(default=None, ge=1, description="Ramp duration in steps (ramp).")
    shape: RampShape = Field(default=RampShape.LINEAR, description="Ramp interpolation shape.")
    min_weight: float | None = Field(default=None, ge=0.0, description="Lower clamp (scale).")
    max_weight: float | None = Field(default=None, ge=0.0, description="Upper clamp (scale).")

    @model_validator(mode="after")
    def _fields_match_type(self) -> ActionConfig:
        t = self.type
        for f in _ACTION_REQUIRED[t]:
            if getattr(self, f) is None:
                raise ValueError(f"loss_schedule action type={t.value!r} requires '{f}'.")
        for f in _ACTION_FORBIDDEN[t]:
            if getattr(self, f) is not None:
                raise ValueError(
                    f"loss_schedule action type={t.value!r} does not read '{f}' "
                    f"(would be a silent no-op; CLAUDE.md #15). Remove it."
                )
        return self


class MonitorAfterConfig(StrictSchema):
    """Post-change feedback: watch a metric for ``window`` checks after firing."""

    metric: str = Field(description="val_metrics key to observe after the change.")
    mode: MetricMode = Field(default=MetricMode.MIN, description="Improvement direction.")
    window: int = Field(default=3, ge=1, description="Checks to observe after firing.")
    min_delta: float = Field(default=0.0, ge=0.0, description="Strict improvement margin.")
    expect: ExpectKind = Field(
        default=ExpectKind.IMPROVE,
        description="Whether the metric must improve or just not worsen.",
    )
    on_failure: OnFailureKind = Field(
        default=OnFailureKind.HOLD, description="What to do if the expectation is not met."
    )


class LossScheduleRule(StrictSchema):
    """A single (target, trigger, action[, monitor_after]) rule."""

    name: str = Field(description="Unique human label, stamped into provenance events.")
    target: str = Field(description="Loss-term name whose weight this rule changes.")
    trigger: TriggerConfig
    action: ActionConfig
    monitor_after: MonitorAfterConfig | None = Field(
        default=None, description="Optional post-change metric monitoring."
    )

    @model_validator(mode="after")
    def _plateau_compounding_action_needs_guard(self) -> LossScheduleRule:
        # A plateau trigger with cooldown=0 (the default) re-fires every
        # `patience` checks for the whole duration of a flat metric. For the
        # COMPOUNDING actions (scale multiplies the current weight; ramp re-arms)
        # that silently runs away (0.5 -> 0.25 -> 0.125 ...). Require an explicit
        # cooldown >= 1, or a monitor_after block (which locks re-firing during
        # its window), so the run-away is opt-in, not the default (L1).
        if (
            self.trigger.type in (TriggerType.METRIC_PLATEAU, TriggerType.LOSS_PLATEAU)
            and self.action.type in (ActionType.SCALE, ActionType.RAMP)
            and not (self.trigger.cooldown is not None and self.trigger.cooldown >= 1)
            and self.monitor_after is None
        ):
            raise ValueError(
                f"loss_schedule rule {self.name!r}: a {self.trigger.type.value} trigger "
                f"driving a {self.action.type.value} action compounds on every re-fire of a "
                f"sustained plateau. Set trigger.cooldown >= 1 (or add a monitor_after block) "
                f"to bound the re-fire rate."
            )
        return self


class LossScheduleConfigSchema(StrictSchema):
    """Top-level ``loss_schedule:`` block."""

    enabled: bool = Field(default=False, description="Enable dynamic loss scheduling.")
    rules: list[LossScheduleRule] = Field(
        default_factory=list, description="Ordered list of schedule rules."
    )
    # Checkpoint-on-change: snapshot the model before a triggered weight change
    # takes effect, so the pre-change state is recoverable. Throttled by a
    # user-set iteration interval (never hardcoded): when triggers fire often,
    # at most one such checkpoint is saved per ``checkpoint_min_interval`` steps.
    checkpoint_on_change: bool = Field(
        default=False,
        description="Save a checkpoint before a triggered loss-weight change proceeds.",
    )
    checkpoint_min_interval: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Minimum iterations between checkpoint-on-change saves (throttle for "
            "frequent triggers). REQUIRED when checkpoint_on_change is true; there "
            "is no hardcoded default — set it in the config or via the API."
        ),
    )

    @model_validator(mode="after")
    def _unique_rule_names(self) -> LossScheduleConfigSchema:
        seen: set[str] = set()
        for rule in self.rules:
            if rule.name in seen:
                raise ValueError(f"duplicate loss_schedule rule name {rule.name!r}.")
            seen.add(rule.name)
        return self

    @model_validator(mode="after")
    def _require_checkpoint_interval(self) -> LossScheduleConfigSchema:
        # #15: an advertised knob must be wired to a concrete value, not a silent
        # hardcoded fallback. If saving-on-change is on, the throttle interval
        # must be set explicitly by the user (config or API).
        if self.checkpoint_on_change and self.checkpoint_min_interval is None:
            raise ValueError(
                "loss_schedule.checkpoint_on_change is true but "
                "checkpoint_min_interval is unset. Set the throttle interval "
                "(iterations) explicitly; there is no hardcoded default."
            )
        return self


__all__ = [
    "ActionConfig",
    "LossScheduleConfigSchema",
    "LossScheduleRule",
    "MonitorAfterConfig",
    "TriggerConfig",
]
