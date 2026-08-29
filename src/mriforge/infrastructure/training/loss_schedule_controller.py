"""Runtime controller for the declarative ``loss_schedule:`` block.

Turns a :class:`~mriforge.config.schemas.loss_schedule.LossScheduleConfigSchema`
into per-step ``{loss_term: weight}`` overrides that the loss computer reads via
the ``loop_state.loss_weight_overrides`` seam. It is the wired counterpart to the
(formerly inert) ``LossScheduler`` facade and the consumer of the early-stopping
plateau primitive.

Three capabilities, matching the three sub-questions the feature answers:

1. **Clock curriculum** -- ``iteration`` / ``epoch`` triggers drive
   set/scale/enable/disable/ramp actions at a fixed step. Ramps interpolate via
   :meth:`LossScheduler.compute_schedule`.
2. **Stagnation trigger** -- ``metric_plateau`` / ``loss_plateau`` triggers use a
   :class:`PlateauMonitor` (the same strict comparator as early stopping) to fire
   when a validation metric or training loss stops improving; ``metric_threshold``
   fires on a level crossing.
3. **Post-change monitoring** -- an optional ``monitor_after`` block watches the
   metric for a window of checks after a change and, if the expectation is not
   met, rolls back / holds / continues.

Lifecycle (driven by the training loop):

* :meth:`on_iteration` every step -- advances ramps + clock triggers.
* :meth:`on_validation` at each validation -- plateau/threshold triggers and
  post-change windows (they only have a metric to read at validation cadence).

Both return the current override map; the loop assigns it to
``loop_state.loss_weight_overrides``. Every fire/restore is appended to
:attr:`events` for provenance (#15).

Metric-key resolution mirrors early stopping: a configured ``val_ssim`` resolves
to the bare ``ssim`` the validator actually emits (the F-EARLYSTOP-FUZZY alias
class), and a metric that is never emitted warns once rather than silently
never-firing.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Any

from mriforge.config.schemas.enums import (
    ActionType,
    ExpectKind,
    MetricMode,
    OnFailureKind,
    RampShape,
    TriggerType,
)
from mriforge.config.schemas.loss_schedule import (
    LossScheduleConfigSchema,
    LossScheduleRule,
)
from mriforge.infrastructure.services.metric_keys import resolve_metric_key
from mriforge.infrastructure.training.loss_scheduler import LossScheduler
from mriforge.infrastructure.training.plateau_monitor import PlateauMonitor

logger = logging.getLogger(__name__)

# Cap on retained provenance events (bounded memory + artifact size for
# high-frequency ``every`` triggers; the most recent N are kept). See L5.
_MAX_EVENTS = 10_000


@dataclass
class _RampState:
    """An in-progress ``ramp`` action interpolating a term's weight over time."""

    target: str
    start_step: int
    from_value: float
    to_value: float
    over: int
    shape: RampShape


@dataclass
class _PostChangeState:
    """An active ``monitor_after`` window observing a metric after a change."""

    rule_name: str
    target: str
    metric: str
    mode: MetricMode
    window: int
    min_delta: float
    expect: ExpectKind
    on_failure: OnFailureKind
    pre_override: float | None  # override value before the change (for rollback)
    baseline: float | None = None  # captured at fire time, or deferred (clock)
    armed: bool = False  # True once a finite baseline is captured
    checks_done: int = 0
    checks_missing: int = 0  # validations where the metric was absent/non-finite
    improved_ever: bool = False
    worsened_ever: bool = False


def _strictly_better(mode: MetricMode, value: float, ref: float, min_delta: float) -> bool:
    if mode == MetricMode.MIN:
        return value < ref - min_delta
    return value > ref + min_delta


def _strictly_worse(mode: MetricMode, value: float, ref: float, min_delta: float) -> bool:
    if mode == MetricMode.MIN:
        return value > ref + min_delta
    return value < ref - min_delta


class LossScheduleController:
    """Resolve dynamic loss-term weight overrides from a ``loss_schedule:`` block."""

    def __init__(
        self,
        config: LossScheduleConfigSchema,
        loss_config: Any = None,
        start_iteration: int = 0,
        start_epoch: int = 0,
    ) -> None:
        """Args:
        config: The validated ``loss_schedule`` block.
        loss_config: The full training settings (or anything exposing
            ``.losses`` / ``.objectives``), used to resolve a target term's
            static base weight for ``scale`` / ``enable`` actions. Optional;
            missing => base 1.0.
        start_iteration: The iteration the loop resumes from. Clock ``at``
            triggers at or before it are pre-fired so a resumed run does not
            replay an already-completed curriculum (M4). Clock curricula are
            deterministic functions of the step, so this reconstructs their
            state without serializing the controller.
        start_epoch: The epoch the loop resumes from (for ``epoch`` triggers).
        """
        self.enabled = config.enabled
        self.rules: list[LossScheduleRule] = list(config.rules)
        self._loss_config = loss_config

        # Checkpoint-on-change throttle (interval is config-driven, never a
        # hardcoded literal). Validated at load: when checkpoint_on_change is on,
        # checkpoint_min_interval is guaranteed non-None by the schema.
        self.checkpoint_on_change = config.checkpoint_on_change
        self.checkpoint_min_interval = config.checkpoint_min_interval
        self._change_pending = False
        self._last_checkpoint_iteration: int | None = None

        self._overrides: dict[str, float] = {}
        self._base_weights: dict[str, float] = {}
        # Bounded provenance log (deque drops oldest beyond _MAX_EVENTS).
        self.events: deque[dict[str, Any]] = deque(maxlen=_MAX_EVENTS)

        # Per-rule runtime state.
        self._fired: dict[str, bool] = {}  # one-shot 'at' triggers
        self._last_fire: dict[str, int] = {}  # 'every' triggers + threshold de-dup
        self._monitors: dict[str, PlateauMonitor] = {}  # plateau/loss_plateau
        self._active_ramps: dict[str, _RampState] = {}  # keyed by rule name
        self._post: dict[str, _PostChangeState] = {}  # active monitor_after windows
        self._missing_key_warned: set[str] = set()  # warn-once per rule

        for rule in self.rules:
            t = rule.trigger
            if t.type in (TriggerType.METRIC_PLATEAU, TriggerType.LOSS_PLATEAU):
                self._monitors[rule.name] = PlateauMonitor(
                    mode=t.mode or MetricMode.MIN,
                    patience=int(t.patience),
                    min_delta=t.min_delta if t.min_delta is not None else 0.0,
                    cooldown=t.cooldown if t.cooldown is not None else 0,
                )

        if start_iteration > 0 or start_epoch > 0:
            self._seed_from_resume(start_iteration, start_epoch)

    # ------------------------------------------------------------------ resume

    def _seed_from_resume(self, start_iteration: int, start_epoch: int) -> None:
        """Pre-fire clock ``at`` triggers already past at resume time (M4).

        Without this, a fresh controller on resume re-fires every one-shot ``at``
        rule on the first resumed step and (for ramps) replays the whole curve
        from the resume point. Clock curricula are deterministic in the step, so
        we reconstruct their settled state here: completed ramps jump to
        ``to_value``; a ramp still mid-flight at resume is re-armed on its
        ORIGINAL ``start_step`` so it continues correctly.
        """
        for rule in self.rules:
            t = rule.trigger
            if t.type not in (TriggerType.ITERATION, TriggerType.EPOCH) or t.at is None:
                continue
            pos = start_iteration if t.type == TriggerType.ITERATION else start_epoch
            if pos < t.at:
                continue
            self._fired[rule.name] = True
            action = rule.action
            target = rule.target
            base = self._base_weight(target)
            if action.type == ActionType.SET_WEIGHT:
                self._overrides[target] = float(action.to)
            elif action.type == ActionType.ENABLE:
                self._overrides[target] = float(action.to) if action.to is not None else base
            elif action.type == ActionType.DISABLE:
                self._overrides[target] = 0.0
            elif action.type == ActionType.SCALE:
                new = base * float(action.factor)
                if action.min_weight is not None:
                    new = max(new, action.min_weight)
                if action.max_weight is not None:
                    new = min(new, action.max_weight)
                self._overrides[target] = new
            elif action.type == ActionType.RAMP:
                # Only iteration ramps interpolate per-step; reconstruct on the
                # ORIGINAL start_step (=at) so the curve resumes in phase.
                if t.type == TriggerType.ITERATION and start_iteration < t.at + int(action.over):
                    self._active_ramps[rule.name] = _RampState(
                        target=target,
                        start_step=int(t.at),
                        from_value=base,
                        to_value=float(action.to),
                        over=int(action.over),
                        shape=action.shape,
                    )
                    self._overrides[target] = LossScheduler.compute_schedule(
                        self._ramp_schedule(self._active_ramps[rule.name]),
                        base_value=float(action.to),
                        step=start_iteration,
                    )
                else:
                    self._overrides[target] = float(action.to)

    # ------------------------------------------------------------------ public

    def current_overrides(self) -> dict[str, float]:
        """Return a copy of the active ``{term: weight}`` override map."""
        return dict(self._overrides)

    def consume_checkpoint_request(self, iteration: int) -> bool:
        """Return ``True`` iff a checkpoint should be saved now for a fired change.

        A triggered weight change sets a pending flag (when ``checkpoint_on_change``
        is enabled). This returns ``True`` for the FIRST eligible step, then
        throttles: a change fired within ``checkpoint_min_interval`` iterations of
        the last save is DROPPED (the request is cleared, not deferred), so any
        save that does fire is stamped at the firing change's own iteration with
        genuinely pre-change model weights (M5). The interval is the user-set
        config value (no hardcoded fallback). Calling this consumes the request:
        on ``True`` it clears the pending flag and records ``iteration`` as the
        last save point; on a throttled change it also clears the pending flag.
        """
        if not self.checkpoint_on_change or not self._change_pending:
            return False
        interval = self.checkpoint_min_interval  # config-driven, validated non-None
        if (
            self._last_checkpoint_iteration is not None
            and (iteration - self._last_checkpoint_iteration) < interval
        ):
            # Drop, don't defer: a stale later save would capture post-change
            # weights at an unrelated iteration. The next distinct change saves
            # at its own step.
            self._change_pending = False
            return False
        self._change_pending = False
        self._last_checkpoint_iteration = iteration
        return True

    def on_iteration(self, iteration: int, epoch: int) -> dict[str, float]:
        """Advance ramps and fire clock (iteration/epoch) triggers."""
        if not self.enabled:
            return {}
        self._advance_ramps(iteration)
        for rule in self.rules:
            # Don't change again mid post-change window, nor restart a rule's own
            # still-running ramp (L2: an 'every' ramp must finish before re-arming).
            if rule.name in self._post or rule.name in self._active_ramps:
                continue
            if rule.trigger.type in (
                TriggerType.ITERATION,
                TriggerType.EPOCH,
            ) and self._clock_fires(rule, iteration, epoch):
                self._apply_action(
                    rule, iteration, reason=self._clock_reason(rule, iteration, epoch)
                )
        return self.current_overrides()

    def on_validation(
        self,
        iteration: int,
        epoch: int,
        val_metrics: dict[str, float] | None,
        train_losses: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Fire metric/loss plateau + threshold triggers and process post-change windows."""
        if not self.enabled:
            return {}
        val_metrics = val_metrics or {}
        train_losses = train_losses or {}

        # Snapshot windows that existed BEFORE trigger evaluation, so a window
        # started by a trigger firing this call does not also consume this same
        # metric as its first observation (its baseline is captured now).
        existing_post = set(self._post)

        for rule in self.rules:
            if rule.name in self._post:  # don't change again mid post-change window
                continue
            t = rule.trigger
            if t.type == TriggerType.METRIC_PLATEAU:
                key = self._resolve_key(rule, t.metric, val_metrics)
                if key is not None and self._monitors[rule.name].fire_if_plateaued(
                    float(val_metrics[key]), iteration
                ):
                    self._apply_action(
                        rule,
                        iteration,
                        reason=f"{t.metric} plateau",
                        val_metrics=val_metrics,
                    )
            elif t.type == TriggerType.LOSS_PLATEAU:
                key = self._resolve_key(rule, t.loss_key, train_losses)
                if key is not None and self._monitors[rule.name].fire_if_plateaued(
                    float(train_losses[key]), iteration
                ):
                    self._apply_action(
                        rule,
                        iteration,
                        reason=f"{t.loss_key} plateau",
                        val_metrics=val_metrics,
                    )
            elif t.type == TriggerType.METRIC_THRESHOLD:
                key = self._resolve_key(rule, t.metric, val_metrics)
                if key is not None and self._threshold_fires(rule, float(val_metrics[key])):
                    self._apply_action(
                        rule,
                        iteration,
                        reason=f"{t.metric} crossed {t.value}",
                        val_metrics=val_metrics,
                    )

        self._process_post_change(existing_post, iteration, val_metrics)
        return self.current_overrides()

    # ---------------------------------------------------------- metric keys

    def _resolve_key(
        self, rule: LossScheduleRule, name: str | None, available: dict[str, float]
    ) -> str | None:
        """Resolve a configured metric/loss key against the live dict.

        Reuses the early-stopping alias rules (``val_ssim`` -> ``ssim`` etc.).
        Warns once per rule when the key is never present, so a typo / never-
        emitted metric is visible instead of a silent never-fire (#10/#15).
        """
        if name is None:
            return None
        resolved = resolve_metric_key(name, available)
        if resolved is None and rule.name not in self._missing_key_warned and available:
            self._missing_key_warned.add(rule.name)
            logger.warning(
                "[loss_schedule] rule=%s: metric %r not found in %s; rule will not "
                "fire until it is emitted.",
                rule.name,
                name,
                sorted(available),
            )
        return resolved

    # ------------------------------------------------------------- clock logic

    def _clock_fires(self, rule: LossScheduleRule, iteration: int, epoch: int) -> bool:
        t = rule.trigger
        pos = iteration if t.type == TriggerType.ITERATION else epoch
        if t.at is not None:
            if pos >= t.at and not self._fired.get(rule.name):
                self._fired[rule.name] = True
                return True
            return False
        if (
            t.every is not None
            and pos > 0
            and pos % t.every == 0
            and self._last_fire.get(rule.name) != pos
        ):
            self._last_fire[rule.name] = pos
            return True
        return False

    def _clock_reason(self, rule: LossScheduleRule, iteration: int, epoch: int) -> str:
        unit = "iteration" if rule.trigger.type == TriggerType.ITERATION else "epoch"
        pos = iteration if rule.trigger.type == TriggerType.ITERATION else epoch
        return f"{unit}={pos}"

    def _threshold_fires(self, rule: LossScheduleRule, value: float) -> bool:
        """Fire once when ``value`` crosses ``trigger.value`` in the mode direction."""
        t = rule.trigger
        mode = t.mode or MetricMode.MIN
        crossed = value <= t.value if mode == MetricMode.MIN else value >= t.value
        if crossed and not self._fired.get(rule.name):
            self._fired[rule.name] = True
            return True
        return False

    @staticmethod
    def _ramp_schedule(ramp: _RampState) -> dict[str, Any]:
        return {
            "type": ("linear_warmup" if ramp.shape == RampShape.LINEAR else "sigmoid_rampup"),
            "start_step": ramp.start_step,
            ("warmup_steps" if ramp.shape == RampShape.LINEAR else "ramp_steps"): ramp.over,
            "initial_value": ramp.from_value,
            "final_value": ramp.to_value,
        }

    def _advance_ramps(self, iteration: int) -> None:
        for name, ramp in list(self._active_ramps.items()):
            self._overrides[ramp.target] = LossScheduler.compute_schedule(
                self._ramp_schedule(ramp), base_value=ramp.to_value, step=iteration
            )
            if iteration >= ramp.start_step + ramp.over:
                self._overrides[ramp.target] = ramp.to_value
                del self._active_ramps[name]

    # ----------------------------------------------------------- action apply

    def _base_weight(self, target: str) -> float:
        if target not in self._base_weights:
            self._base_weights[target] = self._resolve_static_weight(target)
        return self._base_weights[target]

    def _resolve_static_weight(self, target: str) -> float:
        """Resolve a term's static base weight via the SAME SSOT the loss computer uses,
        so ``scale`` / ``enable`` start from the configured base rather than a divergent
        controller-local default (M3)."""
        # Lazy import to keep controller import cheap and avoid any import cycle.
        from mriforge.models.losses.weights import build_loss_weight_table

        # `_loss_config` is the full TrainingSettings; the table is built from its
        # `losses` block.
        return build_loss_weight_table(getattr(self._loss_config, "losses", None)).weight(
            target, iteration=1_000_000
        )

    def _cancel_ramps_for(self, target: str, keep_rule: str | None = None) -> None:
        """Drop active ramps writing ``target`` so a later non-ramp action on the
        same term is authoritative (L6: otherwise the ramp clobbers it next step)."""
        for name in [
            n for n, r in self._active_ramps.items() if r.target == target and n != keep_rule
        ]:
            del self._active_ramps[name]

    def _apply_action(
        self,
        rule: LossScheduleRule,
        iteration: int,
        *,
        reason: str,
        val_metrics: dict[str, float] | None = None,
    ) -> None:
        target = rule.target
        action = rule.action
        base = self._base_weight(target)
        current = self._overrides.get(target, base)
        pre_override = self._overrides.get(target)  # None if never overridden

        if action.type == ActionType.SET_WEIGHT:
            self._cancel_ramps_for(target)
            self._overrides[target] = float(action.to)
        elif action.type == ActionType.SCALE:
            self._cancel_ramps_for(target)
            new = current * float(action.factor)
            if action.min_weight is not None:
                new = max(new, action.min_weight)
            if action.max_weight is not None:
                new = min(new, action.max_weight)
            self._overrides[target] = new
        elif action.type == ActionType.ENABLE:
            self._cancel_ramps_for(target)
            self._overrides[target] = float(action.to) if action.to is not None else base
        elif action.type == ActionType.DISABLE:
            self._cancel_ramps_for(target)
            self._overrides[target] = 0.0
        elif action.type == ActionType.RAMP:
            self._cancel_ramps_for(target, keep_rule=rule.name)
            self._active_ramps[rule.name] = _RampState(
                target=target,
                start_step=iteration,
                from_value=current,
                to_value=float(action.to),
                over=int(action.over),
                shape=action.shape,
            )
            self._overrides[target] = current  # immediate value; ramp advances it

        new_weight = self._overrides.get(target)
        self._record_event(rule, iteration, current, new_weight, reason)
        logger.info(
            "[loss_schedule] rule=%s target=%s %.4g -> %.4g (%s)",
            rule.name,
            target,
            current,
            new_weight if new_weight is not None else float("nan"),
            reason,
        )

        # Flag that a change fired so the loop can snapshot the pre-change model
        # (throttled by checkpoint_min_interval via consume_checkpoint_request).
        if self.checkpoint_on_change:
            self._change_pending = True

        # Start post-change monitoring for ANY trigger type (M2). When a metric
        # baseline is available at fire time (plateau/threshold/loss_plateau all
        # fire at validation) we capture it now from the MONITOR metric (M1);
        # clock triggers (no val_metrics here) defer baseline capture to the next
        # validation.
        if rule.monitor_after is not None:
            self._start_post_change(rule, pre_override=pre_override, val_metrics=val_metrics)

    # -------------------------------------------------------- post-change loop

    def _start_post_change(
        self,
        rule: LossScheduleRule,
        pre_override: float | None,
        val_metrics: dict[str, float] | None,
    ) -> None:
        ma = rule.monitor_after
        assert ma is not None
        baseline: float | None = None
        armed = False
        if val_metrics:
            key = resolve_metric_key(ma.metric, val_metrics)
            if key is not None and math.isfinite(float(val_metrics[key])):
                baseline = float(val_metrics[key])
                armed = True
        self._post[rule.name] = _PostChangeState(
            rule_name=rule.name,
            target=rule.target,
            metric=ma.metric,
            mode=ma.mode,
            window=ma.window,
            min_delta=ma.min_delta,
            expect=ma.expect,
            on_failure=ma.on_failure,
            pre_override=pre_override,
            baseline=baseline,
            armed=armed,
        )

    def _process_post_change(
        self, existing: set[str], iteration: int, val_metrics: dict[str, float]
    ) -> None:
        for name in list(existing):
            state = self._post.get(name)
            if state is None:
                continue
            key = resolve_metric_key(state.metric, val_metrics)
            value = float(val_metrics[key]) if key is not None else None

            # Deferred baseline (clock triggers): capture on the first validation
            # that supplies a finite value; do not count it as a window check.
            if not state.armed:
                if value is not None and math.isfinite(value):
                    state.baseline = value
                    state.armed = True
                else:
                    state.checks_missing += 1
                    if state.checks_missing >= state.window:
                        self._finish_post_change(state, iteration, missing=True)
                        del self._post[name]
                continue

            if value is None or not math.isfinite(value):
                # Metric absent / non-finite: do not burn a window check on it,
                # but time out if it is permanently missing (M6/M7).
                state.checks_missing += 1
                if state.checks_missing >= state.window:
                    self._finish_post_change(state, iteration, missing=True)
                    del self._post[name]
                continue

            assert state.baseline is not None
            if _strictly_better(state.mode, value, state.baseline, state.min_delta):
                state.improved_ever = True
            if _strictly_worse(state.mode, value, state.baseline, state.min_delta):
                state.worsened_ever = True
            state.checks_done += 1
            if state.checks_done >= state.window:
                self._finish_post_change(state, iteration)
                del self._post[name]

    def _finish_post_change(
        self, state: _PostChangeState, iteration: int, missing: bool = False
    ) -> None:
        if missing:
            success = False
            reason = "metric_missing"
        elif state.expect == ExpectKind.IMPROVE:
            success = state.improved_ever
            reason = None
        else:  # NOT_WORSE
            success = not state.worsened_ever
            reason = None

        if success:
            self.events.append(
                {
                    "rule": state.rule_name,
                    "iteration": iteration,
                    "target": state.target,
                    "monitor_after": "ok",
                    "metric": state.metric,
                    "expect": state.expect.value,
                }
            )
            return

        # Expectation failed (or metric missing) -> apply on_failure.
        if state.on_failure == OnFailureKind.ROLLBACK:
            if state.pre_override is None:
                self._overrides.pop(state.target, None)
            else:
                self._overrides[state.target] = state.pre_override
            verdict = "rollback"
        elif state.on_failure == OnFailureKind.HOLD:
            verdict = "hold"
        else:  # CONTINUE
            verdict = "continue"

        tag = f"failed:{verdict}" if reason is None else f"failed:{verdict}:{reason}"
        self.events.append(
            {
                "rule": state.rule_name,
                "iteration": iteration,
                "target": state.target,
                "monitor_after": tag,
                "metric": state.metric,
                "expect": state.expect.value,
            }
        )
        logger.warning(
            "[loss_schedule] rule=%s monitor_after FAILED (%s on %s%s) -> %s",
            state.rule_name,
            state.expect.value,
            state.metric,
            " missing" if missing else "",
            verdict,
        )

    def _record_event(
        self,
        rule: LossScheduleRule,
        iteration: int,
        old: float,
        new: float | None,
        reason: str,
    ) -> None:
        self.events.append(
            {
                "rule": rule.name,
                "iteration": iteration,
                "target": rule.target,
                "action": rule.action.type.value,
                "old_weight": old,
                "new_weight": new,
                "trigger": reason,
            }
        )


__all__ = ["LossScheduleController"]
