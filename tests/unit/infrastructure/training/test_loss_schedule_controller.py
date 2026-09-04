"""Tests for ``LossScheduleController``.

Targets ``spectramr.infrastructure.training.loss_schedule_controller`` — the
runtime that turns a ``loss_schedule:`` block into per-step ``{term: weight}``
overrides. Covers each trigger kind, each action kind, ramp parity with
``LossScheduler``, and the ``monitor_after`` rollback/hold/continue paths.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from spectramr.config.schemas.loss import LossConfigSchema
from spectramr.config.schemas.loss_schedule import LossScheduleConfigSchema
from spectramr.infrastructure.training.loss_schedule_controller import (
    LossScheduleController,
)
from spectramr.infrastructure.training.loss_scheduler import LossScheduler


def _ctrl(rules, loss_config=None) -> LossScheduleController:
    cfg = LossScheduleConfigSchema(enabled=True, rules=rules)
    return LossScheduleController(cfg, loss_config=loss_config)


def _losscfg(**lambdas) -> SimpleNamespace:
    """A stand-in settings object exposing ``.losses``.

    Builds a REAL ``LossConfigSchema``: the base weight is now resolved by the loss-weight
    SSOT, which distinguishes an author-written lambda from a schema default via
    ``model_fields_set``. A ``SimpleNamespace`` has no such thing, so a fake config would
    silently declare nothing and every term would fall back to its schema default.
    """
    return SimpleNamespace(losses=LossConfigSchema(reconstruction=lambdas))


# ---------------------------------------------------------------------------
# Disabled
# ---------------------------------------------------------------------------


def test_disabled_controller_is_noop() -> None:
    cfg = LossScheduleConfigSchema(
        enabled=False,
        rules=[
            {
                "name": "r",
                "target": "l1",
                "trigger": {"type": "iteration", "at": 10},
                "action": {"type": "disable"},
            },
        ],
    )
    ctrl = LossScheduleController(cfg)
    assert ctrl.on_iteration(100, 0) == {}
    assert ctrl.on_validation(100, 0, {"val_ssim": 0.5}) == {}


# ---------------------------------------------------------------------------
# Clock triggers
# ---------------------------------------------------------------------------


def test_iteration_at_set_weight_fires_once() -> None:
    ctrl = _ctrl(
        [
            {
                "name": "r",
                "target": "l1",
                "trigger": {"type": "iteration", "at": 5000},
                "action": {"type": "set_weight", "to": 2.0},
            },
        ]
    )
    assert ctrl.on_iteration(4999, 0) == {}  # before
    assert ctrl.on_iteration(5000, 0) == {"l1": 2.0}  # fires
    assert ctrl.on_iteration(5001, 0) == {"l1": 2.0}  # persists, not re-fired
    assert sum(1 for e in ctrl.events if e.get("action") == "set_weight") == 1


def test_epoch_at_disable_fires() -> None:
    ctrl = _ctrl(
        [
            {
                "name": "r",
                "target": "l1",
                "trigger": {"type": "epoch", "at": 3},
                "action": {"type": "disable"},
            },
        ]
    )
    assert ctrl.on_iteration(100, 2) == {}
    assert ctrl.on_iteration(200, 3) == {"l1": 0.0}


def test_every_fires_each_interval() -> None:
    ctrl = _ctrl(
        [
            {
                "name": "r",
                "target": "l1",
                "trigger": {"type": "iteration", "every": 100},
                "action": {"type": "scale", "factor": 0.5},
            },
        ],
        loss_config=_losscfg(lambda_l1=1.0),
    )
    ctrl.on_iteration(100, 0)  # 1.0 * 0.5 = 0.5
    assert ctrl.current_overrides()["l1"] == pytest.approx(0.5)
    ctrl.on_iteration(150, 0)  # not a multiple -> unchanged
    assert ctrl.current_overrides()["l1"] == pytest.approx(0.5)
    ctrl.on_iteration(200, 0)  # 0.5 * 0.5 = 0.25 (scales current)
    assert ctrl.current_overrides()["l1"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Actions: scale clamp, enable from base
# ---------------------------------------------------------------------------


def test_scale_respects_min_weight_clamp() -> None:
    ctrl = _ctrl(
        [
            {
                "name": "r",
                "target": "l1",
                "trigger": {"type": "iteration", "at": 10},
                "action": {"type": "scale", "factor": 0.01, "min_weight": 0.1},
            },
        ],
        loss_config=_losscfg(lambda_l1=1.0),
    )
    ctrl.on_iteration(10, 0)
    assert ctrl.current_overrides()["l1"] == pytest.approx(0.1)  # clamped, not 0.01


def test_enable_resolves_base_weight_from_config() -> None:
    ctrl = _ctrl(
        [
            {
                "name": "r",
                "target": "perceptual",
                "trigger": {"type": "iteration", "at": 10},
                "action": {"type": "enable"},
            },
        ],
        loss_config=_losscfg(lambda_perceptual=0.3),
    )
    ctrl.on_iteration(10, 0)
    assert ctrl.current_overrides()["perceptual"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Ramp action — parity with LossScheduler
# ---------------------------------------------------------------------------


def test_ramp_interpolates_and_completes() -> None:
    ctrl = _ctrl(
        [
            {
                "name": "r",
                "target": "perceptual",
                "trigger": {"type": "iteration", "at": 1000},
                "action": {"type": "ramp", "to": 0.1, "over": 1000, "shape": "linear"},
            },
        ],
        loss_config=_losscfg(lambda_perceptual=0.0),
    )
    ctrl.on_iteration(1000, 0)  # start: from 0.0
    assert ctrl.current_overrides()["perceptual"] == pytest.approx(0.0)
    ctrl.on_iteration(1500, 0)  # halfway
    assert ctrl.current_overrides()["perceptual"] == pytest.approx(0.05, abs=1e-6)
    ctrl.on_iteration(2000, 0)  # complete
    assert ctrl.current_overrides()["perceptual"] == pytest.approx(0.1)
    ctrl.on_iteration(2500, 0)  # stays at target after completion
    assert ctrl.current_overrides()["perceptual"] == pytest.approx(0.1)


def test_ramp_matches_loss_scheduler_math() -> None:
    ctrl = _ctrl(
        [
            {
                "name": "r",
                "target": "hfen",  # a real schema-backed term: the SSOT rejects an undeclared loss
                "trigger": {"type": "iteration", "at": 100},
                "action": {"type": "ramp", "to": 1.0, "over": 400, "shape": "linear"},
            },
        ],
        loss_config=_losscfg(lambda_hfen=0.0),
    )
    ctrl.on_iteration(100, 0)
    ctrl.on_iteration(300, 0)
    expected = LossScheduler.compute_schedule(
        {
            "type": "linear_warmup",
            "start_step": 100,
            "warmup_steps": 400,
            "initial_value": 0.0,
            "final_value": 1.0,
        },
        base_value=1.0,
        step=300,
    )
    assert ctrl.current_overrides()["hfen"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Plateau / threshold triggers
# ---------------------------------------------------------------------------


def test_metric_plateau_fires_scale() -> None:
    ctrl = _ctrl(
        [
            {
                "name": "r",
                "target": "l1",
                "trigger": {
                    "type": "metric_plateau",
                    "metric": "val_ssim",
                    "mode": "max",
                    "patience": 2,
                    "cooldown": 5,
                },  # cooldown required for plateau+scale (L1)
                "action": {"type": "scale", "factor": 0.5},
            },
        ],
        loss_config=_losscfg(lambda_l1=1.0),
    )
    ctrl.on_validation(100, 0, {"val_ssim": 0.5})  # improvement (beats -inf)
    ctrl.on_validation(200, 0, {"val_ssim": 0.5})  # wait=1
    assert "l1" not in ctrl.current_overrides()
    ctrl.on_validation(300, 0, {"val_ssim": 0.5})  # wait=2 -> FIRE
    assert ctrl.current_overrides()["l1"] == pytest.approx(0.5)


def test_loss_plateau_fires_on_training_loss() -> None:
    ctrl = _ctrl(
        [
            {
                "name": "r",
                "target": "l1",
                "trigger": {
                    "type": "loss_plateau",
                    "loss_key": "g_total_loss",
                    "patience": 1,
                },
                "action": {"type": "disable"},
            },
        ]
    )
    ctrl.on_validation(100, 0, {}, train_losses={"g_total_loss": 0.2})  # improvement
    ctrl.on_validation(200, 0, {}, train_losses={"g_total_loss": 0.2})  # wait=1 -> FIRE
    assert ctrl.current_overrides()["l1"] == 0.0


def test_metric_threshold_fires_once_on_crossing() -> None:
    ctrl = _ctrl(
        [
            {
                "name": "r",
                "target": "perceptual",
                "trigger": {
                    "type": "metric_threshold",
                    "metric": "val_psnr",
                    "value": 30.0,
                    "mode": "max",
                },
                "action": {"type": "enable", "to": 0.2},
            },
        ]
    )
    ctrl.on_validation(100, 0, {"val_psnr": 28.0})  # below threshold
    assert "perceptual" not in ctrl.current_overrides()
    ctrl.on_validation(200, 0, {"val_psnr": 31.0})  # crosses -> FIRE
    assert ctrl.current_overrides()["perceptual"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# monitor_after — rollback / hold
# ---------------------------------------------------------------------------


def _plateau_with_monitor(on_failure: str):
    return [
        {
            "name": "r",
            "target": "l1",
            "trigger": {
                "type": "metric_plateau",
                "metric": "val_ssim",
                "mode": "max",
                "patience": 1,
            },
            "action": {"type": "scale", "factor": 0.5},
            "monitor_after": {
                "metric": "val_ssim",
                "mode": "max",
                "window": 2,
                "expect": "improve",
                "on_failure": on_failure,
            },
        }
    ]


def test_monitor_after_rollback_restores_when_metric_does_not_improve() -> None:
    ctrl = _ctrl(_plateau_with_monitor("rollback"), loss_config=_losscfg(lambda_l1=1.0))
    ctrl.on_validation(100, 0, {"val_ssim": 0.5})  # improvement, no fire
    ctrl.on_validation(
        200, 0, {"val_ssim": 0.5}
    )  # wait=1 -> FIRE scale 0.5; window opens, baseline 0.5
    assert ctrl.current_overrides()["l1"] == pytest.approx(0.5)
    ctrl.on_validation(300, 0, {"val_ssim": 0.5})  # window check 1 (flat, no improve)
    ctrl.on_validation(400, 0, {"val_ssim": 0.5})  # window check 2 -> FAIL -> rollback
    assert "l1" not in ctrl.current_overrides()  # pre-change had no override -> removed
    assert any(
        e.get("monitor_after", "").startswith("failed:rollback") for e in ctrl.events
    )


def test_monitor_after_success_keeps_change() -> None:
    ctrl = _ctrl(_plateau_with_monitor("rollback"), loss_config=_losscfg(lambda_l1=1.0))
    ctrl.on_validation(100, 0, {"val_ssim": 0.5})
    ctrl.on_validation(200, 0, {"val_ssim": 0.5})  # FIRE, baseline 0.5
    ctrl.on_validation(300, 0, {"val_ssim": 0.55})  # improved
    ctrl.on_validation(400, 0, {"val_ssim": 0.6})  # improved -> success
    assert ctrl.current_overrides()["l1"] == pytest.approx(0.5)  # change kept
    assert any(e.get("monitor_after") == "ok" for e in ctrl.events)


def test_monitor_after_hold_keeps_change_on_failure() -> None:
    ctrl = _ctrl(_plateau_with_monitor("hold"), loss_config=_losscfg(lambda_l1=1.0))
    ctrl.on_validation(100, 0, {"val_ssim": 0.5})
    ctrl.on_validation(200, 0, {"val_ssim": 0.5})  # FIRE
    ctrl.on_validation(300, 0, {"val_ssim": 0.5})
    ctrl.on_validation(400, 0, {"val_ssim": 0.5})  # FAIL but hold
    assert ctrl.current_overrides()["l1"] == pytest.approx(0.5)  # held, not rolled back
    assert any(
        e.get("monitor_after", "").startswith("failed:hold") for e in ctrl.events
    )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_events_record_old_and_new_weight() -> None:
    ctrl = _ctrl(
        [
            {
                "name": "r",
                "target": "l1",
                "trigger": {"type": "iteration", "at": 10},
                "action": {"type": "set_weight", "to": 2.0},
            },
        ],
        loss_config=_losscfg(lambda_l1=1.0),
    )
    ctrl.on_iteration(10, 0)
    ev = next(e for e in ctrl.events if e.get("action") == "set_weight")
    assert ev["old_weight"] == pytest.approx(1.0)
    assert ev["new_weight"] == pytest.approx(2.0)
    assert ev["target"] == "l1"
    assert ev["trigger"] == "iteration=10"


# ---------------------------------------------------------------------------
# Checkpoint-on-change: save before a triggered change, throttled by a
# user-configured interval (never hardcoded).
# ---------------------------------------------------------------------------


def _ckpt_ctrl(interval: int | None, on_change: bool = True, every: int = 50):
    cfg = LossScheduleConfigSchema(
        enabled=True,
        checkpoint_on_change=on_change,
        checkpoint_min_interval=interval,
        rules=[
            {
                "name": "r",
                "target": "l1",
                "trigger": {"type": "iteration", "every": every},
                "action": {"type": "disable"},
            }
        ],
    )
    return LossScheduleController(cfg)


def test_no_checkpoint_request_without_a_change() -> None:
    ctrl = _ckpt_ctrl(interval=100)
    assert ctrl.consume_checkpoint_request(5) is False  # nothing fired yet


def test_disabled_checkpoint_on_change_never_requests() -> None:
    ctrl = _ckpt_ctrl(interval=None, on_change=False)
    ctrl.on_iteration(50, 0)  # rule fires (change applied)
    assert ctrl.consume_checkpoint_request(50) is False  # feature off -> no request


def test_first_change_requests_checkpoint() -> None:
    ctrl = _ckpt_ctrl(interval=100)
    ctrl.on_iteration(50, 0)  # fire -> pending
    assert ctrl.consume_checkpoint_request(50) is True  # first ever -> save


def test_consume_resets_pending() -> None:
    ctrl = _ckpt_ctrl(interval=100)
    ctrl.on_iteration(50, 0)
    assert ctrl.consume_checkpoint_request(50) is True
    assert ctrl.consume_checkpoint_request(50) is False  # pending cleared


def test_checkpoint_throttled_by_user_interval() -> None:
    """Frequent triggers coalesce: at most one save per checkpoint_min_interval."""
    ctrl = _ckpt_ctrl(interval=100, every=50)
    ctrl.on_iteration(50, 0)
    assert ctrl.consume_checkpoint_request(50) is True  # save, last=50
    ctrl.on_iteration(100, 0)  # fires again -> pending
    assert ctrl.consume_checkpoint_request(100) is False  # 100-50=50 < 100 -> throttled
    ctrl.on_iteration(150, 0)  # fires again -> still pending
    assert ctrl.consume_checkpoint_request(150) is True  # 150-50=100 >= 100 -> save


def test_interval_is_config_driven_not_hardcoded() -> None:
    """A small vs large interval changes throttling — proving the gate reads config."""
    fast = _ckpt_ctrl(interval=10, every=50)
    fast.on_iteration(50, 0)
    assert fast.consume_checkpoint_request(50) is True
    fast.on_iteration(100, 0)
    assert fast.consume_checkpoint_request(100) is True  # 100-50=50 >= 10 -> save

    slow = _ckpt_ctrl(interval=1000, every=50)
    slow.on_iteration(50, 0)
    assert slow.consume_checkpoint_request(50) is True
    slow.on_iteration(100, 0)
    assert slow.consume_checkpoint_request(100) is False  # 50 < 1000 -> throttled


def test_plateau_change_requests_checkpoint() -> None:
    cfg = LossScheduleConfigSchema(
        enabled=True,
        checkpoint_on_change=True,
        checkpoint_min_interval=100,
        rules=[
            {
                "name": "r",
                "target": "l1",
                "trigger": {
                    "type": "metric_plateau",
                    "metric": "val_ssim",
                    "mode": "max",
                    "patience": 1,
                },
                "action": {"type": "disable"},
            }
        ],
    )
    ctrl = LossScheduleController(cfg)
    ctrl.on_validation(100, 0, {"val_ssim": 0.5})  # improvement, no fire
    ctrl.on_validation(200, 0, {"val_ssim": 0.5})  # wait=1 -> FIRE -> pending
    assert ctrl.consume_checkpoint_request(200) is True


def test_throttled_change_is_dropped_not_deferred() -> None:
    """A change fired inside the throttle window is DROPPED, not held for a later
    save at an unrelated iteration capturing post-change weights (M5)."""
    cfg = LossScheduleConfigSchema(
        enabled=True,
        checkpoint_on_change=True,
        checkpoint_min_interval=100,
        rules=[
            {
                "name": "a",
                "target": "l1",
                "trigger": {"type": "iteration", "at": 50},
                "action": {"type": "disable"},
            },
            {
                "name": "b",
                "target": "l2",
                "trigger": {"type": "iteration", "at": 60},
                "action": {"type": "disable"},
            },
        ],
    )
    ctrl = LossScheduleController(cfg)
    ctrl.on_iteration(50, 0)  # a fires -> pending
    assert ctrl.consume_checkpoint_request(50) is True  # first save, last=50
    ctrl.on_iteration(60, 0)  # b fires -> pending
    assert ctrl.consume_checkpoint_request(60) is False  # 60-50=10<100 -> DROPPED
    # No new change fires; the dropped request must not resurface later.
    assert ctrl.consume_checkpoint_request(170) is False  # would be True under defer


# ---------------------------------------------------------------------------
# C2 — metric-key alias resolution (val_ prefix) + missing-key diagnostics
# ---------------------------------------------------------------------------


def test_metric_plateau_resolves_val_prefix_alias() -> None:
    """Docs use ``val_ssim``; the validator emits bare ``ssim`` -- the trigger
    must still fire via the shared early-stopping alias resolver (C2)."""
    ctrl = _ctrl(
        [
            {
                "name": "r",
                "target": "l1",
                "trigger": {
                    "type": "metric_plateau",
                    "metric": "val_ssim",
                    "mode": "max",
                    "patience": 1,
                    "cooldown": 5,
                },
                "action": {"type": "scale", "factor": 0.5},
            },
        ],
        loss_config=_losscfg(lambda_l1=1.0),
    )
    ctrl.on_validation(100, 0, {"ssim": 0.5})  # bare key, improvement
    ctrl.on_validation(200, 0, {"ssim": 0.5})  # wait=1 -> FIRE via alias
    assert ctrl.current_overrides()["l1"] == pytest.approx(0.5)


def test_unresolvable_metric_never_fires_and_warns_once(caplog) -> None:
    import logging

    ctrl = _ctrl(
        [
            {
                "name": "r",
                "target": "l1",
                "trigger": {
                    "type": "metric_plateau",
                    "metric": "val_made_up",
                    "mode": "max",
                    "patience": 1,
                    "cooldown": 5,
                },
                "action": {"type": "scale", "factor": 0.5},
            },
        ],
        loss_config=_losscfg(lambda_l1=1.0),
    )
    with caplog.at_level(logging.WARNING):
        ctrl.on_validation(100, 0, {"ssim": 0.5})
        ctrl.on_validation(200, 0, {"ssim": 0.5})
    assert "l1" not in ctrl.current_overrides()  # never fired (silent no-op avoided)
    assert sum("not found" in r.getMessage() for r in caplog.records) == 1  # warn once


# ---------------------------------------------------------------------------
# M1 / M2 — monitor_after baseline metric + non-metric trigger coverage
# ---------------------------------------------------------------------------


def test_monitor_after_uses_monitor_metric_baseline_not_trigger() -> None:
    """Trigger on ssim, monitor psnr: the baseline must be psnr@fire, so a
    falling psnr rolls back. Pre-fix it compared psnr against an ssim baseline
    and never rolled back (M1)."""
    rules = [
        {
            "name": "r",
            "target": "l1",
            "trigger": {
                "type": "metric_plateau",
                "metric": "val_ssim",
                "mode": "max",
                "patience": 1,
                "cooldown": 5,
            },
            "action": {"type": "scale", "factor": 0.5},
            "monitor_after": {
                "metric": "val_psnr",
                "mode": "max",
                "window": 2,
                "expect": "improve",
                "on_failure": "rollback",
            },
        }
    ]
    ctrl = _ctrl(rules, loss_config=_losscfg(lambda_l1=1.0))
    ctrl.on_validation(
        100, 0, {"val_ssim": 0.5, "val_psnr": 30.0}
    )  # improvement, no fire
    ctrl.on_validation(
        200, 0, {"val_ssim": 0.5, "val_psnr": 30.0}
    )  # FIRE; baseline psnr=30
    assert ctrl.current_overrides()["l1"] == pytest.approx(0.5)
    ctrl.on_validation(300, 0, {"val_ssim": 0.5, "val_psnr": 29.0})  # psnr falls
    ctrl.on_validation(
        400, 0, {"val_ssim": 0.5, "val_psnr": 28.0}
    )  # psnr falls -> FAIL -> rollback
    assert "l1" not in ctrl.current_overrides()


def test_monitor_after_works_for_loss_plateau() -> None:
    """monitor_after watching a val metric must engage for a loss_plateau trigger
    (baseline captured from the val metric at fire time) (M2)."""
    rules = [
        {
            "name": "r",
            "target": "l1",
            "trigger": {
                "type": "loss_plateau",
                "loss_key": "g_total_loss",
                "patience": 1,
            },
            "action": {"type": "disable"},
            "monitor_after": {
                "metric": "val_ssim",
                "mode": "max",
                "window": 2,
                "expect": "improve",
                "on_failure": "rollback",
            },
        }
    ]
    ctrl = _ctrl(rules, loss_config=_losscfg(lambda_l1=1.0))
    ctrl.on_validation(100, 0, {"val_ssim": 0.5}, train_losses={"g_total_loss": 0.2})
    ctrl.on_validation(
        200, 0, {"val_ssim": 0.5}, train_losses={"g_total_loss": 0.2}
    )  # FIRE disable
    assert ctrl.current_overrides()["l1"] == 0.0
    ctrl.on_validation(300, 0, {"val_ssim": 0.5}, train_losses={"g_total_loss": 0.2})
    ctrl.on_validation(
        400, 0, {"val_ssim": 0.5}, train_losses={"g_total_loss": 0.2}
    )  # FAIL -> rollback
    assert "l1" not in ctrl.current_overrides()


def test_monitor_after_works_for_clock_trigger_deferred_baseline() -> None:
    """A clock trigger has no metric at fire time, so the baseline is captured at
    the first subsequent validation, then the window runs (M2)."""
    rules = [
        {
            "name": "r",
            "target": "perceptual",
            "trigger": {"type": "iteration", "at": 1000},
            "action": {"type": "set_weight", "to": 0.5},
            "monitor_after": {
                "metric": "val_psnr",
                "mode": "max",
                "window": 2,
                "expect": "improve",
                "on_failure": "rollback",
            },
        }
    ]
    ctrl = _ctrl(rules, loss_config=_losscfg(lambda_perceptual=0.1))
    ctrl.on_iteration(1000, 0)  # fire; window deferred (no val_metrics yet)
    assert ctrl.current_overrides()["perceptual"] == pytest.approx(0.5)
    ctrl.on_validation(1100, 0, {"val_psnr": 30.0})  # arms baseline=30 (not a check)
    ctrl.on_validation(1200, 0, {"val_psnr": 29.0})  # check 1 (worse)
    ctrl.on_validation(1300, 0, {"val_psnr": 28.0})  # check 2 -> FAIL -> rollback
    assert "perceptual" not in ctrl.current_overrides()


def test_post_change_times_out_when_monitor_metric_missing() -> None:
    """A monitor metric that never appears must time out (and apply on_failure)
    instead of wedging the rule in _post forever (M6)."""
    rules = [
        {
            "name": "r",
            "target": "l1",
            "trigger": {
                "type": "metric_plateau",
                "metric": "val_ssim",
                "mode": "max",
                "patience": 1,
                "cooldown": 5,
            },
            "action": {"type": "scale", "factor": 0.5},
            "monitor_after": {
                "metric": "val_typo",
                "mode": "max",
                "window": 2,
                "expect": "improve",
                "on_failure": "rollback",
            },
        }
    ]
    ctrl = _ctrl(rules, loss_config=_losscfg(lambda_l1=1.0))
    ctrl.on_validation(100, 0, {"val_ssim": 0.5})
    ctrl.on_validation(
        200, 0, {"val_ssim": 0.5}
    )  # FIRE; val_typo absent -> deferred unarmed
    assert ctrl.current_overrides()["l1"] == pytest.approx(0.5)
    ctrl.on_validation(300, 0, {"val_ssim": 0.5})  # miss 1
    ctrl.on_validation(
        400, 0, {"val_ssim": 0.5}
    )  # miss 2 >= window -> timeout -> rollback
    assert "l1" not in ctrl.current_overrides()
    assert any("metric_missing" in str(e.get("monitor_after", "")) for e in ctrl.events)


# ---------------------------------------------------------------------------
# L2 — an 'every' ramp completes instead of perpetually restarting
# ---------------------------------------------------------------------------


def test_every_ramp_completes_before_reramping() -> None:
    ctrl = _ctrl(
        [
            {
                "name": "r",
                "target": "hfen",  # a real schema-backed term: the SSOT rejects an undeclared loss
                "trigger": {"type": "iteration", "every": 5},
                "action": {"type": "ramp", "to": 1.0, "over": 10},
            },
        ],
        loss_config=_losscfg(lambda_hfen=0.0),
    )
    ctrl.on_iteration(5, 0)  # ramp starts (from 0.0)
    ctrl.on_iteration(
        10, 0
    )  # mid-ramp; rule skipped while ramp active -> not restarted
    ctrl.on_iteration(15, 0)  # ramp reaches its target
    assert ctrl.current_overrides()["hfen"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# M4 — resume seeding (clock curricula reconstructed, not replayed)
# ---------------------------------------------------------------------------


def test_resume_seeds_completed_ramp_without_replay() -> None:
    rules = [
        {
            "name": "r",
            "target": "perceptual",
            "trigger": {"type": "iteration", "at": 100},
            "action": {"type": "ramp", "to": 1.0, "over": 50},
        }
    ]
    cfg = LossScheduleConfigSchema(enabled=True, rules=rules)
    ctrl = LossScheduleController(
        cfg, loss_config=_losscfg(lambda_perceptual=0.0), start_iteration=300
    )
    assert ctrl.current_overrides()["perceptual"] == pytest.approx(
        1.0
    )  # settled, not replayed
    out = ctrl.on_iteration(301, 0)  # must NOT re-fire / reset to base
    assert out["perceptual"] == pytest.approx(1.0)


def test_resume_reconstructs_midflight_ramp_in_phase() -> None:
    rules = [
        {
            "name": "r",
            "target": "hfen",  # a real schema-backed term: the SSOT rejects an undeclared loss
            "trigger": {"type": "iteration", "at": 100},
            "action": {"type": "ramp", "to": 1.0, "over": 400},
        }
    ]
    cfg = LossScheduleConfigSchema(enabled=True, rules=rules)
    ctrl = LossScheduleController(
        cfg, loss_config=_losscfg(lambda_hfen=0.0), start_iteration=300
    )
    expected = LossScheduler.compute_schedule(
        {
            "type": "linear_warmup",
            "start_step": 100,
            "warmup_steps": 400,
            "initial_value": 0.0,
            "final_value": 1.0,
        },
        base_value=1.0,
        step=300,
    )
    assert ctrl.current_overrides()["hfen"] == pytest.approx(
        expected
    )  # in-phase, not from 300


def test_resume_seeds_set_weight_and_does_not_refire() -> None:
    rules = [
        {
            "name": "r",
            "target": "l1",
            "trigger": {"type": "iteration", "at": 100},
            "action": {"type": "set_weight", "to": 2.0},
        }
    ]
    cfg = LossScheduleConfigSchema(enabled=True, rules=rules)
    ctrl = LossScheduleController(
        cfg, loss_config=_losscfg(lambda_l1=1.0), start_iteration=500
    )
    assert ctrl.current_overrides()["l1"] == pytest.approx(2.0)
    n_before = len(ctrl.events)
    ctrl.on_iteration(501, 0)  # already fired at seed -> no new event
    assert len(ctrl.events) == n_before
