"""Pin the *wiring* of the workflow maturity gate, not just its rule.

``enforce_pipeline_maturity_for_config`` is exhaustively tested as a domain
function elsewhere (``tests/unit/domain/workflows/``). Every one of those tests
calls it directly, so deleting the call from ``pipelines/train.py`` left the
whole suite green: the rule was tested, its wiring was not. That is pitfall #16
one layer up — a mechanism that is advertised at the pipeline entry but never
proven to fire there.

Two independent layers pin it here, because either alone is weak:

* **Source pins** (``inspect.getsource`` + substring / index) — cheap, and they
  pin *ordering*, which no single behavioural call can observe directly. But a
  substring assert can pass vacuously if the line is dead code.
* **Behavioural pins** — call the real ``run_training_pipeline`` /
  ``run_inference_pipeline`` and assert ``WorkflowNotImplementedError``. These
  are only meaningful because the gate sits before any container build, data
  load, or ``.to(device)``: a STUB config raises with no dataset, no
  checkpoint, and no CUDA. They cannot pass vacuously.

The ordering test is behavioural too: ``validate_config_health`` and
``bootstrap.build_container`` are monkeypatched to raise sentinels, so a STUB
config that still raises ``WorkflowNotImplementedError`` proves the gate ran
*first*. ``test_health_check_sentinel_is_reached_when_the_gate_allows`` is the
control for it — it proves the sentinel patch is live, so that the ordering
test genuinely goes red if the gate is removed rather than silently passing.

Scope note: ``Maturity.EVAL_ONLY`` is NOT exercised through a pipeline here.
No regime currently declares it (every MR regime is LIVE, the rest are STUB),
so there is no config that would reach that branch. The branch is covered at
the domain level; inventing a fake profile to reach it through the pipeline
would test the fake, not the wiring.

CPU-only by construction: the gate raises before any device work, and the
inference test passes ``device="cpu"`` (an explicit opt-in the accelerated-run
contract permits).
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest
import yaml

import mriforge.pipelines.infer as infer_mod
import mriforge.pipelines.train as train_mod
from mriforge.config.settings import TrainingSettings
from mriforge.domain.exceptions import WorkflowNotImplementedError

# A STUB regime: the framework has no forward operator/losses/strategy for it,
# so every pipeline must refuse it. See WORKFLOW_PROFILES.
_STUB_REGIME = "ultrasound"
# The only regime honestly runnable on this cluster's data today.
_LIVE_REGIME = "mri_structural"


def _config_dict(regime: str) -> dict[str, Any]:
    """Smallest config the schema accepts, declaring ``regime``."""
    return {
        "config_version": "1.0",
        "model": {"model_type": "unet"},
        "data": {"dataset_type": "synthetic", "data_path": "/nonexistent"},
        "optimization": {"learning_rate": 1e-4},
        "logging": {"run_name": "maturity_gate_wiring"},
        "workflow": {"regime": regime, "task": "reconstruction"},
    }


def _settings(regime: str) -> TrainingSettings:
    return TrainingSettings.settings_from_dict(_config_dict(regime))


class _SentinelError(Exception):
    """Raised by a monkeypatched step to mark that the step was reached."""


def _code(fn: Any) -> str:
    """Source of ``fn`` with comment-only lines removed.

    A bare ``inspect.getsource`` substring assert cannot tell a live call from
    a commented-out one: commenting the gate out leaves the call text on the
    line, so the assert still passes. This was observed, not theorised — the
    mutation check for this file caught it. Dropping comment lines makes the
    source pins fail on that mutation. Inline trailing comments are not
    stripped (quote-aware parsing is not worth it here); the behavioural tests
    below are the real backstop.
    """
    src = inspect.getsource(fn)
    return "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    )


# ---------------------------------------------------------------------------
# Source pins — the callsites exist, with the right pipeline kind
# ---------------------------------------------------------------------------


def test_train_pipeline_calls_the_gate_with_train() -> None:
    src = _code(train_mod.run_training_pipeline)
    assert 'enforce_pipeline_maturity_for_config(config, "train")' in src


def test_infer_pipeline_calls_the_gate_with_predict() -> None:
    src = _code(infer_mod.run_inference_pipeline)
    assert 'enforce_pipeline_maturity_for_config(config, "predict")' in src


def test_gate_precedes_health_check_and_container_build_in_source() -> None:
    """Ordering is the whole point: a STUB must raise before resources.

    Asserted by source index because a single call cannot observe the order of
    two steps when the first one always raises. The behavioural counterpart is
    ``test_gate_fires_before_health_check_and_container_build``.
    """
    src = _code(train_mod.run_training_pipeline)
    gate = src.index("enforce_pipeline_maturity_for_config(config,")
    health = src.index("validate_config_health(config)")
    container = src.index("bootstrap.build_container(")

    assert gate < health, "maturity gate must precede validate_config_health"
    assert gate < container, "maturity gate must precede the container build"


# ---------------------------------------------------------------------------
# Behavioural pins — the gate actually fires through the real pipeline
# ---------------------------------------------------------------------------


def test_stub_regime_raises_from_run_training_pipeline() -> None:
    """The real pipeline refuses a STUB regime, with no data and no GPU."""
    with pytest.raises(WorkflowNotImplementedError) as exc:
        train_mod.run_training_pipeline(_settings(_STUB_REGIME), device="cpu")

    msg = str(exc.value)
    assert _STUB_REGIME in msg
    assert "STUB" in msg
    # The message embeds the pipeline kind, so this pins the *argument*
    # train.py passes, not merely that some call happened.
    assert "'train'" in msg


def test_stub_regime_raises_from_run_inference_pipeline(tmp_path: Path) -> None:
    """Inference refuses a STUB regime before touching the checkpoint.

    The checkpoint/input paths below do not exist: reaching them would be a
    different error, so the expected raise also pins that the gate runs early.
    """
    cfg = tmp_path / "stub_arm.yaml"
    cfg.write_text(yaml.safe_dump(_config_dict(_STUB_REGIME)))

    with pytest.raises(WorkflowNotImplementedError) as exc:
        infer_mod.run_inference_pipeline(
            config_path=cfg,
            checkpoint_path=tmp_path / "missing.pt",
            input_path=tmp_path / "missing_input",
            output_path=tmp_path / "out",
            device="cpu",
        )

    msg = str(exc.value)
    assert _STUB_REGIME in msg
    assert "'predict'" in msg


def test_gate_fires_before_health_check_and_container_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Behavioural ordering: neither later step is reached for a STUB regime.

    Both steps are booby-trapped. If the gate were moved after either one, the
    sentinel would surface instead of WorkflowNotImplementedError.
    """

    def _health_boom(config: Any) -> Any:
        raise _SentinelError("validate_config_health was reached")

    def _container_boom(*args: Any, **kwargs: Any) -> Any:
        raise _SentinelError("bootstrap.build_container was reached")

    monkeypatch.setattr(train_mod, "validate_config_health", _health_boom)
    monkeypatch.setattr(train_mod.bootstrap, "build_container", _container_boom)

    with pytest.raises(WorkflowNotImplementedError):
        train_mod.run_training_pipeline(_settings(_STUB_REGIME), device="cpu")


def test_health_check_sentinel_is_reached_when_the_gate_allows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control for the ordering test above — proves the trap is live.

    Without this, ``test_gate_fires_before_health_check_and_container_build``
    could pass for the wrong reason (a patch target that never fires would make
    it unfalsifiable). A LIVE regime passes the gate and must hit the sentinel,
    which shows the patched step is genuinely on the path the STUB config would
    have taken had the gate not stopped it first.
    """

    def _health_boom(config: Any) -> Any:
        raise _SentinelError("validate_config_health was reached")

    monkeypatch.setattr(train_mod, "validate_config_health", _health_boom)

    with pytest.raises(_SentinelError):
        train_mod.run_training_pipeline(_settings(_LIVE_REGIME), device="cpu")
