"""``_clip_and_log_gradients`` must not pay a host sync it cannot use.

``_compute_gradient_norm`` used to end in ``return float(total_norm)``. On CUDA
that blocks until the queue drains, it runs once per optimiser step, and a
Scalene profile of ``experiment_11_attention_none`` charged it 4.64 % of the run
(~43 s over 150 steps) -- for a value **no production caller reads**:
``amp_policy.backward_and_step`` and ``step_executor`` both invoke the clip hook
as a bare statement.

The norm is now returned as a device tensor and materialised only when a warning
could actually be emitted. The ``not enable_clip`` case deliberately keeps its
per-step copy: with clipping off there is no protection, the explosion warning
is meant to fire every step, and the norm exists for that warning alone.

Same shape as the deferral in ``TestModelOutputScaleContextIsDeferred``
(``test_base.py``), which pins #1188.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

CLIP_VALUE = 1.0


class _Strat(BaseTrainingStrategy):
    """Concrete subclass so ``object.__new__`` is not blocked by the ABC."""

    def _compute_losses_impl(self, *args, **kwargs):  # pragma: no cover - stub
        return {}


class _FloatCountingTensor:
    """Counts ``float(...)`` -- the device-to-host copy under test.

    ``__float__`` is looked up on the type, so it is defined explicitly rather
    than left to ``__getattr__``.
    """

    def __init__(self, value: float) -> None:
        self._real = torch.tensor(value)
        self.float_calls = 0

    def __float__(self) -> float:
        self.float_calls += 1
        return float(self._real)

    def __getattr__(self, name: str):
        return getattr(self._real, name)


class _RecordingLogger:
    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.infos: list[str] = []

    def log_warning(self, message: str, **_: object) -> None:
        self.warnings.append(message)

    def log_info(self, message: str, **_: object) -> None:
        self.infos.append(message)


def _strategy(*, clip_enabled: bool, log_gradients: bool = False, log_interval: int = 30) -> _Strat:
    """A bare strategy whose real config helpers do the reading.

    The nested config is built out rather than stubbing
    ``_get_gradient_clipping_config`` / ``_get_gradient_logging_config``, so the
    branch under test is reached the way production reaches it.
    """
    strategy = object.__new__(_Strat)
    strategy.config = SimpleNamespace(
        optimization=SimpleNamespace(
            gradient=SimpleNamespace(clip=SimpleNamespace(value=CLIP_VALUE, enabled=clip_enabled))
        ),
        logging=SimpleNamespace(
            log_gradients=log_gradients, intervals=SimpleNamespace(log=log_interval)
        ),
    )
    strategy.env = SimpleNamespace(model_type="generator")
    strategy.logging_service = _RecordingLogger()
    return strategy


def _model_with_grads(scale: float = 3.0) -> torch.nn.Module:
    model = torch.nn.Linear(4, 4, bias=False)
    model.weight.grad = torch.full_like(model.weight, scale)
    return model


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch) -> _FloatCountingTensor:
    """Replace the norm with a tensor that reports every host copy."""
    counting = _FloatCountingTensor(2.0)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *a, **k: counting)
    return counting


class TestTheSyncIsDeferred:
    def test_ordinary_clipping_step_pays_no_host_copy(self, probe) -> None:
        """step=1: clipping on, nothing to log, no anomaly check -- 0 syncs."""
        strategy = _strategy(clip_enabled=True)
        strategy._clip_and_log_gradients(_model_with_grads(), epoch=0, step=1)
        assert probe.float_calls == 0

    def test_a_long_run_of_steps_pays_nothing(self, probe) -> None:
        """The saving is per-step; one leaked sync per step is the whole defect."""
        strategy = _strategy(clip_enabled=True)
        model = _model_with_grads()
        for step in range(1, 100):
            strategy._clip_and_log_gradients(model, epoch=0, step=step)
        assert probe.float_calls == 0

    def test_return_is_the_documented_not_measured_sentinel(self, probe) -> None:
        """The deferred step reports *no number*, not the number zero.

        This used to assert ``== 0.0``. That sentinel was indistinguishable from
        a genuinely vanished gradient -- a real, reportable measurement -- so the
        ``-> float`` contract claimed every return value meant something. The
        sentinel is now ``None``; ``0.0`` is reserved for an actual zero norm.
        """
        strategy = _strategy(clip_enabled=True)
        result = strategy._clip_and_log_gradients(_model_with_grads(), epoch=0, step=1)
        assert result is None


class TestTheSyncStillHappensWhenItIsOwed:
    """Deferring must not cost the diagnostic (non-negotiable 3)."""

    def test_anomaly_step_materialises_once(self, probe) -> None:
        strategy = _strategy(clip_enabled=True)
        strategy._clip_and_log_gradients(_model_with_grads(), epoch=0, step=100)
        assert probe.float_calls == 1

    def test_logging_step_materialises_once(self, probe) -> None:
        strategy = _strategy(clip_enabled=True, log_gradients=True, log_interval=30)
        strategy._clip_and_log_gradients(_model_with_grads(), epoch=0, step=30)
        assert probe.float_calls == 1

    def test_clipping_disabled_computes_no_norm_on_an_idle_step(self, probe) -> None:
        """Pre-existing early return: with nothing to clip or log, nothing runs.

        This is the branch that made a ``not enable_clip`` clause in the
        materialise condition dead logic -- such a step returns before the norm
        is ever computed, so the sync it would have guarded cannot occur.
        """
        strategy = _strategy(clip_enabled=False)
        for step in (1, 2, 3):
            assert strategy._clip_and_log_gradients(_model_with_grads(), epoch=0, step=step) is None
        assert probe.float_calls == 0

    def test_clipping_disabled_still_materialises_on_an_anomaly_step(self, probe) -> None:
        """With no clipping there is no protection, so the warning must be live."""
        strategy = _strategy(clip_enabled=False)
        strategy._clip_and_log_gradients(_model_with_grads(), epoch=0, step=100)
        assert probe.float_calls == 1

    def test_explosion_warning_survives_at_the_anomaly_cadence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *a, **k: torch.tensor(5000.0))
        strategy = _strategy(clip_enabled=True)
        strategy._clip_and_log_gradients(_model_with_grads(), epoch=0, step=100)
        assert any("GRADIENT EXPLOSION" in w for w in strategy.logging_service.warnings)


class TestClippingItselfIsUnchanged:
    """The norm is deferred; the clip is not."""

    def test_compute_gradient_norm_returns_a_tensor(self) -> None:
        strategy = _strategy(clip_enabled=True)
        norm = strategy._compute_gradient_norm(_model_with_grads(), True, CLIP_VALUE)
        assert isinstance(norm, torch.Tensor)
        assert not isinstance(norm, float)

    def test_gradients_are_actually_clipped_without_materialising(self) -> None:
        strategy = _strategy(clip_enabled=True)
        model = _model_with_grads(scale=100.0)
        strategy._clip_and_log_gradients(model, epoch=0, step=1)
        assert float(model.weight.grad.norm()) <= CLIP_VALUE + 1e-5

    def test_norm_is_not_clipped_when_clipping_is_disabled(self) -> None:
        strategy = _strategy(clip_enabled=False)
        model = _model_with_grads(scale=100.0)
        strategy._clip_and_log_gradients(model, epoch=0, step=1)
        assert float(model.weight.grad.norm()) > CLIP_VALUE


class TestTheSentinelIsDistinguishableFromARealZero:
    """The reason ``0.0`` was wrong: it is also a legitimate answer.

    A vanished gradient is a finding a caller may want to log or alarm on. While
    the skip path returned ``0.0`` too, no consumer could tell "the norm was
    zero" from "no norm was taken", and the ``-> float`` annotation asserted the
    first reading. These tests pin both halves so a future revert to a numeric
    sentinel turns them red rather than merely changing a value.
    """

    def test_a_genuinely_zero_gradient_still_reports_the_number(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *a, **k: torch.tensor(0.0))
        strategy = _strategy(clip_enabled=True)
        # step=100 is an anomaly step, so the norm IS materialised.
        result = strategy._clip_and_log_gradients(_model_with_grads(), epoch=0, step=100)
        assert result == 0.0
        assert isinstance(result, float)

    def test_the_skipped_step_reports_no_number_at_all(self, probe) -> None:
        strategy = _strategy(clip_enabled=True)
        assert strategy._clip_and_log_gradients(_model_with_grads(), epoch=0, step=1) is None

    def test_the_two_are_not_equal(self, monkeypatch: pytest.MonkeyPatch, probe) -> None:
        """The discrimination the old sentinel destroyed, stated directly."""
        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *a, **k: torch.tensor(0.0))
        strategy = _strategy(clip_enabled=True)
        measured = strategy._clip_and_log_gradients(_model_with_grads(), epoch=0, step=100)
        skipped = strategy._clip_and_log_gradients(_model_with_grads(), epoch=0, step=1)
        assert measured is not None
        assert skipped is None

    def test_the_annotation_admits_the_sentinel(self) -> None:
        """``-> float`` was a lie once a non-measurement could be returned."""
        import inspect

        from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

        sig = inspect.signature(BaseTrainingStrategy._clip_and_log_gradients)
        assert sig.return_annotation in ("float | None", float | None)
