"""Regression tests for SM-001: ``AdversarialMixin``'s per-step closures must
NOT trigger a GPU->CPU sync.

Previously the ``d_closure`` / ``g_closure`` bodies stored
``float(tensor.detach())`` into ``_last_step_metrics``. ``float()`` on a CUDA
tensor calls ``__float__`` == ``.item()`` == a blocking D2H transfer, executed
on EVERY training step (NN#9 violation). The fix defers that conversion: the
closures store detached *tensors* on-device, and ``get_last_metrics()`` resolves
them to Python floats at the coarser reporting cadence.

These tests are CPU-only and mock the heavy collaborators (generator,
discriminator, loss computer) so the real closures run unchanged.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from mriforge.infrastructure.training.strategies.mixins import adversarial
from mriforge.infrastructure.training.strategies.mixins.adversarial import (
    AdversarialMixin,
)


class _LossOutput(SimpleNamespace):
    """Duck-typed stand-in for the loss-computer output: exposes ``.total`` and
    ``.components`` exactly like ``UnifiedGANLossComputer`` results."""


class _StubLossComputer:
    """Minimal loss computer matching the attributes the closures probe."""

    def compute_discriminator_loss(self, **_: Any) -> _LossOutput:
        return _LossOutput(
            total=torch.tensor(0.7, requires_grad=True),
            components={"adv": torch.tensor(0.3)},
        )

    def compute_generator_loss(self, **_: Any) -> _LossOutput:
        return _LossOutput(
            total=torch.tensor(0.5, requires_grad=True),
            components={"adv": torch.tensor(0.2)},
        )


class _Strat(AdversarialMixin):
    """Bare adversarial-mixin host wired with CPU-only stubs for everything the
    closures touch, so ``train_step_adversarial`` exercises the real closure
    bodies (and thus the SM-001 metric-storage path)."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self._step_counter = 0
        self.loss_computer = _StubLossComputer()
        gen = torch.nn.Identity()
        disc = torch.nn.Conv2d(1, 1, 1)
        self._gen = gen
        self._disc = disc
        # disc_updates lives at config.losses.gan.disc_updates (v6.0 SSOT)
        cfg = SimpleNamespace(
            losses=SimpleNamespace(gan=SimpleNamespace(disc_updates=1)),
            training=SimpleNamespace(enforce_output_range=False),
        )
        self.config = cfg
        self.env = SimpleNamespace(config=cfg, step=0, criterion_l1=None)
        self.state = SimpleNamespace(opt_d=None, opt_g=None, config=cfg)

    # --- stubbed collaborators the closures read -------------------------
    @property
    def generator_model(self) -> torch.nn.Module:
        return self._gen

    @property
    def discriminator_model(self) -> torch.nn.Module:
        return self._disc

    def _to_device(self, data: Any) -> Any:
        return data

    def _unpack_batch(self, batch: Any) -> tuple[Any, Any]:
        return batch["input"], batch["target"]


def _run_closures() -> _Strat:
    strat = _Strat()
    img = torch.randn(1, 1, 4, 4)
    batch = {"input": img, "target": img.clone()}
    step_configs = strat.train_step_adversarial(batch, epoch=0, iteration=5)
    # Execute every produced closure (d then g) exactly as the Trainer would.
    for cfg in step_configs:
        cfg["closure"]()
    return strat


def test_closures_store_tensors_not_floats() -> None:
    """SM-001 core: after the closures run, stored metrics are detached tensors
    (on-device), proving no per-step D2H float() conversion happened inside the
    closure body."""
    strat = _run_closures()
    metrics = strat._last_step_metrics
    assert metrics, "closures should have populated _last_step_metrics"
    # The loss totals are stored as tensors, NOT Python floats.
    assert isinstance(metrics["d_total_loss"], torch.Tensor)
    assert isinstance(metrics["g_total_loss"], torch.Tensor)
    # And they are detached (no grad history leaks into the metrics dict).
    assert not metrics["d_total_loss"].requires_grad
    assert not metrics["g_total_loss"].requires_grad
    # Component metrics likewise remain tensors.
    assert isinstance(metrics["d_adv"], torch.Tensor)


def test_get_last_metrics_stays_on_device() -> None:
    """The D2H sync is deferred PAST get_last_metrics, not to it (#707).

    This asserted "a dict of pure Python floats", which was the original SM-001
    design: the closures stop syncing per key, and `get_last_metrics` converts.
    The sync audit found the second half was still wrong -- `training_loop`
    calls this on EVERY iteration, outside the `log_interval` gate, so the sync
    count per step never dropped. It only moved. The loop's gated converter now
    owns the single fused transfer.

    The values are still the ones the closures stored, so what this test really
    guarded (the closures publish real metrics) is unchanged.
    """
    strat = _run_closures()
    resolved = strat.get_last_metrics()
    assert resolved, "get_last_metrics should expose the stored metrics"
    assert all(isinstance(v, torch.Tensor) for v in resolved.values())
    # fp32 round-off: the stub stores 0.7/0.5 which widen-then-narrow through float32.
    assert float(resolved["d_total_loss"]) == pytest.approx(0.7, abs=1e-6)
    assert float(resolved["g_total_loss"]) == pytest.approx(0.5, abs=1e-6)


def test_no_bare_float_detach_in_metric_storage() -> None:
    """AST guard: the closure metric-storage sites must not wrap a ``.detach()``
    in ``float(...)`` (the exact D2H-sync pattern SM-001 removed)."""
    source = Path(inspect.getsourcefile(adversarial)).read_text()
    tree = ast.parse(source)

    offenders: list[int] = []
    for node in ast.walk(tree):
        # Match float(<x>.detach())
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "float"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Call)
            and isinstance(node.args[0].func, ast.Attribute)
            and node.args[0].func.attr == "detach"
        ):
            offenders.append(node.lineno)

    assert not offenders, (
        "float(x.detach()) (a per-step GPU->CPU sync) reintroduced at "
        f"adversarial.py lines {offenders}; defer to get_last_metrics()."
    )


def test_train_step_reads_loop_state_not_frozen_env_step() -> None:
    """WS-3 follow-up: ``train_step_adversarial``'s ``current_step`` (fed to the
    ``train_metric_interval`` throttle) reads the live ``loop_state`` seam via
    ``resolve_loop_iteration(self)``, not the frozen ``self.env.step`` 0 that
    made the throttle fire every step (pitfall #16).

    The helper degrades to 0 when ``loop_state`` is absent, so the bare
    ``_Strat(AdversarialMixin)`` stub used by the other tests in this module
    (no strategy base, no loop_state) stays safe."""
    code = "\n".join(
        ln
        for ln in inspect.getsource(
            AdversarialMixin.train_step_adversarial
        ).splitlines()
        if not ln.lstrip().startswith("#")
    )
    assert "self.env.step" not in code
    assert "resolve_loop_iteration(self)" in code
