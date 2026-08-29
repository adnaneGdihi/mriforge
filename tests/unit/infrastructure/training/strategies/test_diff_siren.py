"""Unit tests for DIFFSirenStrategy fail-loud + iteration-threading contract.

The strategy's two distinctive terms (cross-contrast LNCC, SIREN PDE Laplacian)
were wrapped in bare ``except Exception`` that zeroed / dropped them on any
error, silently collapsing the arm to a plain reconstruction while smoke-PASSing
(pitfalls #9/#16). And the loss computer was handed ``kwargs.get("step", 0)`` —
a key the training loop never sets (it passes ``iteration``), freezing any
step-gated loss schedule at 0. These tests pin the fixed behaviour.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mriforge.infrastructure.training.strategies.diff_siren import (  # noqa: E402
    DIFFSirenStrategy,
)


class _StubLossOutput:
    def __init__(self, total: torch.Tensor) -> None:
        self.total = total
        self.components: dict[str, torch.Tensor] = {}


class _RecordingLossComputer:
    def __init__(self) -> None:
        self.last_iteration: int | None = None

    def compute(self, *, pred, target, epoch, iteration, losses_dict):  # noqa: ANN001
        self.last_iteration = iteration
        return _StubLossOutput(pred.abs().mean())


class _IdentityModel:
    """No ``siren`` attribute -> the PDE branch is skipped."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.clone().requires_grad_(True)


def _strategy(lncc):
    s = DIFFSirenStrategy.__new__(DIFFSirenStrategy)
    s.device = torch.device("cpu")
    s.env = SimpleNamespace(losses={})
    s.model = _IdentityModel()
    s.loss_computer = _RecordingLossComputer()
    s.lncc = lncc
    s.lambda_img = 1.0
    s.lambda_pde = 1e-3
    s.curriculum_switch_epoch = 10
    s.is_stage_2 = True
    return s


def test_iteration_threaded_from_loop_state() -> None:
    s = _strategy(lncc=lambda p, t: torch.tensor(0.0, requires_grad=True))
    s.loop_state = SimpleNamespace(iteration=77)
    x = torch.randn(1, 1, 8, 8)
    s._compute_losses_impl(x, x, epoch=0, batch=None)
    assert s.loss_computer.last_iteration == 77


def test_lncc_failure_propagates() -> None:
    """A raising LNCC must NOT be silently zeroed — the cross-contrast term is
    the strategy's distinctive objective."""

    def _boom(pred, target):
        raise RuntimeError("LNCC blew up")

    s = _strategy(lncc=_boom)
    x = torch.randn(1, 1, 8, 8)
    with pytest.raises(RuntimeError, match="LNCC"):
        s._compute_losses_impl(x, x, epoch=0, batch=None)


def test_lncc_success_contributes_to_total() -> None:
    s = _strategy(lncc=lambda p, t: (p - t).abs().mean())
    x = torch.randn(1, 1, 8, 8)
    out = s._compute_losses_impl(x, x, epoch=0, batch=None)
    assert "loss_lncc" in out
    assert "g_total_loss" in out
    assert out["g_total_loss"].requires_grad
