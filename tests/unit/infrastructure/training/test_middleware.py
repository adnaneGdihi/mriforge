"""Tests for the training middleware composition chain.

Regression coverage for the bug where ``MiddlewareChain.wrap_backward`` only
executed the *last* registered middleware (it called
``self.middlewares[-1].wrap_backward(...)`` instead of composing the whole
chain). The forward path already composed correctly via nested closures; the
backward path silently dropped every middleware except the final one.
"""

from __future__ import annotations

from typing import Any

import torch

from mriforge.infrastructure.training.middleware import (
    GradientClippingMiddleware,
    MiddlewareChain,
    TrainingMiddleware,
)


class _RecordingMiddleware(TrainingMiddleware):
    """Middleware that records the order in which it runs forward/backward."""

    def __init__(self, name: str, log: list[str]) -> None:
        self.name = name
        self.log = log

    def wrap_forward(self, step_fn: Any, *args: Any, **kwargs: Any) -> Any:
        self.log.append(f"fwd-enter:{self.name}")
        result = step_fn(*args, **kwargs)
        self.log.append(f"fwd-exit:{self.name}")
        return result

    def wrap_backward(
        self,
        loss: Any,
        optimizer: Any,
        next_fn: Any = None,
        **kwargs: Any,
    ) -> None:
        self.log.append(f"bwd-enter:{self.name}")
        if next_fn is not None:
            next_fn(loss, optimizer, **kwargs)
        self.log.append(f"bwd-exit:{self.name}")


def _make_loss_and_optimizer() -> tuple[torch.Tensor, torch.optim.Optimizer, torch.nn.Parameter]:
    param = torch.nn.Parameter(torch.ones(4))
    optimizer = torch.optim.SGD([param], lr=0.1)
    loss = (param * 2.0).sum()
    return loss, optimizer, param


def test_wrap_backward_runs_all_middlewares_in_order() -> None:
    """RED before fix: only the last middleware ran. All three must run, in order."""
    log: list[str] = []
    chain = MiddlewareChain(
        [
            _RecordingMiddleware("A", log),
            _RecordingMiddleware("B", log),
            _RecordingMiddleware("C", log),
        ]
    )

    loss, optimizer, _ = _make_loss_and_optimizer()
    chain.wrap_backward(loss, optimizer)

    enters = [entry for entry in log if entry.startswith("bwd-enter")]
    assert enters == ["bwd-enter:A", "bwd-enter:B", "bwd-enter:C"], log

    # Onion ordering: first registered is outermost (enter first, exit last).
    assert log == [
        "bwd-enter:A",
        "bwd-enter:B",
        "bwd-enter:C",
        "bwd-exit:C",
        "bwd-exit:B",
        "bwd-exit:A",
    ]


def test_wrap_backward_single_middleware_still_runs() -> None:
    """A single middleware composes and the optimizer step is applied once."""
    log: list[str] = []
    chain = MiddlewareChain([_RecordingMiddleware("solo", log)])

    loss, optimizer, param = _make_loss_and_optimizer()
    chain.wrap_backward(loss, optimizer)

    assert log == ["bwd-enter:solo", "bwd-exit:solo"]
    # The canonical backward + a single SGD step ran (lr=0.1, grad==2 per element):
    # param moves 1.0 -> 0.8 exactly once. grad is None because the canonical core
    # calls zero_grad(set_to_none=True) after stepping (perf.md).
    assert param.grad is None
    assert torch.allclose(param.data, torch.full((4,), 0.8))


def test_wrap_backward_empty_chain_does_canonical_step() -> None:
    """Empty chain falls back to a single backward + optimizer step."""
    chain = MiddlewareChain([])
    loss, optimizer, param = _make_loss_and_optimizer()
    chain.wrap_backward(loss, optimizer)
    # grad is None after zero_grad(set_to_none=True); the step is observable in data.
    assert param.grad is None
    assert torch.allclose(param.data, torch.full((4,), 0.8))


def test_wrap_backward_real_middlewares_all_participate() -> None:
    """Recording + gradient-clipping middlewares both fire; clipping is applied."""
    log: list[str] = []
    clipper = GradientClippingMiddleware(max_norm=1.0, norm_type=2.0)
    chain = MiddlewareChain([_RecordingMiddleware("rec", log), clipper])

    param = torch.nn.Parameter(torch.ones(4))
    optimizer = torch.optim.SGD([param], lr=1.0)
    # grad of (param*10).sum() is 10 per element -> norm 20, must be clipped to 1.
    loss = (param * 10.0).sum()
    chain.wrap_backward(loss, optimizer)

    # The recording middleware ran (proves it is not skipped).
    assert log == ["bwd-enter:rec", "bwd-exit:rec"]
    # Clipping happened: post-step param reflects the clipped gradient, not raw 10.
    # With clipped total norm 1.0 across 4 elements, each grad ~= 0.5; lr=1 -> 1-0.5.
    assert torch.allclose(param.data, torch.full((4,), 0.5))


def test_wrap_forward_runs_all_middlewares_in_order() -> None:
    """Forward composition was already correct; guard against regression."""
    log: list[str] = []
    chain = MiddlewareChain(
        [
            _RecordingMiddleware("A", log),
            _RecordingMiddleware("B", log),
            _RecordingMiddleware("C", log),
        ]
    )

    def core() -> str:
        log.append("core")
        return "ok"

    result = chain.wrap_forward(core)
    assert result == "ok"
    assert log == [
        "fwd-enter:A",
        "fwd-enter:B",
        "fwd-enter:C",
        "core",
        "fwd-exit:C",
        "fwd-exit:B",
        "fwd-exit:A",
    ]
