"""Training Middleware Infrastructure
===================================

Composable middleware for training strategies following the Decorator pattern.
Middleware handles cross-cutting concerns like logging, AMP, and gradient clipping.

Compliance:
- soft_design.md: Decorator Pattern for cross-cutting concerns
- debugger.md: Structured JSON logging with correlation IDs
- perf.md: AMP via torch.amp.autocast
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch import Tensor
    from torch.optim import Optimizer

# A backward continuation: performs the remaining backward/step work for the
# rest of the middleware chain (innermost link is the canonical backward+step).
BackwardNext = Callable[..., None]

logger = logging.getLogger(__name__)


class TrainingMiddleware(ABC):
    """Base class for training middleware.

    Middleware wraps training steps to add cross-cutting functionality
    like logging, AMP, profiling, or gradient clipping.
    """

    @abstractmethod
    def wrap_forward(
        self,
        step_fn: callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Wrap a forward/training step.

        Args:
            step_fn: The original step function to wrap.
            *args: Arguments for the step function.
            **kwargs: Keyword arguments for the step function.

        Returns:
            Result from the step function.
        """
        ...

    @abstractmethod
    def wrap_backward(
        self,
        loss: Tensor,
        optimizer: Optimizer,
        next_fn: BackwardNext | None = None,
        **kwargs: Any,
    ) -> None:
        """Wrap backward pass and optimizer step.

        Args:
            loss: Loss tensor.
            optimizer: Optimizer instance.
            next_fn: Continuation that performs the rest of the middleware chain
                (the innermost link runs the canonical ``backward``/``step``).
                When ``None`` the middleware is the terminal link and must do the
                backward and optimizer step itself.
            **kwargs: Additional arguments.
        """
        ...


class LoggingMiddleware(TrainingMiddleware):
    """Middleware for structured logging with correlation IDs.

    Compliance: debugger.md Priority 0 - Structured JSON logging.
    """

    def __init__(
        self,
        log_interval: int = 100,
        include_timing: bool = True,
    ) -> None:
        """Initialize logging middleware.

        Args:
            log_interval: Log every N steps.
            include_timing: Include step timing in logs.
        """
        self.log_interval = log_interval
        self.include_timing = include_timing
        self.step_count = 0

    def wrap_forward(
        self,
        step_fn: callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Wrap forward step with timing and correlation ID."""
        self.step_count += 1
        correlation_id = str(uuid.uuid4())[:8]

        start_time = time.perf_counter() if self.include_timing else None

        try:
            result = step_fn(*args, **kwargs)

            if self.step_count % self.log_interval == 0:
                log_data = {
                    "step": self.step_count,
                    "correlation_id": correlation_id,
                }
                if self.include_timing and start_time is not None:
                    elapsed = time.perf_counter() - start_time
                    log_data["elapsed_ms"] = f"{elapsed * 1000:.2f}"

                # Extract losses if available
                if isinstance(result, dict):
                    # Filter for scalar values/floats for clean logging
                    losses = {
                        k: float(v) if hasattr(v, "item") else v
                        for k, v in result.items()
                        if isinstance(v, (int, float)) or hasattr(v, "item")
                    }
                    log_data.update(losses)
                    message = f"Training step completed: {losses}"
                else:
                    message = "Training step completed"

                logger.info(message, extra=log_data)

            return result

        except Exception as e:
            logger.exception(
                "Training step failed",
                extra={"correlation_id": correlation_id, "error": str(e)},
            )
            raise

    def wrap_backward(
        self,
        loss: Tensor,
        optimizer: Optimizer,
        next_fn: BackwardNext | None = None,
        **kwargs: Any,
    ) -> None:
        """Wrap backward pass (no additional logic for logging)."""
        if next_fn is not None:
            next_fn(loss, optimizer, **kwargs)
            return
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)  # perf.md: set_to_none=True


class AMPMiddleware(TrainingMiddleware):
    """Middleware for Automatic Mixed Precision training.

    Compliance:
    - perf.md: torch.autocast for FP16/BF16
    - perf.md: set_to_none=True in zero_grad
    """

    def __init__(
        self,
        device_type: str = "cuda",
        dtype: torch.dtype = torch.float16,
        enabled: bool = True,
    ) -> None:
        """Initialize AMP middleware.

        Args:
            device_type: Device type for autocast ('cuda', 'cpu').
            dtype: Target dtype (float16, bfloat16).
            enabled: Whether AMP is enabled.
        """
        self.device_type = device_type
        self.dtype = dtype
        self.enabled = enabled
        self.scaler = torch.amp.GradScaler() if enabled else None

    def wrap_forward(
        self,
        step_fn: callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Wrap forward step with autocast."""
        if not self.enabled:
            return step_fn(*args, **kwargs)

        with torch.amp.autocast(device_type=self.device_type, dtype=self.dtype):
            return step_fn(*args, **kwargs)

    def wrap_backward(
        self,
        loss: Tensor,
        optimizer: Optimizer,
        next_fn: BackwardNext | None = None,
        **kwargs: Any,
    ) -> None:
        """Wrap backward pass with grad scaling."""
        if not self.enabled or self.scaler is None:
            if next_fn is not None:
                next_fn(loss, optimizer, **kwargs)
                return
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            return

        # AMP owns scaling of the backward and optimizer step. When composed in a
        # chain it must be the innermost (terminal) link so scaling is correct;
        # there is no scaled continuation to delegate to.
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()
        optimizer.zero_grad(set_to_none=True)


class GradientClippingMiddleware(TrainingMiddleware):
    """Middleware for gradient clipping.

    Compliance: restruct.md Phase 3 - Gradient clipping at Trainer level.
    """

    def __init__(
        self,
        max_norm: float = 1.0,
        norm_type: float = 2.0,
    ) -> None:
        """Initialize gradient clipping middleware.

        Args:
            max_norm: Maximum gradient norm.
            norm_type: Type of norm to use.
        """
        self.max_norm = max_norm
        self.norm_type = norm_type

    def wrap_forward(
        self,
        step_fn: callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Forward pass (no modification)."""
        return step_fn(*args, **kwargs)

    def wrap_backward(
        self,
        loss: Tensor,
        optimizer: Optimizer,
        next_fn: BackwardNext | None = None,
        **kwargs: Any,
    ) -> None:
        """Backward pass with gradient clipping.

        Clipping must run *between* the backward and the optimizer step, so this
        middleware always owns the backward/clip/step sequence and is therefore a
        terminal (innermost) link in the chain. ``next_fn`` is accepted for a
        uniform signature but not delegated to — a continuation that performed
        its own ``step`` would step on unclipped gradients.
        """
        loss.backward()

        # Get parameters from optimizer
        for param_group in optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(
                param_group["params"],
                self.max_norm,
                self.norm_type,
            )

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


class MiddlewareChain:
    """Chain of middleware applied in order.

    Composes multiple middleware into a single callable chain.
    """

    def __init__(self, middlewares: list[TrainingMiddleware] | None = None) -> None:
        """Initialize middleware chain.

        Args:
            middlewares: List of middleware to apply in order.
        """
        self.middlewares = middlewares or []

    def add(self, middleware: TrainingMiddleware) -> MiddlewareChain:
        """Add middleware to chain.

        Args:
            middleware: Middleware to add.

        Returns:
            Self for chaining.
        """
        self.middlewares.append(middleware)
        return self

    def wrap_forward(
        self,
        step_fn: callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Apply all middleware wrapping to forward step."""
        # Build nested wrapping from inside out
        current_fn = step_fn
        for middleware in reversed(self.middlewares):
            prev_fn = current_fn
            current_fn = lambda *a, m=middleware, fn=prev_fn, **kw: m.wrap_forward(fn, *a, **kw)

        return current_fn(*args, **kwargs)

    def wrap_backward(
        self,
        loss: Tensor,
        optimizer: Optimizer,
        **kwargs: Any,
    ) -> None:
        """Apply backward through the full middleware chain.

        Every registered middleware runs, composed onion-style: the first
        registered middleware is the outermost link (it sees ``next_fn`` as the
        rest of the chain) and the innermost continuation performs the canonical
        ``backward``/``step``/``zero_grad``. A middleware that owns the backward
        and optimizer step itself (e.g. AMP scaling, gradient clipping) ignores
        its ``next_fn`` and acts as the terminal link.
        """

        def canonical(loss_t: Tensor, opt: Optimizer, **_: Any) -> None:
            loss_t.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        if not self.middlewares:
            canonical(loss, optimizer, **kwargs)
            return

        def _link(middleware: TrainingMiddleware, inner: BackwardNext) -> BackwardNext:
            def _wrapped(loss_t: Tensor, opt: Optimizer, **kw: Any) -> None:
                middleware.wrap_backward(loss_t, opt, next_fn=inner, **kw)

            return _wrapped

        # Build the continuation chain from inside out so the first registered
        # middleware ends up outermost (runs first), mirroring wrap_forward.
        next_fn: BackwardNext = canonical
        for middleware in reversed(self.middlewares):
            next_fn = _link(middleware, next_fn)

        next_fn(loss, optimizer, **kwargs)


__all__ = [
    "AMPMiddleware",
    "GradientClippingMiddleware",
    "LoggingMiddleware",
    "MiddlewareChain",
    "TrainingMiddleware",
]
