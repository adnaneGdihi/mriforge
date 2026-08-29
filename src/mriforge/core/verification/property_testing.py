"""
Property-Based Testing Framework (Lightweight Hypothesis Replacement).

This module provides a minimal implementation of property-based testing primitives
to enable invariant verification in environments where `hypothesis` is not available.

Features:
- @given decorator for test generation
- Strategies: tensors, floats, integers
- Shrinking (Not implemented in v1 prototype)
"""

import logging
import random
from typing import Any

import torch

logger = logging.getLogger(__name__)


class SearchStrategy:
    """Base class for data generation strategies."""

    def draw(self) -> Any:
        """Generate a single example."""
        raise NotImplementedError


class FloatStrategy(SearchStrategy):
    """Generates floating point numbers."""

    def __init__(self, min_value: float = -1e6, max_value: float = 1e6):
        """__init__.

        Args:
            min_value (float): Description.
            max_value (float): Description.
        """
        self.min_value = min_value
        self.max_value = max_value

    def draw(self) -> float:
        """draw.

        Returns:
            float: Description.
        """
        return random.uniform(self.min_value, self.max_value)


class IntegerStrategy(SearchStrategy):
    """Generates integers."""

    def __init__(self, min_value: int = -100, max_value: int = 100):
        """__init__.

        Args:
            min_value (int): Description.
            max_value (int): Description.
        """
        self.min_value = min_value
        self.max_value = max_value

    def draw(self) -> int:
        """draw.

        Returns:
            int: Description.
        """
        return random.randint(self.min_value, self.max_value)


class TensorStrategy(SearchStrategy):
    """Generates PyTorch Tensors with random shapes and values."""

    def __init__(
        self,
        shape: tuple[int, ...] | None = None,
        min_dim: int = 1,
        max_dim: int = 64,
        ndim: int = 4,
        dtype: torch.dtype = torch.float32,
        min_value: float = -10.0,
        max_value: float = 10.0,
    ):
        """__init__.

        Args:
            shape (Optional[tuple[int, ...]]): Description.
            min_dim (int): Description.
            max_dim (int): Description.
            ndim (int): Description.
            dtype (torch.dtype): Description.
            min_value (float): Description.
            max_value (float): Description.
        """
        self.shape = shape
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.ndim = ndim
        self.dtype = dtype
        self.min_value = min_value
        self.max_value = max_value

    def draw(self) -> torch.Tensor:
        """draw.

        Returns:
            torch.Tensor: Description.
        """
        if self.shape:
            target_shape = self.shape
        else:
            # Generate random shape
            target_shape = tuple(
                random.randint(self.min_dim, self.max_dim) for _ in range(self.ndim)
            )

        if self.dtype in (torch.float32, torch.float16, torch.float64):
            tensor = (
                torch.rand(target_shape, dtype=self.dtype) * (self.max_value - self.min_value)
                + self.min_value
            )
        elif self.dtype in (torch.complex64, torch.complex128):
            real = torch.rand(target_shape) * (self.max_value - self.min_value) + self.min_value
            imag = torch.rand(target_shape) * (self.max_value - self.min_value) + self.min_value
            tensor = torch.complex(real, imag).to(dtype=self.dtype)
        elif self.dtype in (torch.int32, torch.int64):
            tensor = torch.randint(
                int(self.min_value),
                int(self.max_value),
                target_shape,
                dtype=self.dtype,
            )
        else:
            raise NotImplementedError(f"Strategy for {self.dtype} not implemented")

        return tensor


def floats(min_value=-1e4, max_value=1e4) -> FloatStrategy:
    """floats.

    Args:
        min_value (Any): Description.
        max_value (Any): Description.
    Returns:
        FloatStrategy: Description.
    """
    return FloatStrategy(min_value, max_value)


def integers(min_value=-100, max_value=100) -> IntegerStrategy:
    """integers.

    Args:
        min_value (Any): Description.
        max_value (Any): Description.
    Returns:
        IntegerStrategy: Description.
    """
    return IntegerStrategy(min_value, max_value)


def tensors(
    shape=None,
    min_dim=1,
    max_dim=64,
    ndim=4,
    dtype=torch.float32,
    min_value=-10.0,
    max_value=10.0,
) -> TensorStrategy:
    """tensors.

    Args:
        shape (Any): Description.
        min_dim (Any): Description.
        max_dim (Any): Description.
        ndim (Any): Description.
        dtype (Any): Description.
        min_value (Any): Lower bound on generated values (forwarded to
            ``TensorStrategy``; previously dropped so the bound was a no-op).
        max_value (Any): Upper bound on generated values.
    Returns:
        TensorStrategy: Description.
    """
    return TensorStrategy(
        shape=shape,
        min_dim=min_dim,
        max_dim=max_dim,
        ndim=ndim,
        dtype=dtype,
        min_value=min_value,
        max_value=max_value,
    )


def given(*strategies, **kwargs_strategies):
    """
    Decorator to run a test function multiple times with generated data.

    Usage:
        @given(t=tensors(ndim=2), x=floats())
        def test_prop(t, x):
            assert ...
    """
    num_examples = 10  # Default number of fuzzing iterations

    def decorator(func):
        # Do not use functools.wraps(func) here because it preserves the signature
        # which confuses pytest (it looks for arguments as fixtures).
        # Instead, we manually copy metadata but keep the wrapper signature generic.
        """decorator.

        Args:
            func (Any): Description.
        Returns:
            Any: Description.
        """

        def wrapper(*args, **kwargs):
            # Parse arguments to find strategies
            # We support @given(strategy1, strategy2) -> passed as args to func
            # And @given(x=strategy1) -> passed as kwargs to func

            """wrapper.

            Returns:
                Any: Description.
            """
            for i in range(num_examples):
                # Draw from positional strategies
                drawn_args = [s.draw() for s in strategies]

                # Draw from keyword strategies
                drawn_kwargs = {k: s.draw() for k, s in kwargs_strategies.items()}

                # Combine with any manually passed args (usually none for test helpers)
                final_args = list(args) + drawn_args
                final_kwargs = {**kwargs, **drawn_kwargs}

                try:
                    func(*final_args, **final_kwargs)
                except Exception as e:
                    logger.error("FAILED with args: %s, kwargs: %s", drawn_args, drawn_kwargs)
                    raise e

            logger.info("PASSED %s randomized examples.", num_examples)

        # Manually copy metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__module__ = func.__module__

        return wrapper

    return decorator
