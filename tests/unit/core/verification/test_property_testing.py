"""Tests for ``mriforge.core.verification.property_testing``.

WS-2 Core closes the ``core/verification/`` 0-test gap called out in the
campaign plan (``TODO/createadetailedplancachedcascade.md`` §WS-2). The module
is a lightweight, dependency-free stand-in for ``hypothesis`` used to fuzz
invariants when the real library is absent, so these tests pin its contract:

* each :class:`SearchStrategy` ``draw()`` honours its declared bounds / dtype,
* unsupported dtypes raise (no silent fallback — CLAUDE.md #9),
* the ``@given`` decorator runs the wrapped property ``num_examples`` times,
  forwards positional **and** keyword strategies, and re-raises on failure.
"""

from __future__ import annotations

import pytest
import torch

from mriforge.core.verification.property_testing import (
    FloatStrategy,
    IntegerStrategy,
    SearchStrategy,
    TensorStrategy,
    floats,
    given,
    integers,
    tensors,
)

pytestmark = pytest.mark.unit


class TestSearchStrategyBase:
    def test_base_draw_is_abstract(self) -> None:
        with pytest.raises(NotImplementedError):
            SearchStrategy().draw()


class TestFloatStrategy:
    def test_draw_within_bounds(self) -> None:
        s = FloatStrategy(min_value=-2.0, max_value=5.0)
        for _ in range(256):
            v = s.draw()
            assert isinstance(v, float)
            assert -2.0 <= v <= 5.0

    def test_degenerate_range_is_constant(self) -> None:
        s = FloatStrategy(min_value=3.0, max_value=3.0)
        assert s.draw() == pytest.approx(3.0)


class TestIntegerStrategy:
    def test_draw_within_bounds_inclusive(self) -> None:
        s = IntegerStrategy(min_value=-3, max_value=3)
        seen: set[int] = set()
        for _ in range(512):
            v = s.draw()
            assert isinstance(v, int)
            assert -3 <= v <= 3
            seen.add(v)
        # ``random.randint`` is inclusive on both ends — both endpoints reachable.
        assert -3 in seen and 3 in seen


class TestTensorStrategy:
    def test_explicit_shape_and_float_bounds(self) -> None:
        s = TensorStrategy(shape=(2, 3, 4, 4), min_value=-1.0, max_value=1.0)
        t = s.draw()
        assert tuple(t.shape) == (2, 3, 4, 4)
        assert t.dtype == torch.float32
        assert torch.all(t >= -1.0) and torch.all(t <= 1.0)

    def test_random_shape_respects_ndim_and_dim_bounds(self) -> None:
        s = TensorStrategy(min_dim=2, max_dim=5, ndim=3)
        for _ in range(32):
            t = s.draw()
            assert t.dim() == 3
            assert all(2 <= d <= 5 for d in t.shape)

    def test_complex_dtype(self) -> None:
        s = TensorStrategy(shape=(2, 2), dtype=torch.complex64)
        t = s.draw()
        assert t.is_complex()
        assert t.dtype == torch.complex64
        assert tuple(t.shape) == (2, 2)

    def test_integer_dtype(self) -> None:
        # ``torch.randint`` treats ``high`` as exclusive.
        s = TensorStrategy(shape=(64,), dtype=torch.int64, min_value=0, max_value=10)
        t = s.draw()
        assert t.dtype == torch.int64
        assert tuple(t.shape) == (64,)
        assert torch.all(t >= 0) and torch.all(t < 10)

    def test_unsupported_dtype_raises(self) -> None:
        s = TensorStrategy(shape=(2,), dtype=torch.bool)
        with pytest.raises(NotImplementedError):
            s.draw()


class TestFactories:
    def test_floats_factory(self) -> None:
        s = floats(-1.0, 1.0)
        assert isinstance(s, FloatStrategy)
        assert (s.min_value, s.max_value) == (-1.0, 1.0)

    def test_integers_factory(self) -> None:
        s = integers(0, 9)
        assert isinstance(s, IntegerStrategy)
        assert (s.min_value, s.max_value) == (0, 9)

    def test_tensors_factory(self) -> None:
        s = tensors(ndim=2, max_dim=8)
        assert isinstance(s, TensorStrategy)
        assert s.ndim == 2
        assert s.max_dim == 8

    def test_tensors_factory_plumbs_value_bounds(self) -> None:
        # Regression: tensors() previously dropped min_value/max_value, so the
        # bounds were a silent no-op (the factory always used TensorStrategy's
        # ±10 defaults). A degenerate [0, 0] range must now yield all-zeros.
        s = tensors(shape=(2, 2), min_value=0.0, max_value=0.0)
        assert (s.min_value, s.max_value) == (0.0, 0.0)
        assert torch.equal(s.draw(), torch.zeros(2, 2))

    def test_tensors_factory_bounds_respected(self) -> None:
        s = tensors(shape=(4, 4), min_value=2.0, max_value=3.0)
        t = s.draw()
        assert torch.all(t >= 2.0) and torch.all(t <= 3.0)


class TestGivenDecorator:
    def test_runs_num_examples_times(self) -> None:
        calls = {"n": 0}

        @given(x=floats(-1.0, 1.0))
        def prop(x: float) -> None:
            calls["n"] += 1
            assert -1.0 <= x <= 1.0

        prop()
        assert calls["n"] == 10  # module default num_examples

    def test_positional_strategy_forwarding(self) -> None:
        seen: list[int] = []

        @given(integers(0, 5))
        def prop(v: int) -> None:
            seen.append(v)

        prop()
        assert len(seen) == 10
        assert all(0 <= v <= 5 for v in seen)

    def test_failure_propagates(self) -> None:
        @given(x=floats(0.0, 1.0))
        def prop(x: float) -> None:
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            prop()

    def test_metadata_preserved_for_pytest(self) -> None:
        @given(x=floats())
        def my_prop(x: float) -> None:
            """A docstring."""

        # The wrapper deliberately avoids functools.wraps (so pytest does not
        # treat strategy args as fixtures) but still copies name/doc/module.
        assert my_prop.__name__ == "my_prop"
        assert my_prop.__doc__ == "A docstring."


@pytest.mark.property
@given(t=tensors(shape=(2, 1, 8, 8), dtype=torch.float32))
def test_property_abs_is_nonnegative(t: torch.Tensor) -> None:
    """End-to-end: a ``@given``-decorated module-level property is collectable
    by pytest and exercises the strategy on every drawn example."""
    assert torch.all(t.abs() >= 0.0)
