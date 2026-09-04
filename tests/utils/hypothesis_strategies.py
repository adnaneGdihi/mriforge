"""Hypothesis strategies for property-based tests in spectraMR.

This module requires the ``hypothesis`` library.  If it is not installed
the import guard at the top of each public function raises a clear
``ImportError`` rather than crashing at import time — that way test
*collection* still works and individual tests are skipped gracefully via
``pytest.importorskip("hypothesis")``.

Usage::

    from hypothesis import given
    from tests.utils.hypothesis_strategies import tensor_shapes, dtypes_real

    @given(shape=tensor_shapes(dims=2), dtype=dtypes_real())
    def test_something(shape, dtype):
        import torch
        x = torch.randn(shape, dtype=dtype)
        ...

Note: all strategies are lazy — the ``hypothesis`` import only happens
inside each function so that this module is importable even when
``hypothesis`` is absent.
"""

from __future__ import annotations

import sys

# Guard: try to import hypothesis at module level only to detect presence.
_HYPOTHESIS_AVAILABLE: bool = False
try:
    import hypothesis  # noqa: F401 — presence check only
    _HYPOTHESIS_AVAILABLE = True
except ImportError:
    pass


def _require_hypothesis():
    """Raise a clear ``ImportError`` if hypothesis is not installed."""
    if not _HYPOTHESIS_AVAILABLE:
        raise ImportError(
            "The 'hypothesis' package is required for property-based tests. "
            "Install it with: pip install hypothesis"
        )


# ---------------------------------------------------------------------------
# Shape strategies
# ---------------------------------------------------------------------------

def tensor_shapes(
    dims: int = 2,
    min_size: int = 8,
    max_size: int = 64,
):
    """Strategy producing ``dims``-tuples of spatial sizes.

    Each element is drawn independently from ``[min_size, max_size]``
    (inclusive).  The result is a plain Python ``tuple`` so it can be
    passed directly to ``torch.randn``.

    Args:
        dims: Number of spatial dimensions (default 2 for 2D images).
        min_size: Minimum size per dimension (inclusive, default 8).
        max_size: Maximum size per dimension (inclusive, default 64).

    Returns:
        A :class:`hypothesis.strategies.SearchStrategy` of ``tuple[int, ...]``.
    """
    _require_hypothesis()
    from hypothesis import strategies as st  # noqa: PLC0415

    single_dim = st.integers(min_value=min_size, max_value=max_size)
    return st.tuples(*[single_dim] * dims)


# ---------------------------------------------------------------------------
# Dtype strategies
# ---------------------------------------------------------------------------

def dtypes_complex():
    """Strategy producing complex torch dtypes.

    Returns:
        A strategy drawing from ``{torch.complex64, torch.complex128}``.
    """
    _require_hypothesis()
    import torch  # noqa: PLC0415
    from hypothesis import strategies as st  # noqa: PLC0415

    return st.sampled_from([torch.complex64, torch.complex128])


def dtypes_real():
    """Strategy producing real-valued torch dtypes.

    Returns:
        A strategy drawing from ``{torch.float32, torch.float64}``.
        ``float16`` and ``bfloat16`` are excluded because many CPU ops do
        not support them (autocast is GPU-only in most paths).
    """
    _require_hypothesis()
    import torch  # noqa: PLC0415
    from hypothesis import strategies as st  # noqa: PLC0415

    return st.sampled_from([torch.float32, torch.float64])


# ---------------------------------------------------------------------------
# MRI-specific strategies
# ---------------------------------------------------------------------------

def acceleration_factors():
    """Strategy producing realistic MRI acceleration factors.

    Returns:
        A strategy drawing integers from ``{2, 3, 4, 6, 8}``.
    """
    _require_hypothesis()
    from hypothesis import strategies as st  # noqa: PLC0415

    return st.sampled_from([2, 3, 4, 6, 8])


def coil_counts():
    """Strategy producing realistic receive-coil counts.

    Returns:
        A strategy drawing from ``{1, 4, 8, 16, 32}``.
    """
    _require_hypothesis()
    from hypothesis import strategies as st  # noqa: PLC0415

    return st.sampled_from([1, 4, 8, 16, 32])


def bloch_params():
    """Strategy producing physiologically plausible Bloch equation parameters.

    Generates ``(T1, T2, M0)`` tuples (all in seconds for T1/T2) where:

    * ``T1`` ∈ ``[0.1, 4.5]`` s (grey matter ~1.3 s, CSF ~4.0 s)
    * ``T2`` ∈ ``[0.01, T1]`` s  (T2 ≤ T1 always; WM T2 ~70 ms, CSF ~500 ms)
    * ``M0`` ∈ ``[0.1, 1.0]`` (normalised equilibrium magnetisation)

    The ``T2 ≤ T1`` constraint is enforced by the filter, so hypothesis
    may need to discard draws — the constraint is physiologically
    non-trivial to express as a direct strategy, but the rejection rate
    is low because T2 is drawn from the full ``[0.01, T1]`` range.

    Returns:
        A strategy of ``(float, float, float)`` tuples satisfying
        ``0.01 ≤ T2 ≤ T1 ≤ 4.5`` and ``0.1 ≤ M0 ≤ 1.0``.
    """
    _require_hypothesis()
    from hypothesis import strategies as st  # noqa: PLC0415

    t1_strategy = st.floats(min_value=0.1, max_value=4.5, allow_nan=False, allow_infinity=False)
    m0_strategy = st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False)

    @st.composite
    def _bloch(draw):
        t1 = draw(t1_strategy)
        # T2 must be in [0.01, T1]
        t2 = draw(
            st.floats(
                min_value=0.01,
                max_value=t1,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        m0 = draw(m0_strategy)
        return (t1, t2, m0)

    return _bloch()
