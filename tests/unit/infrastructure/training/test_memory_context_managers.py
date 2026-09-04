"""Unit tests for memory management context managers.

Targets ``spectramr.infrastructure.training.memory_context_managers``:
- ``memory_cleanup_context``
- ``memory_efficient_context``
- ``gpu_memory_monitor``

All tests are CPU-safe: the managers branch on ``torch.cuda.is_available()``
internally, so GPU-specific paths are skipped automatically on CPU.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------


def test_canary_memory_cleanup_context_yields() -> None:
    from spectramr.infrastructure.training.memory_context_managers import (
        memory_cleanup_context,
    )

    result = []
    with memory_cleanup_context("test_op"):
        result.append(1)
    assert result == [1]


# ---------------------------------------------------------------------------
# memory_cleanup_context
# ---------------------------------------------------------------------------


def test_memory_cleanup_context_passes_through_value() -> None:
    from spectramr.infrastructure.training.memory_context_managers import (
        memory_cleanup_context,
    )

    accumulator = []
    with memory_cleanup_context("canary"):
        accumulator.extend([10, 20, 30])
    assert accumulator == [10, 20, 30]


def test_memory_cleanup_context_reraises_generic_exception() -> None:
    from spectramr.infrastructure.training.memory_context_managers import (
        memory_cleanup_context,
    )

    with pytest.raises(RuntimeError, match="test error"):
        with memory_cleanup_context("bad_op"):
            raise RuntimeError("test error")


# ---------------------------------------------------------------------------
# memory_efficient_context
# ---------------------------------------------------------------------------


def test_memory_efficient_context_executes_body() -> None:
    from spectramr.infrastructure.training.memory_context_managers import (
        memory_efficient_context,
    )

    side_effect = []
    with memory_efficient_context(enable_gc=True, empty_cache=False):
        side_effect.append("done")
    assert side_effect == ["done"]


@pytest.mark.parametrize("enable_gc,empty_cache", [(True, False), (False, False)])
def test_memory_efficient_context_param_combinations(enable_gc, empty_cache) -> None:
    from spectramr.infrastructure.training.memory_context_managers import (
        memory_efficient_context,
    )

    # Should not raise regardless of combination.
    with memory_efficient_context(enable_gc=enable_gc, empty_cache=empty_cache):
        pass


def test_memory_efficient_context_reraises() -> None:
    from spectramr.infrastructure.training.memory_context_managers import (
        memory_efficient_context,
    )

    with pytest.raises(ValueError, match="sentinel"):
        with memory_efficient_context():
            raise ValueError("sentinel")


# ---------------------------------------------------------------------------
# gpu_memory_monitor
# ---------------------------------------------------------------------------


def test_gpu_memory_monitor_yields_stats_dict() -> None:
    from spectramr.infrastructure.training.memory_context_managers import (
        gpu_memory_monitor,
    )

    with gpu_memory_monitor("test_monitor") as stats:
        pass
    assert isinstance(stats, dict)
    assert "peak_memory_mb" in stats
    assert "initial_memory_mb" in stats


def test_gpu_memory_monitor_stats_non_negative() -> None:
    from spectramr.infrastructure.training.memory_context_managers import (
        gpu_memory_monitor,
    )

    with gpu_memory_monitor("nn") as stats:
        pass
    # On CPU the values stay 0.0
    for key, val in stats.items():
        assert isinstance(val, float), f"stats[{key}] should be float"


# ---------------------------------------------------------------------------
# Edge: empty context body
# ---------------------------------------------------------------------------


def test_edge_all_context_managers_work_with_no_body() -> None:
    from spectramr.infrastructure.training.memory_context_managers import (
        gpu_memory_monitor,
        memory_cleanup_context,
        memory_efficient_context,
    )

    with memory_cleanup_context():
        pass
    with memory_efficient_context():
        pass
    with gpu_memory_monitor():
        pass
