"""Tests for the MetricContext seam (mriforge.core.metrics.context)."""

from __future__ import annotations

import pytest
import torch

from mriforge.core.metrics.context import MetricContext, resolve_context


def test_defaults_all_none() -> None:
    ctx = MetricContext()
    assert ctx.y_kspace is None
    assert ctx.mask is None
    assert ctx.coil_maps is None
    assert ctx.nr_params is None


def test_frozen() -> None:
    from dataclasses import FrozenInstanceError

    ctx = MetricContext()
    with pytest.raises(FrozenInstanceError):
        ctx.y_kspace = torch.zeros(1)  # type: ignore[misc]


def test_has() -> None:
    k = torch.zeros(1, 1, 4, 4)
    ctx = MetricContext(y_kspace=k, mask=k)
    assert ctx.has("y_kspace", "mask")
    assert not ctx.has("y_kspace", "coil_maps")
    assert not ctx.has("coil_maps")


def test_require_raises_on_missing() -> None:
    ctx = MetricContext(y_kspace=torch.zeros(1))
    ctx.require("ndcr", "y_kspace")  # present → no raise
    with pytest.raises(ValueError, match="requires MetricContext fields"):
        ctx.require("ndcr", "y_kspace", "coil_maps")


def test_from_kwargs_maps_sensitivity_maps_alias() -> None:
    smaps = torch.zeros(1, 4, 8, 8)
    ctx = MetricContext.from_kwargs({"sensitivity_maps": smaps, "acceleration": 4.0})
    assert ctx.coil_maps is smaps
    assert ctx.acceleration == 4.0


def test_from_kwargs_ignores_unknown() -> None:
    ctx = MetricContext.from_kwargs({"not_a_field": 1, "mask": torch.zeros(2)})
    assert ctx.mask is not None
    # unknown key did not crash and is not stored anywhere
    assert not hasattr(ctx, "not_a_field")


def test_resolve_context_passthrough() -> None:
    ctx = MetricContext(acceleration=2.0)
    assert resolve_context(ctx, {}) is ctx


def test_resolve_context_builds_from_kwargs() -> None:
    out = resolve_context(None, {"acceleration": 3.0})
    assert isinstance(out, MetricContext)
    assert out.acceleration == 3.0
