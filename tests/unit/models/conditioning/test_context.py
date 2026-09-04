"""Tests for the canonical ConditioningContext SSOT (2026-05-26).

The context bundles every condition source (diffusion timestep, acquisition
order, spatial coords, degradation severity, acquisition params, contrast,
motion pose) as named optional tensors so a strategy populates only the
active ones and a model consumes a single object.
"""

from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.conditioning import ConditioningContext  # noqa: E402


def test_empty_context_has_no_active_sources() -> None:
    assert ConditioningContext().active_sources == ()


def test_active_sources_lists_only_provided_fields() -> None:
    ctx = ConditioningContext(
        diffusion_t=torch.zeros(2), severity_vec=torch.zeros(2, 5)
    )
    assert set(ctx.active_sources) == {"diffusion_t", "severity_vec"}


def test_extra_sources_appear_in_active() -> None:
    ctx = ConditioningContext(extra={"foo": torch.zeros(2, 3)})
    assert "foo" in ctx.active_sources


def test_batch_size_inferred_from_first_tensor() -> None:
    assert ConditioningContext(severity_vec=torch.zeros(4, 5)).batch_size == 4
    assert ConditioningContext().batch_size is None


def test_to_moves_all_tensors() -> None:
    ctx = ConditioningContext(
        diffusion_t=torch.zeros(2), extra={"foo": torch.zeros(2, 3)}
    )
    moved = ctx.to("cpu")
    assert moved.diffusion_t.device.type == "cpu"
    assert moved.extra["foo"].device.type == "cpu"


def test_frozen_cannot_mutate() -> None:
    ctx = ConditioningContext(diffusion_t=torch.zeros(2))
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.diffusion_t = torch.ones(2)  # type: ignore[misc]


def test_has_helper() -> None:
    ctx = ConditioningContext(diffusion_t=torch.zeros(2))
    assert ctx.has("diffusion_t")
    assert not ctx.has("severity_vec")
    assert not ctx.has("not_a_field")
