"""Tests for ``IDownstreamModelService``.

Targets ``spectramr.domain.interfaces.i_downstream_model_service``. The
``runtime_checkable`` protocol introduced by ``CC-3`` of
``TODO/backlog_paradigm_expansion_roadmap.md``.

Categories:

- A duck-typed adapter implementing all four members satisfies
  ``isinstance(..., IDownstreamModelService)``
- A class missing one of the required members fails the check
- The protocol re-exports from ``spectramr.domain.interfaces``
"""

from __future__ import annotations

import torch

from spectramr.domain.interfaces import IDownstreamModelService


# ---------------------------------------------------------------------------
# Conformant adapter
# ---------------------------------------------------------------------------


class _DummyAdapter:
    @property
    def task_name(self) -> str:
        return "dummy_task"

    @property
    def num_classes(self) -> int | None:
        return 2

    def forward(self, reconstruction, **kwargs):
        return reconstruction.mean(dim=(-1, -2), keepdim=True)

    def compute_metric(self, reconstruction, ground_truth, **kwargs):
        return {"dummy_metric": 0.0}


def test_duck_typed_adapter_satisfies_protocol() -> None:
    """A class with the four members + the right shapes is an instance."""
    assert isinstance(_DummyAdapter(), IDownstreamModelService)


def test_dummy_forward_returns_tensor() -> None:
    """Forward path runs end-to-end on a tiny batch."""
    adapter = _DummyAdapter()
    out = adapter.forward(torch.zeros(1, 1, 4, 4))
    assert isinstance(out, torch.Tensor)


def test_dummy_compute_metric_returns_dict() -> None:
    """Compute-metric returns a mapping with at least one key."""
    adapter = _DummyAdapter()
    metrics = adapter.compute_metric(
        torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4)
    )
    assert isinstance(metrics, dict)
    assert len(metrics) >= 1


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


def test_missing_task_name_fails_protocol_check() -> None:
    """Class lacking ``task_name`` is not an instance."""

    class _NoTaskName:
        @property
        def num_classes(self) -> int | None:
            return None

        def forward(self, reconstruction, **kwargs):
            return reconstruction

        def compute_metric(self, reconstruction, ground_truth, **kwargs):
            return {}

    assert not isinstance(_NoTaskName(), IDownstreamModelService)


def test_missing_compute_metric_fails_protocol_check() -> None:
    """Class lacking ``compute_metric`` is not an instance."""

    class _NoCompute:
        task_name = "x"
        num_classes = None

        def forward(self, reconstruction, **kwargs):
            return reconstruction

    assert not isinstance(_NoCompute(), IDownstreamModelService)
