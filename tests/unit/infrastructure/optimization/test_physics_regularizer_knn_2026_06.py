"""Regression: TTO physics regularizer called KNN with non-existent kwargs.

2026-06-11 infrastructure audit. ``PhysicsRegularizer`` invoked the KNN graph
regularizer as ``knn_regularizer(means=..., features=...)`` but the forward
signature is ``(positions, attributes, attribute_weights=None)`` — a guaranteed
``TypeError`` whenever ``lambda_knn > 0``. The call now uses the right names.
"""

from __future__ import annotations

import inspect

import torch

from spectramr.infrastructure.optimization.regularizers.knn_graph import (
    KNNGraphRegularizer,
)


def test_knn_regularizer_accepts_positions_attributes_kwargs():
    reg = KNNGraphRegularizer(k=2)
    positions = torch.randn(1, 6, 3)
    attributes = torch.randn(1, 6, 4)
    loss = reg(positions=positions, attributes=attributes)
    assert isinstance(loss, torch.Tensor)


def test_physics_regularizer_callsite_uses_correct_kwargs():
    from spectramr.infrastructure.optimization.tto import physics_regularizer

    src = inspect.getsource(physics_regularizer)
    assert "positions=means_eval, attributes=params.features" in src
    # The broken kwargs must be gone.
    assert "knn_regularizer(means=" not in src
