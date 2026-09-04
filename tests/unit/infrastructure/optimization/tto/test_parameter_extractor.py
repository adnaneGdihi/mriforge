"""Unit tests for infrastructure/optimization/tto/parameter_extractor.py.

Covers: OptimizableParameters, ParameterExtractor.
"""

from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_optimizable_parameters_canary():
    """Basic container holds tensors and returns correct list."""
    from spectramr.infrastructure.optimization.tto.parameter_extractor import (
        OptimizableParameters,
    )

    means = torch.randn(10, 3)
    scales = torch.randn(10, 3)
    params = OptimizableParameters(means=means, scales=scales)
    lst = params.to_list()
    assert len(lst) == 2
    assert params.is_valid()


# ---------------------------------------------------------------------------
# Parametrized
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs,expected_len",
    [
        ({}, 0),
        ({"means": torch.zeros(5, 3)}, 1),
        ({"means": torch.zeros(5, 3), "opacity": torch.zeros(5, 1)}, 2),
        (
            {
                "means": torch.zeros(5, 3),
                "scales": torch.zeros(5, 3),
                "rotations": torch.zeros(5, 4),
                "features": torch.zeros(5, 8),
                "opacity": torch.zeros(5, 1),
            },
            5,
        ),
    ],
)
def test_to_list_length(kwargs, expected_len):
    """to_list length matches number of non-None fields."""
    from spectramr.infrastructure.optimization.tto.parameter_extractor import (
        OptimizableParameters,
    )

    params = OptimizableParameters(**kwargs)
    assert len(params.to_list()) == expected_len


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_is_valid_without_means_returns_false():
    """Container without means is considered invalid."""
    from spectramr.infrastructure.optimization.tto.parameter_extractor import (
        OptimizableParameters,
    )

    params = OptimizableParameters(scales=torch.zeros(5, 3))
    assert not params.is_valid()


@pytest.mark.unit
def test_parameter_extractor_with_direct_attrs():
    """ParameterExtractor handles an object with .means attribute."""
    from spectramr.infrastructure.optimization.tto.parameter_extractor import (
        ParameterExtractor,
    )

    class FakeGaussian:
        means = torch.randn(4, 3)
        scales = torch.randn(4, 3)
        quaternions = torch.randn(4, 4)
        features = torch.randn(4, 8)
        opacity = torch.randn(4, 1)

    extractor = ParameterExtractor(device=torch.device("cpu"))
    result = extractor.extract(FakeGaussian())
    assert result.is_valid()
    assert result.means is not None
    assert result.means.requires_grad


# ---------------------------------------------------------------------------
# Expected failure
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_parameter_extractor_non_tensor_ignored():
    """Non-tensor attributes are silently ignored (not raised)."""
    from spectramr.infrastructure.optimization.tto.parameter_extractor import (
        ParameterExtractor,
    )

    class BadGaussian:
        means = [1.0, 2.0, 3.0]  # list, not tensor

    extractor = ParameterExtractor(device=torch.device("cpu"))
    result = extractor.extract(BadGaussian())
    # means was a list, so make_optimizable returns None
    assert not result.is_valid()
