"""Unit tests for edge Q-KAN quantization.

Regression coverage for the silent-fallback fix in
``src/mriforge/models/edge/quantization.py``: unsupported bit-widths must RAISE
(pitfall #9) rather than silently degrading to an identity / unquantized layer.
"""

import pytest
import torch

from mriforge.models.edge.quantization import (
    QKANLayer,
    SplineQuantizer,
    create_q_kan,
)


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# SplineQuantizer.forward — supported bit-widths still behave correctly
# --------------------------------------------------------------------------- #
def test_spline_quantizer_1bit_binary() -> None:
    x = torch.tensor([-0.8, -0.2, 0.2, 0.8])
    q = SplineQuantizer.apply(x, 1)
    assert torch.allclose(q, torch.tensor([-1.0, -1.0, 1.0, 1.0]))


def test_spline_quantizer_2bit_ternary() -> None:
    x = torch.tensor([-0.8, -0.2, 0.2, 0.8])
    q = SplineQuantizer.apply(x, 2)
    assert torch.allclose(q, torch.tensor([-1.0, 0.0, 0.0, 1.0]))


def test_ste_backward_passthrough() -> None:
    """Straight-through estimator passes gradients unchanged."""
    x = torch.randn(4, requires_grad=True)
    q = SplineQuantizer.apply(x, 2)
    q.sum().backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.ones_like(x))


# --------------------------------------------------------------------------- #
# SplineQuantizer.forward — unsupported bit-widths must RAISE (no silent id)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bits", [0, 3, 4, 8])
def test_spline_quantizer_unsupported_bits_raises(bits: int) -> None:
    with pytest.raises(ValueError, match="Unsupported quantization bits"):
        SplineQuantizer.apply(torch.randn(4), bits)


# --------------------------------------------------------------------------- #
# QKANLayer.__init__ — validate bits at construction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bits", [1, 2])
def test_qkan_layer_supported_bits_build_and_run(bits: int) -> None:
    layer = QKANLayer(in_features=8, out_features=8, grid_size=4, bits=bits)
    x = torch.randn(2, 16, 8)
    out = layer(x)
    assert out.shape == (2, 16, 8)


@pytest.mark.parametrize("bits", [4, 8])
def test_qkan_layer_unsupported_bits_raises(bits: int) -> None:
    with pytest.raises(ValueError, match="unsupported bits"):
        QKANLayer(in_features=8, out_features=8, grid_size=4, bits=bits)


def test_create_q_kan_propagates_bits_validation() -> None:
    """The factory must surface the same validation (no silent no-op)."""
    with pytest.raises(ValueError, match="unsupported bits"):
        create_q_kan(in_features=8, out_features=8, grid_size=4, bits=8)
    layer = create_q_kan(in_features=8, out_features=8, grid_size=4, bits=2)
    assert isinstance(layer, QKANLayer)
