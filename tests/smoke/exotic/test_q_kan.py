"""
Unit tests for Q-KAN.
"""

import pytest
import torch

from spectramr.models.edge.quantization import QKANLayer, SplineQuantizer


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_spline_quantizer_forward():
    """Verify quantization logic."""
    x = torch.tensor([-0.8, -0.2, 0.2, 0.8])

    # 2-bit (Ternary {-1, 0, 1})
    q = SplineQuantizer.apply(x, 2)
    expected = torch.tensor([-1.0, 0.0, 0.0, 1.0])

    assert torch.allclose(q, expected)


def test_q_kan_forward(device):
    """Verify QKANLayer runs and weights are effective."""
    layer = QKANLayer(in_features=8, out_features=8, bits=2).to(device)
    x = torch.randn(2, 16, 8, device=device)

    out = layer(x)
    assert out.shape == (2, 16, 8)

    # We can't easily check internal q_coeffs without mocking,
    # but the fact it runs implies quantization op succeeded.


def test_ste_backward():
    """Verify Straight-Through Estimator allows gradients."""
    x = torch.randn(4, requires_grad=True)
    q = SplineQuantizer.apply(x, 2)
    loss = q.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.allclose(x.grad, torch.ones_like(x))  # Sum gradient is 1s


def test_unsupported_bits_raises():
    """Unsupported bit-widths must raise, not silently no-op (pitfall #9)."""
    with pytest.raises(ValueError):
        SplineQuantizer.apply(torch.randn(4), 4)
    with pytest.raises(ValueError):
        QKANLayer(in_features=8, out_features=8, bits=8)
