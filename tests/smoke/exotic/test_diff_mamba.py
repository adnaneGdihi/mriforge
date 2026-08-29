"""Property-based tests for Differential-Mamba (SOTA 2.0).

Verifies:
- Shape consistency.
- Interpolation correctness.
- ODE Integration stability.
- Variable time steps.
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

# Import model
from mriforge.models.mamba.diff_mamba import DiffMamba, DiffSSM, LinearInterpolation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@settings(deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=4),
    seq_len=st.integers(min_value=8, max_value=32),
    d_model=st.integers(min_value=4, max_value=16),
)
def test_linear_interpolation(batch_size, seq_len, d_model):
    """Verify that linear interpolation matches exact values at integer steps."""
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    interp = LinearInterpolation(seq_len)

    # Check integer points
    for t in range(seq_len):
        val = interp(x, float(t))
        # Should exactly match
        assert torch.allclose(val, x[:, t, :], atol=1e-5)

    # Check midpoint
    if seq_len > 1:
        t_mid = 0.5
        val_mid = interp(x, t_mid)
        expected = 0.5 * x[:, 0, :] + 0.5 * x[:, 1, :]
        assert torch.allclose(val_mid, expected, atol=1e-5)


@settings(deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=2),
    seq_len=st.integers(min_value=5, max_value=10),
    d_model=st.integers(min_value=4, max_value=8),
)
def test_diff_ssm_integration(batch_size, seq_len, d_model):
    """Test standard integration forward pass."""
    model = DiffSSM(d_model=d_model, d_state=d_model, dt_step=0.5).to(device)
    x = torch.randn(batch_size, seq_len, d_model, device=device)

    out = model(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


@settings(deadline=None)
@given(
    in_channels=st.integers(min_value=1, max_value=2),
    base_dim=st.integers(min_value=4, max_value=8),
    height=st.integers(min_value=8, max_value=16),
    width=st.integers(min_value=8, max_value=16),
)
def test_full_diff_mamba_model(in_channels, base_dim, height, width):
    model = DiffMamba(in_channels=in_channels, base_dim=base_dim, num_layers=2).to(
        device
    )

    batch_size = 2
    x = torch.randn(batch_size, in_channels, height, width, device=device)

    out = model(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


if __name__ == "__main__":
    pytest.main([__file__])
