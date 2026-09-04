"""
Unit tests for shared tensor operations.
"""

import pytest
import torch

try:
    from spectramr.shared.ops import complex_abs
except ImportError:
    complex_abs = None


# @# pytest.mark.unit
class TestSharedOps:
    def test_complex_abs(self):
        if complex_abs is None:
            pytest.skip("complex_abs not available")

        # Complex tensor: [B, 2, H, W] or [B, H, W, 2]
        x = torch.randn(1, 2, 32, 32)
        mag = complex_abs(x)
        assert mag.shape == (1, 1, 32, 32)
        assert torch.all(mag >= 0)
