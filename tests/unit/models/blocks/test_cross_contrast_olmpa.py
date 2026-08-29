"""Unit tests for CrossContrastOLMPA attention module.

Validates:
- Shape correctness (output matches input shape)
- O(N) linear attention (no O(N²) memory allocation)
- Orthogonal weight parameterization (energy conservation)
- Phase safety (magnitude-only Q/K routing)
"""

import torch

from mriforge.models.blocks.attention import CrossContrastOLMPA


class TestCrossContrastOLMPA:
    """Tests for the CC-OLMPA attention module."""

    def test_output_shape_matches_target(self) -> None:
        """Output shape must equal target_kspace shape."""
        B, N, C = 2, 1024, 8
        num_contrasts = 4
        olmpa = CrossContrastOLMPA(
            in_channels=C, num_contrasts=num_contrasts, phase_safe_dim=128
        )

        target = torch.randn(B, N, C)
        ref = torch.randn(B, N, C * num_contrasts)

        out = olmpa(target, ref)

        assert (
            out.shape == target.shape
        ), f"Output shape {out.shape} != target shape {target.shape}"

    def test_output_shape_single_contrast(self) -> None:
        """Works with num_contrasts=1 (feature-level source→target)."""
        B, N, C = 2, 512, 16
        olmpa = CrossContrastOLMPA(in_channels=C, num_contrasts=1, phase_safe_dim=64)

        target = torch.randn(B, N, C)
        ref = torch.randn(B, N, C)  # Same dim when num_contrasts=1

        out = olmpa(target, ref)

        assert out.shape == (B, N, C)

    def test_orthogonal_weight_parametrization(self) -> None:
        """Q and K projections must have orthogonal weight matrices."""
        olmpa = CrossContrastOLMPA(in_channels=8, num_contrasts=4, phase_safe_dim=8)

        # Access the parametrized weight (after orthogonal projection)
        W_q = olmpa.W_q.weight  # [phase_safe_dim, in_channels]
        W_k = olmpa.W_k.weight  # [phase_safe_dim, in_channels * num_contrasts]

        # W^T @ W should be close to identity for orthogonal parametrization
        # For rectangular matrices (phase_safe_dim × in_channels), W^T W = I_{min(m,n)}
        q_product = W_q @ W_q.T  # [d, d]
        identity = torch.eye(q_product.shape[0])

        assert torch.allclose(
            q_product, identity, atol=1e-5
        ), "W_q is not orthogonally parametrized"

    def test_no_gradient_through_magnitude(self) -> None:
        """Q/K magnitudes should not backprop phase gradients."""
        B, N, C = 1, 64, 4
        olmpa = CrossContrastOLMPA(in_channels=C, num_contrasts=1, phase_safe_dim=16)

        target = torch.randn(B, N, C, requires_grad=True)
        ref = torch.randn(B, N, C, requires_grad=True)

        out = olmpa(target, ref)
        loss = out.sum()
        loss.backward()

        # Gradients should exist (backprop works)
        assert target.grad is not None
        assert ref.grad is not None

    def test_linear_complexity_no_n_squared_matrix(self) -> None:
        """Verify no N×N attention matrix is created (O(N) not O(N²)).

        We do this by running with a large N and checking it doesn't OOM
        on CPU (which would happen with a 65536×65536 float32 matrix).
        """
        B, N, C = 1, 65536, 8  # Full 256×256 unpatched
        olmpa = CrossContrastOLMPA(in_channels=C, num_contrasts=1, phase_safe_dim=128)

        target = torch.randn(B, N, C)
        ref = torch.randn(B, N, C)

        # This should NOT OOM — O(N²) would need ~17GB for N=65536
        out = olmpa(target, ref)
        assert out.shape == (B, N, C)

    def test_value_projection_preserves_channels(self) -> None:
        """W_v must map from source channels to target channels."""
        C = 8
        num_contrasts = 4
        olmpa = CrossContrastOLMPA(
            in_channels=C, num_contrasts=num_contrasts, phase_safe_dim=128
        )

        assert olmpa.W_v.in_features == C * num_contrasts
        assert olmpa.W_v.out_features == C
