"""Unit tests for KinematicForwardOperator."""

from __future__ import annotations

import pytest
import torch

HAS_NUFFT = False
try:
    import torchkbnufft  # noqa: F401

    HAS_NUFFT = True
except ImportError:
    pass


@pytest.mark.skipif(not HAS_NUFFT, reason="torchkbnufft not installed")
class TestKinematicForwardOperator:
    """Tests for differentiable rigid-body k-space motion operator."""

    @pytest.fixture
    def op(self):
        from spectramr.infrastructure.physics.kinematic_forward import (
            KinematicForwardOperator,
        )

        return KinematicForwardOperator(im_size=(32, 32))

    def test_init(self, op) -> None:
        """KinematicForwardOperator initializes with im_size."""
        assert op.im_size == (32, 32)
        assert op.num_readout_lines == 32

    def test_base_traj_shape(self, op) -> None:
        """Base trajectory should be [2, num_readout * num_freq]."""
        assert op.base_traj.shape[0] == 2
        assert op.base_traj.shape[1] == 32 * 32

    def test_forward_output_shape(self, op) -> None:
        """Output should be corrupted k-space."""

        image = torch.randn(1, 1, 32, 32, dtype=torch.complex64)
        theta = torch.zeros(1, 3, 32)  # zero motion
        out = op(image, theta)
        assert out.numel() > 0, "Output is empty"

    def test_random_motion_bounded(self, op) -> None:
        """Random motion parameters should be bounded."""
        theta = torch.randn(1, 3, 32)
        # dx, dy (translations) and dtheta (rotation) are just tensors
        assert theta.shape == (1, 3, 32)

    def test_zero_motion(self) -> None:
        """Zero motion tensor should be all zeros."""
        theta = torch.zeros(1, 3, 32)
        assert theta.abs().max() == 0.0

    def test_generate_random_motion_scoped_generator_reproducible(self, op) -> None:
        """2026-07-01: the ``generator`` param threads ALL draws — same seed →
        identical trajectory, and the global CPU stream is untouched (replaces
        the global manual_seed + save/restore bracket at strategy call sites)."""
        gen = torch.Generator(device="cpu")

        gen.manual_seed(123)
        a = op.generate_random_motion(
            batch_size=2, motion_type="smooth", generator=gen
        )
        gen.manual_seed(123)
        b = op.generate_random_motion(
            batch_size=2, motion_type="smooth", generator=gen
        )
        assert torch.equal(a, b)
        assert a.shape == (2, 3, op.num_readout_lines)

        # Global CPU stream untouched by generator-scoped draws.
        torch.manual_seed(99)
        expected_next = torch.rand(4)
        torch.manual_seed(99)
        gen.manual_seed(123)
        op.generate_random_motion(batch_size=2, motion_type="smooth", generator=gen)
        assert torch.equal(expected_next, torch.rand(4))

    def test_generate_random_motion_sudden_honours_generator(self, op) -> None:
        """The 'sudden' branch (randint onset/duration + per-sample rand) also
        draws exclusively from the scoped generator."""
        gen = torch.Generator(device="cpu")
        gen.manual_seed(7)
        a = op.generate_random_motion(
            batch_size=3, motion_type="sudden", generator=gen
        )
        gen.manual_seed(7)
        b = op.generate_random_motion(
            batch_size=3, motion_type="sudden", generator=gen
        )
        assert torch.equal(a, b)
        assert a.shape == (3, 3, op.num_readout_lines)


class TestKinematicForwardOperatorNoNUFFT:
    """Import guard tests when torchkbnufft is not available."""

    @pytest.mark.skipif(HAS_NUFFT, reason="torchkbnufft IS installed")
    def test_import_error_without_nufft(self) -> None:
        """KinematicForwardOperator should raise ImportError without torchkbnufft."""
        from spectramr.infrastructure.physics.kinematic_forward import (
            KinematicForwardOperator,
        )

        with pytest.raises(ImportError, match="torchkbnufft"):
            KinematicForwardOperator(im_size=(32, 32))


@pytest.mark.skipif(not HAS_NUFFT, reason="torchkbnufft not installed")
def test_translation_phase_recovers_shift() -> None:
    """A pure translation of ``s`` pixels is recovered as ``s`` (2026-07 fix).

    The Fourier-shift phase previously carried a spurious 2π/W factor and
    under-scaled every translation by ~W/(2π): a 5-px shift was recovered as
    ~1 px. With the trajectory in torchkbnufft's radian convention the phase is
    exp(-i·ω·d_pixels).
    """
    import torchkbnufft as tkbn

    from spectramr.infrastructure.physics.kinematic_forward import (
        KinematicForwardOperator,
    )

    n = 32
    op = KinematicForwardOperator(im_size=(n, n))
    op.eval()
    img = torch.zeros(1, 1, n, n, dtype=torch.complex64)
    img[0, 0, n // 2, n // 2] = 1.0
    shift = 5.0
    theta = torch.zeros(1, 3, n)
    theta[:, 0, :] = shift  # dx = 5 px on every readout line, no rotation
    with torch.no_grad():
        k = op(img, theta)
        adj = tkbn.KbNufftAdjoint(im_size=(n, n))
        adj.eval()
        rec = adj(k.reshape(1, 1, -1), op.base_traj.unsqueeze(0))[0, 0].abs()
    peak = int(torch.argmax(rec))
    row, col = peak // n - n // 2, peak % n - n // 2
    assert (row, col) == (5, 0)
