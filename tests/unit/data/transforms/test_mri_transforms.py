"""Tests for ``spectramr.data.transforms.mri_transforms.ComplexNormalization``.

Regression coverage for TM-003: the GPU-syncing sanity-check block (which
materialised ``magnitude.abs().max()`` into a Python bool / formatted string
inside ``__call__``) was removed. These tests pin the surviving contract:

- the transform does not change the input device (no host round-trip),
- it scales the magnitude down (robust-percentile normalisation), and
- it preserves the complex phase exactly (division by a positive real scalar).
"""

from __future__ import annotations

import torch

from spectramr.data.transforms.mri_transforms import ComplexNormalization


class TestComplexNormalization:
    def test_no_gpu_sync_cpu_tensor(self) -> None:
        """Normalising a CPU tensor stays on CPU and reduces magnitude.

        The removed sanity block forced a device sync via ``if max_mag > 100``
        and ``f'{max_mag:.1f}'``. After removal the op is sync-free; we assert
        the output device matches and the peak magnitude shrinks for an
        explosive (0-255-scale) input.
        """
        torch.manual_seed(0)
        # Real/imag stacked [2, H, W] with raw 0-255-style magnitudes.
        data = torch.rand(2, 16, 16) * 255.0
        norm = ComplexNormalization(percentile=0.99)

        out = norm(data)

        assert out.device == data.device
        assert out.shape == data.shape
        # Robust-percentile scaling shrinks the peak magnitude.
        mag_in = (data[0:1] ** 2 + data[1:2] ** 2).sqrt().max()
        mag_out = (out[0:1] ** 2 + out[1:2] ** 2).sqrt().max()
        assert mag_out < mag_in

    def test_preserves_phase(self) -> None:
        """Dividing by a positive real scalar leaves the complex phase intact."""
        torch.manual_seed(1)
        data = torch.randn(1, 16, 16, dtype=torch.complex64) * 200.0
        norm = ComplexNormalization(percentile=0.99)

        out = norm(data)

        assert torch.is_complex(out)
        # Phase angle must be unchanged everywhere the magnitude is non-trivial.
        phase_in = torch.angle(data)
        phase_out = torch.angle(out)
        torch.testing.assert_close(phase_out, phase_in, rtol=1e-5, atol=1e-5)


# ── WS3b: flip/rotation draw from torch's (per-worker seeded) RNG ──────────────

from spectramr.data.transforms.mri_transforms import (  # noqa: E402
    RandomMRIFlip,
    RandomMRIRotation,
)


class TestRandomAugsUseTorchRNG:
    """RandomMRIFlip/Rotation previously used Python's ``random`` module, which
    PyTorch does NOT re-seed per DataLoader worker (only torch's own generator).
    They now draw from ``torch.rand`` so a fixed torch seed reproduces the exact
    augmentation sequence — reproducible and per-worker decorrelated."""

    def test_flip_sequence_is_reproducible_under_torch_seed(self) -> None:
        data = torch.arange(16.0).reshape(1, 1, 4, 4)
        flip = RandomMRIFlip(prob=0.5)

        torch.manual_seed(1234)
        seq_a = [torch.equal(flip(data), torch.flip(data, dims=[-1])) for _ in range(20)]
        torch.manual_seed(1234)
        seq_b = [torch.equal(flip(data), torch.flip(data, dims=[-1])) for _ in range(20)]
        assert seq_a == seq_b
        # prob=0.5 over 20 draws must actually vary (not a stuck constant).
        assert any(seq_a) and not all(seq_a)

    def test_rotation_sequence_is_reproducible_under_torch_seed(self) -> None:
        data = torch.arange(16.0).reshape(1, 1, 4, 4)
        rot = RandomMRIRotation(prob=0.5)

        torch.manual_seed(99)
        seq_a = [torch.equal(rot(data), torch.rot90(data, 1, [-2, -1])) for _ in range(20)]
        torch.manual_seed(99)
        seq_b = [torch.equal(rot(data), torch.rot90(data, 1, [-2, -1])) for _ in range(20)]
        assert seq_a == seq_b

    def test_prob_zero_never_flips(self) -> None:
        data = torch.randn(1, 1, 4, 4)
        flip = RandomMRIFlip(prob=0.0)
        for _ in range(10):
            assert torch.equal(flip(data), data)
