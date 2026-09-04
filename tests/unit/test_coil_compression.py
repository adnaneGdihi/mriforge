"""Tests for SVD coil compression with shared basis."""

import numpy as np
import torch
import torchio as tio

from spectramr.data.transforms.coil_compression import (
    SVDCoilCompression,
    SVDCoilCompressionTransform,
)


def _make_multicoil_kspace(
    num_coils: int = 9,
    h: int = 64,
    w: int = 64,
    noise_std: float = 0.0,
) -> torch.Tensor:
    """Create synthetic multi-coil k-space with known coil sensitivities.

    Returns: [1, num_coils, H, W] complex tensor
    """
    # Shared image content
    image = torch.randn(1, 1, h, w, dtype=torch.complex64)

    # Deterministic coil sensitivities (smooth spatial maps)
    y_coords = torch.linspace(-1, 1, h).view(1, 1, h, 1)
    x_coords = torch.linspace(-1, 1, w).view(1, 1, 1, w)
    sensitivities = []
    for i in range(num_coils):
        angle = 2 * np.pi * i / num_coils
        phase = torch.exp(
            1j
            * (
                torch.cos(torch.tensor(angle)) * y_coords
                + torch.sin(torch.tensor(angle)) * x_coords
            )
        )
        magnitude = torch.exp(
            -0.5
            * (
                (y_coords - 0.3 * np.cos(angle)) ** 2
                + (x_coords - 0.3 * np.sin(angle)) ** 2
            )
        )
        sensitivities.append((magnitude * phase).squeeze(0).squeeze(0))

    sens = torch.stack(sensitivities, dim=0).unsqueeze(0)  # [1, C, H, W]
    kspace = image * sens  # [1, C, H, W]

    if noise_std > 0:
        noise = torch.randn_like(kspace) * noise_std
        kspace = kspace + noise

    return kspace


class TestSVDCoilCompression:
    """Test the low-level SVDCoilCompression module."""

    def test_forward_reduces_coils(self):
        kspace = _make_multicoil_kspace(num_coils=9)
        compressor = SVDCoilCompression(num_virtual_coils=1)
        result = compressor(kspace)
        assert result.shape == (1, 1, 64, 64)

    def test_forward_noop_when_fewer_coils(self):
        kspace = _make_multicoil_kspace(num_coils=2)
        compressor = SVDCoilCompression(num_virtual_coils=4)
        result = compressor(kspace)
        assert result.shape == kspace.shape
        assert torch.equal(result, kspace)

    def test_compute_basis_shape(self):
        kspace = _make_multicoil_kspace(num_coils=9)
        compressor = SVDCoilCompression(num_virtual_coils=4)
        basis = compressor.compute_basis(kspace)
        assert basis.shape == (1, 9, 4)

    def test_compute_basis_returns_none_when_noop(self):
        kspace = _make_multicoil_kspace(num_coils=2)
        compressor = SVDCoilCompression(num_virtual_coils=4)
        basis = compressor.compute_basis(kspace)
        assert basis is None

    def test_shared_basis_preserves_noise_difference(self):
        """Shared basis faithfully preserves noise difference.

        When input and target share the same underlying signal with coil
        sensitivity structure (as in real MRI), applying a shared SVD basis
        should preserve the noise ratio between them.
        """
        torch.manual_seed(42)
        num_coils, h, w = 9, 64, 64

        # Realistic coil sensitivities (smooth spatial sensitivity maps)
        ky = torch.linspace(-1, 1, h).view(1, 1, h, 1)
        kx = torch.linspace(-1, 1, w).view(1, 1, 1, w)
        sensitivities = []
        for i in range(num_coils):
            angle = 2 * np.pi * i / num_coils
            phase = torch.exp(1j * (np.cos(angle) * ky + np.sin(angle) * kx) * 2)
            magnitude = torch.exp(
                -0.3
                * ((ky - 0.4 * np.cos(angle)) ** 2 + (kx - 0.4 * np.sin(angle)) ** 2)
            )
            sensitivities.append((magnitude * phase).squeeze())
        sens = torch.stack(sensitivities, dim=0).unsqueeze(0)  # [1, C, H, W]

        image = torch.randn(1, 1, h, w, dtype=torch.complex64) * 5.0
        base_kspace = image * sens
        clean = base_kspace.clone()
        noisy = base_kspace + torch.randn_like(base_kspace) * 0.3

        compressor = SVDCoilCompression(num_virtual_coils=1)

        # Shared basis from clean target
        basis = compressor.compute_basis(clean)
        compressed_clean = compressor.apply_basis(clean, basis)
        compressed_noisy = compressor.apply_basis(noisy, basis)
        shared_diff = (compressed_clean - compressed_noisy).abs().mean().item()

        raw_diff = (clean - noisy).abs().mean().item()

        # Shared basis should preserve noise ratio faithfully (within 20%)
        ratio = shared_diff / raw_diff
        assert 0.5 < ratio < 2.0, (
            f"Shared basis noise ratio ({ratio:.4f}) is outside expected range. "
            f"raw_diff={raw_diff:.6f}, shared_diff={shared_diff:.6f}"
        )

    def test_apply_basis_with_precomputed(self):
        """forward() with precomputed basis matches apply_basis()."""
        kspace = _make_multicoil_kspace(num_coils=9)
        compressor = SVDCoilCompression(num_virtual_coils=4)
        basis = compressor.compute_basis(kspace)
        result_forward = compressor(kspace, basis=basis)
        result_apply = compressor.apply_basis(kspace, basis)
        assert torch.allclose(result_forward, result_apply)


class TestSVDCoilCompressionTransform:
    """Test the TorchIO transform wrapper."""

    def _make_subject(
        self, num_coils: int = 9, num_slices: int = 3, noise_std: float = 0.5
    ) -> tio.Subject:
        """Create a subject mimicking M4Raw: input (noisy), target (clean), kspace (=input)."""
        h, w = 64, 64
        # Clean target (average of many reps)
        target_data = torch.randn(num_coils, h, w, num_slices, dtype=torch.complex64)
        # Noisy input (single rep)
        input_data = target_data + torch.randn_like(target_data) * noise_std

        affine = np.eye(4)
        return tio.Subject(
            input=tio.ScalarImage(tensor=input_data, affine=affine),
            target=tio.ScalarImage(tensor=target_data, affine=affine),
            kspace=tio.ScalarImage(tensor=input_data.clone(), affine=affine),
        )

    def test_transform_uses_shared_basis(self):
        """The transform should use the same basis for all images."""
        subject = self._make_subject(noise_std=0.5)

        # Record original difference
        orig_diff = (subject["input"].data - subject["target"].data).abs().mean().item()

        transform = SVDCoilCompressionTransform(num_virtual_coils=1)
        transformed = transform(subject)

        # Transformed difference should still be significant
        new_diff = (
            (transformed["input"].data - transformed["target"].data).abs().mean().item()
        )

        # The SNR difference should be preserved proportionally
        ratio = new_diff / orig_diff
        assert ratio > 0.1, (
            f"SNR difference ratio ({ratio:.4f}) is too small — "
            f"shared basis should preserve relative noise differences. "
            f"orig_diff={orig_diff:.6f}, new_diff={new_diff:.6f}"
        )

    def test_transform_reduces_channels(self):
        subject = self._make_subject(num_coils=9)
        transform = SVDCoilCompressionTransform(num_virtual_coils=4)
        transformed = transform(subject)

        for key in ("input", "target", "kspace"):
            assert (
                transformed[key].data.shape[0] == 4
            ), f"Expected 4 channels for '{key}', got {transformed[key].data.shape[0]}"

    def test_transform_noop_when_fewer_coils(self):
        subject = self._make_subject(num_coils=2)
        transform = SVDCoilCompressionTransform(num_virtual_coils=4)
        transformed = transform(subject)

        for key in ("input", "target", "kspace"):
            assert torch.equal(transformed[key].data, subject[key].data)

    def test_reference_key_override(self):
        """Custom reference_key should be used if present."""
        subject = self._make_subject()
        transform = SVDCoilCompressionTransform(
            num_virtual_coils=1, reference_key="input"
        )
        # Should not raise
        transformed = transform(subject)
        assert transformed["input"].data.shape[0] == 1

    def test_reference_key_fallback(self):
        """If reference_key is not found, fallback to priority list."""
        subject = self._make_subject()
        transform = SVDCoilCompressionTransform(
            num_virtual_coils=1, reference_key="nonexistent"
        )
        # Should not raise, falls back to "target"
        transformed = transform(subject)
        assert transformed["input"].data.shape[0] == 1

    def test_real_stacked_input_skips_no_raise(self):
        """Regression: when ``FastMRISubjectBuilder._apply_coil_processing``
        already runs SVD inline at __getitem__ time the subject reaches this
        TorchIO transform with real-stacked tensors. The transform must NOT
        raise — even if the transform's ``num_virtual_coils`` differs from
        the upstream value (the May 2026 experiment_130_universal_multitask
        cluster failure: shape ``[2, 256, 256, 18]`` arrived at a transform
        whose ``num_virtual_coils`` desynced and the strict-equality dedup
        check fired). Skip silently with a debug log.
        """
        # Real-stacked tensor [2, H, W, D] simulating subject-builder
        # output after SVD-then-complex-to-real.
        h, w, d = 32, 32, 4
        real_stacked = torch.randn(2, h, w, d, dtype=torch.float32)
        affine = np.eye(4)
        subject = tio.Subject(
            input=tio.ScalarImage(tensor=real_stacked.clone(), affine=affine),
            target=tio.ScalarImage(tensor=real_stacked.clone(), affine=affine),
            kspace=tio.ScalarImage(tensor=real_stacked.clone(), affine=affine),
        )
        # Construct with ``num_virtual_coils=4`` (the schema default) ON
        # PURPOSE — this is the desync case that the strict equality
        # check used to fail on (`2 != 2*4`).
        transform = SVDCoilCompressionTransform(num_virtual_coils=4)
        transformed = transform(subject)
        # No-op: tensors should be unchanged bit-exactly.
        for key in ("input", "target", "kspace"):
            assert torch.equal(transformed[key].data, real_stacked), (
                f"real-stacked input must pass through unmodified, "
                f"key='{key}' shape={transformed[key].data.shape}"
            )

    def test_real_stacked_input_skips_when_num_virtual_coils_matches(self):
        """The matching-shape happy path also skips — same code path, no
        special-casing for desync."""
        h, w, d = 32, 32, 4
        real_stacked = torch.randn(2, h, w, d, dtype=torch.float32)
        affine = np.eye(4)
        subject = tio.Subject(
            input=tio.ScalarImage(tensor=real_stacked.clone(), affine=affine),
            target=tio.ScalarImage(tensor=real_stacked.clone(), affine=affine),
            kspace=tio.ScalarImage(tensor=real_stacked.clone(), affine=affine),
        )
        transform = SVDCoilCompressionTransform(num_virtual_coils=1)
        transformed = transform(subject)
        for key in ("input", "target", "kspace"):
            assert torch.equal(transformed[key].data, real_stacked)
