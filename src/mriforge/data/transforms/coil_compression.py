"""Coil compression via SVD - preserves linearity.

This module implements hardware-level coil compression using Eigenvalue Decomposition
on the Coil Covariance Matrix, which is mathematically equivalent to SVD but 256x faster.
"""

import logging

import torch
import torch.nn as nn
import torchio as tio

logger = logging.getLogger(__name__)


class SVDCoilCompression(nn.Module):
    """Linear coil compression via Eigenvalue Decomposition on Covariance Matrix.

    References:
        - Huang et al., "Software channel compression for massive array systems,"
          Magnetic Resonance Imaging, 2008.
    """

    def __init__(self, num_virtual_coils: int = 4, calibration_lines: int | None = None):
        """__init__.

        Args:
            num_virtual_coils (int): Target virtual-coil count.
            calibration_lines (int | None): If set, compute the compression basis
                from a centered ``calibration_lines``-row band along H (the
                low-frequency ACS region) instead of the full FoV. ``None``
                (default) uses the full FoV — bit-identical to the prior behavior.
        """
        super().__init__()
        self.num_virtual_coils = num_virtual_coils
        self.calibration_lines = calibration_lines

    def compute_basis(self, kspace: torch.Tensor) -> torch.Tensor:
        """Compute coil compression basis from k-space data.

        Args:
            kspace: [B, C, H, W] complex-valued k-space

        Returns:
            V_top: [B, C, num_virtual_coils] top eigenvectors (projection basis)
        """
        if kspace.dim() != 4:
            raise ValueError(f"Expected 4D k-space [B, C, H, W], got {kspace.shape}")

        B, C, H, W = kspace.shape

        if self.num_virtual_coils >= C:
            return None

        # Restrict the basis estimate to a centered calibration band along H (the
        # low-frequency ACS region) when configured. None → full FoV (parity).
        src = kspace
        if self.calibration_lines is not None and self.calibration_lines < H:
            c0 = H // 2 - self.calibration_lines // 2
            src = kspace[:, :, c0 : c0 + self.calibration_lines, :]
        # Crop affects only H; width W is unchanged. N = (cropped H) * W.
        n_samples = src.shape[2] * W

        k_mat = src.reshape(B, C, -1)  # [B, C, Hcal*W]

        # Covariance: R = (1/N) * X * X^H where N = Hcal*W
        cov = torch.bmm(k_mat, k_mat.conj().transpose(1, 2)) / n_samples

        # Eigenvalue decomposition on C×C matrix
        # torch.linalg.eigh returns eigenvalues in ASCENDING order
        _eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        # Select top N eigenvectors (largest eigenvalues = last columns)
        V_top = eigenvectors[:, :, -self.num_virtual_coils :]
        return V_top

    def apply_basis(self, kspace: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        """Project k-space onto a precomputed basis.

        Args:
            kspace: [B, C, H, W] complex-valued k-space
            basis: [B, C, num_virtual_coils] eigenvector basis

        Returns:
            k_virtual: [B, num_virtual_coils, H, W] compressed k-space
        """
        B, C, H, W = kspace.shape
        k_mat = kspace.reshape(B, C, -1)

        k_virtual = torch.bmm(
            basis.conj().transpose(1, 2),  # [B, num_virtual_coils, C]
            k_mat,  # [B, C, H*W]
        )

        return k_virtual.reshape(B, self.num_virtual_coils, H, W)

    def forward(self, kspace: torch.Tensor, basis: torch.Tensor | None = None) -> torch.Tensor:
        """Compress coils via eigendecomposition of covariance matrix.

        Args:
            kspace: [B, C, H, W] complex-valued k-space
            basis: Optional precomputed basis [B, C, num_virtual_coils].
                   If None, computes basis from input kspace.

        Returns:
            k_virtual: [B, num_virtual_coils, H, W] compressed k-space

        forward method for SVDCoilCompression.

        Executes PyTorch tensor operations.

        Args:
            kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            basis (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if kspace.dim() != 4:
            raise ValueError(f"Expected 4D k-space [B, C, H, W], got {kspace.shape}")

        B, C, H, W = kspace.shape

        if self.num_virtual_coils >= C:
            return kspace

        if basis is None:
            basis = self.compute_basis(kspace)

        return self.apply_basis(kspace, basis)


class SVDCoilCompressionTransform(tio.Transform):
    """TorchIO transform wrapper for SVDCoilCompression.

    Computes the SVD basis from a single reference image and applies it
    consistently to ALL images in the subject. This preserves relative
    noise differences between input and target — critical for learning
    tasks where the target has higher SNR than the input (e.g., averaged
    multi-repetition MRI).

    Without shared basis, independent SVD per image projects each onto its
    own rank-1 subspace, effectively denoising the input and destroying the
    SNR learning signal.
    """

    REFERENCE_KEY_PRIORITY = ("target", "kspace", "input")

    def __init__(
        self,
        num_virtual_coils: int = 4,
        reference_key: str | None = None,
        calibration_lines: int | None = None,
        **kwargs,
    ):
        """__init__.

        Args:
            num_virtual_coils (int): Description.
            reference_key (str | None): Description.
            calibration_lines (int | None): Central k-space rows for the SVD
                basis estimate (None = full FoV; parity with prior behavior).
        """
        super().__init__(**kwargs)
        self.num_virtual_coils = num_virtual_coils
        self.reference_key = reference_key
        self.calibration_lines = calibration_lines
        self.compressor = SVDCoilCompression(
            num_virtual_coils=num_virtual_coils, calibration_lines=calibration_lines
        )

    def _resolve_reference_key(self, subject: tio.Subject) -> str:
        """Find the reference image key for computing SVD basis."""
        if self.reference_key is not None:
            if self.reference_key in subject.get_images_names():
                return self.reference_key
            logger.warning(
                f"Configured reference_key '{self.reference_key}' not found "
                f"in subject. Falling back to priority search."
            )

        image_names = set(subject.get_images_names())
        for key in self.REFERENCE_KEY_PRIORITY:
            if key in image_names:
                return key

        # Fallback to first available
        return next(iter(image_names))

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        # Determine reference image and compute shared basis
        """apply_transform.

        Args:
            subject (tio.Subject): Description.
        Returns:
            tio.Subject: Description.
        """
        ref_key = self._resolve_reference_key(subject)
        ref_data = subject[ref_key].data

        if ref_data.dim() != 4:
            logger.warning(
                f"SVDCoilCompressionTransform expected 4D TorchIO data "
                f"[C, H, W, D], got {ref_data.shape}"
            )
            return subject

        # SVD coil compression operates on COMPLEX k-space — see Buehrer 2007
        # (MRM 57:1131-1139). The compression matrix is complex-valued and the
        # virtual coil is a complex-weighted sum that includes phase rotation.
        # Applying SVD per-component to real-stacked channels (R1, I1, R2, I2,
        # ...) destroys phase coherence and the SENSE-adjoint loss + dual-domain
        # complex-spatial-gradient loss produce incorrect gradients.
        if not torch.is_complex(ref_data):
            # SVD coil compression cannot operate on real-stacked tensors —
            # the compression matrix is intrinsically complex (Buehrer 2007).
            # In practice, real-stacked data reaching this transform always
            # means an upstream stage already adapted the tensor:
            #   * ``FastMRISubjectBuilder._apply_coil_processing`` runs SVD
            #     inline at dataset ``__getitem__`` time and returns a
            #     ``(2*num_virtual_coils, H, W, D)`` real tensor.
            #   * ``M4RawRepetitionDataset`` interleaves R/I per coil into
            #     ``(2*physical_coils, H, W, D)`` without compression.
            #   * Some external preprocessing pipelines stash R/I as channels
            #     before the TorchIO transform list runs.
            # In all three cases the right move is the same: skip this
            # transform with a *visible* debug log, since an early raise here
            # tears down legitimate FastMRI runs whenever the
            # subject-builder ``num_virtual_coils`` and the transform's
            # ``num_virtual_coils`` desync (e.g., the May 2026
            # experiment_130_universal_multitask cluster failure where
            # ``shape[0]=2`` but the transform's ``num_virtual_coils`` was
            # not 1). The earlier strict raise was meant to catch silent
            # fallbacks, but the duplicate-SVD case is a *legitimate*
            # upstream-compression case, not a misconfiguration. The audit
            # check ``svd_compression_phase_safety`` (config-load time) is
            # the right place to flag a true "SVD-on-real-stacked" pipeline
            # bug — runtime-skipping here is the conservative default.
            logger.info(
                "[SVD] Input already real-stacked (shape=%s, dtype=%s) — "
                "assuming subject_builder performed coil compression / R/I "
                "interleaving upstream. Skipping duplicate transform "
                "(num_virtual_coils=%d).",
                tuple(ref_data.shape),
                ref_data.dtype,
                self.num_virtual_coils,
            )
            return subject

        # Permute D to B: [C, H, W, D] -> [D, C, H, W]
        ref_kspace = ref_data.permute(3, 0, 1, 2)
        basis = self.compressor.compute_basis(ref_kspace)

        if basis is None:
            # Already fewer coils than requested — nothing to compress
            return subject

        logger.debug(
            f"[SVD] Computed shared basis from '{ref_key}' "
            f"(shape {ref_data.shape}), "
            f"projecting {ref_data.shape[0]} coils -> {self.num_virtual_coils}"
        )

        # Apply shared basis to ALL images
        for image_name in subject.get_images_names():
            image = subject[image_name]
            data = image.data

            if data.dim() != 4:
                logger.warning(
                    f"SVDCoilCompressionTransform: skipping '{image_name}' "
                    f"with unexpected shape {data.shape}"
                )
                continue

            kspace = data.permute(3, 0, 1, 2)
            compressed_kspace = self.compressor.apply_basis(kspace, basis)
            compressed_data = compressed_kspace.permute(1, 2, 3, 0)
            image.set_data(compressed_data)

        return subject
