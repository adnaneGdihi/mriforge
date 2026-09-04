"""PhysicsEngine - Unified Facade for MRI Physics Operations.

This module provides a high-level API for all physics-related operations in
MRI reconstruction, following the Facade pattern to simplify access to:
- FFT operations (forward/inverse, 2D/3D)
- Data consistency enforcement
- SENSE forward/adjoint operators
- Sampling and trajectory operators

Example:
    >>> from spectramr.infrastructure.physics import PhysicsEngine
    >>> engine = PhysicsEngine(device="cuda")
    >>>
    >>> # Image to k-space
    >>> kspace = engine.image_to_kspace(image)
    >>>
    >>> # Apply data consistency
    >>> dc_image = engine.enforce_data_consistency(prediction, measured_kspace, mask)
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PhysicsEngine:
    """Unified facade for MRI physics operations.

    Consolidates FFT, data consistency, and forward/adjoint operators
    into a single, easy-to-use interface.

    Attributes:
        device: Target device for operations
        data_consistency: Default DC module for reconstruction
    """

    __slots__ = ("_config", "_dc_module", "_device")

    def __init__(
        self,
        device: str | torch.device = "cpu",
        dc_type: Literal["soft", "hard", "adaptive"] = "soft",
        dc_weight: float = 1.0,
    ):
        """Initialize physics engine.

        Args:
            device: Target device (cpu/cuda)
            dc_type: Type of data consistency ('soft', 'hard', 'adaptive')
            dc_weight: Weight for soft data consistency
        """
        self._device = torch.device(device)
        self._config = {"dc_type": dc_type, "dc_weight": dc_weight}
        self._dc_module: nn.Module | None = None

    @property
    def device(self) -> torch.device:
        """Get current device."""
        return self._device

    def to(self, device: str | torch.device) -> PhysicsEngine:
        """Move engine to specified device."""
        self._device = torch.device(device)
        if self._dc_module is not None:
            self._dc_module = self._dc_module.to(self._device)
        return self

    # =========================================================================
    # FFT Operations
    # =========================================================================

    def image_to_kspace(
        self,
        image: torch.Tensor,
        mask: torch.Tensor | None = None,
        centered: bool = True,
    ) -> torch.Tensor:
        """Transform image domain to k-space (frequency domain).

        Args:
            image: Complex or real image tensor (..., H, W)
            mask: Optional sampling mask
            centered: Must be True; centered FFT is the physics SSOT (raises if False)

        Returns:
            K-space tensor (..., H, W)
        """
        from .fft_ops import fft2c, fft2c_masked

        if not centered:
            raise ValueError(
                "PhysicsEngine.image_to_kspace only supports centered FFT "
                "(fft2c is the physics SSOT; centering + norm='ortho' are "
                "correctness-critical). centered=False is not implemented."
            )
        if mask is not None:
            return fft2c_masked(image, mask)
        return fft2c(image)

    def kspace_to_image(
        self,
        kspace: torch.Tensor,
        mask: torch.Tensor | None = None,
        centered: bool = True,
    ) -> torch.Tensor:
        """Transform k-space to image domain.

        Args:
            kspace: Complex k-space tensor (..., H, W)
            mask: Optional sampling mask
            centered: Must be True; centered IFFT is the physics SSOT (raises if False)

        Returns:
            Image tensor (..., H, W)
        """
        from .fft_ops import ifft2c, ifft2c_masked

        if not centered:
            raise ValueError(
                "PhysicsEngine.kspace_to_image only supports centered IFFT "
                "(ifft2c is the physics SSOT; centering + norm='ortho' are "
                "correctness-critical). centered=False is not implemented."
            )
        if mask is not None:
            return ifft2c_masked(kspace, mask)
        return ifft2c(kspace)

    def apply_mask(
        self,
        kspace: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply sampling mask to k-space data.

        Args:
            kspace: K-space tensor
            mask: Sampling mask (1=sampled, 0=unsampled)

        Returns:
            Masked k-space
        """
        from .fft_ops import apply_mask

        return apply_mask(kspace, mask)

    # =========================================================================
    # SENSE Operations (Multi-coil)
    # =========================================================================

    def sense_forward(
        self,
        image: torch.Tensor,
        sensitivity_maps: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """SENSE forward operator: image → multi-coil k-space.

        Args:
            image: Image tensor (B, 1, H, W)
            sensitivity_maps: Sensitivity maps (B, Coils, H, W) or None for single-coil
            mask: Optional sampling mask

        Returns:
            Multi-coil k-space (B, Coils, H, W)
        """
        from .fft_ops import sense_forward

        return sense_forward(image, smaps=sensitivity_maps, mask=mask)

    def sense_adjoint(
        self,
        kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        combine_rss: bool = False,
    ) -> torch.Tensor:
        """SENSE adjoint operator: multi-coil k-space → image.

        Args:
            kspace: Multi-coil k-space (B, Coils, H, W)
            sensitivity_maps: Sensitivity maps (B, Coils, H, W) or None
            mask: Optional sampling mask
            combine_rss: If True, use RSS combination

        Returns:
            Image tensor
        """
        from .fft_ops import sense_adjoint

        return sense_adjoint(
            kspace,
            smaps=sensitivity_maps,
            mask=mask,
            combine_rss=combine_rss,
        )

    # =========================================================================
    # Data Consistency
    # =========================================================================

    def get_dc_module(self) -> nn.Module:
        """Get or create the data consistency module."""
        if self._dc_module is None:
            dc_type = self._config["dc_type"]
            dc_weight = self._config["dc_weight"]

            from .data_consistency import (
                AdaptiveDataConsistency,
                HardDataConsistency,
                SimpleDataConsistency,
            )

            if dc_type == "hard":
                self._dc_module = HardDataConsistency()
            elif dc_type == "adaptive":
                self._dc_module = AdaptiveDataConsistency()
            elif dc_type == "soft":
                self._dc_module = SimpleDataConsistency(weight=dc_weight)
            else:
                raise ValueError(
                    f"Unknown dc_type {dc_type!r}; expected one of 'soft', 'hard', 'adaptive'"
                )

            self._dc_module = self._dc_module.to(self._device)

        return self._dc_module

    def enforce_data_consistency(
        self,
        predicted_image: torch.Tensor,
        measured_kspace: torch.Tensor,
        mask: torch.Tensor | None = None,
        sensitivity_maps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Enforce data consistency between prediction and measurements.

        Args:
            predicted_image: Current image estimate
            measured_kspace: Acquired k-space measurements
            mask: Sampling mask
            sensitivity_maps: Coil sensitivity maps (for multi-coil)

        Returns:
            Data-consistent image estimate
        """
        dc = self.get_dc_module()
        if sensitivity_maps is not None:
            # Only modules whose forward() accepts smaps can honour the
            # multi-coil contract (AdaptiveDataConsistency does real SENSE
            # forward/adjoint). Forwarding to HardDataConsistency /
            # SimpleDataConsistency would silently drop the coil maps
            # (single-coil DC on multi-coil data) -- pitfall #15. Raise
            # instead of degrading.
            import inspect

            if "sensitivity_maps" in inspect.signature(dc.forward).parameters:
                return dc(
                    predicted_image,
                    measured_kspace,
                    mask,
                    sensitivity_maps=sensitivity_maps,
                )
            raise NotImplementedError(
                f"enforce_data_consistency: dc_type="
                f"{self._config['dc_type']!r} ({type(dc).__name__}) does not "
                "support multi-coil sensitivity_maps. Use dc_type='adaptive' "
                "for SENSE-based data consistency, or call without "
                "sensitivity_maps for single-coil DC."
            )
        return dc(predicted_image, measured_kspace, mask)

    # =========================================================================
    # Operator Registry Access
    # =========================================================================

    def create_operator(self, name: str, **kwargs: Any) -> nn.Module:
        """Create a physics operator from the registry.

        Args:
            name: Registered operator name
            **kwargs: Operator configuration

        Returns:
            Configured operator instance
        """
        from .registry import create_operator

        return create_operator(name, **kwargs)

    def list_operators(self) -> list[str]:
        """List all registered physics operators."""
        from .registry import list_operators

        return list(list_operators().keys())

    # =========================================================================
    # Volumetric Operations
    # =========================================================================

    def fft_volume_slice(self, volume: torch.Tensor) -> torch.Tensor:
        """Apply FFT along slice dimension (z-axis) for volumetric data.

        Args:
            volume: (B, C, D, H, W) or (D, H, W)

        Returns:
            Volume in Fourier domain along z
        """
        from .fft_ops import fft_volume_slice_dimension_uncentered

        return fft_volume_slice_dimension_uncentered(volume)

    def ifft_volume_slice(self, kspace: torch.Tensor) -> torch.Tensor:
        """Inverse FFT along slice dimension.

        Args:
            kspace: (B, C, D, H, W) or (D, H, W)

        Returns:
            Volume in image domain
        """
        from .fft_ops import ifft_volume_slice_dimension_uncentered

        return ifft_volume_slice_dimension_uncentered(kspace)

    def fft_volume_spatial_uncentered(self, volume: torch.Tensor) -> torch.Tensor:
        """Apply FFT along spatial dimensions (H, W) for volumetric data.

        Args:
            volume: (B, C, D, H, W)

        Returns:
            Volume in Fourier domain (spatial)
        """
        from .fft_ops import fft_volume_spatial_uncentered

        return fft_volume_spatial_uncentered(volume)

    def ifft_volume_spatial_uncentered(self, kspace: torch.Tensor) -> torch.Tensor:
        """Inverse FFT along spatial dimensions.

        Args:
            kspace: (B, C, D, H, W)

        Returns:
            Volume in image domain
        """
        from .fft_ops import ifft_volume_spatial_uncentered

        return ifft_volume_spatial_uncentered(kspace)


# =========================================================================
# Module-level convenience functions
# =========================================================================

_default_engine: PhysicsEngine | None = None


def get_physics_engine(device: str = "cpu", **kwargs: Any) -> PhysicsEngine:
    """Get or create the default physics engine.

    Args:
        device: Target device
        **kwargs: Additional configuration

    Returns:
        PhysicsEngine instance
    """
    global _default_engine
    if _default_engine is None or str(_default_engine.device) != device:
        _default_engine = PhysicsEngine(device=device, **kwargs)
    return _default_engine


def image_to_kspace(
    image: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Transform image to k-space using default engine."""
    engine = get_physics_engine(device=str(image.device))
    return engine.image_to_kspace(image, mask)


def kspace_to_image(
    kspace: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Transform k-space to image using default engine."""
    engine = get_physics_engine(device=str(kspace.device))
    return engine.kspace_to_image(kspace, mask)


__all__ = [
    "PhysicsEngine",
    "get_physics_engine",
    "image_to_kspace",
    "kspace_to_image",
]
