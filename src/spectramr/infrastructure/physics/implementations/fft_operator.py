"""FFT-based Forward Operator Implementation

This module provides the FFT-based forward operator implementation
that can be registered with the physics operator registry.
"""

import torch

from spectramr.infrastructure.physics.registry import BaseForwardOperator, register_operator


class FFT2DOperator(BaseForwardOperator):
    """2D FFT-based forward operator for MRI reconstruction."""

    def __init__(
        self,
        normalized: bool = True,
        center: bool = True,
        img_size: tuple[int, int] | None = None,
    ):
        """__init__.

        Args:
            normalized (bool): Description.
            center (bool): Description.
            img_size (Optional[tuple[int, int]]): Description.
        """
        super().__init__()
        self.normalized = normalized
        self.center = center
        self._img_size = img_size or (256, 256)  # Default size

    def _validate_setup(self) -> None:
        """Validate operator setup."""
        # No special setup required for FFT

    def forward(
        self,
        img: torch.Tensor,
        smaps: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward operator: image -> k-space.

        Args:
            img: Real-valued image tensor (B, 1 or C, H, W)
            smaps: Optional complex sensitivity maps (B, C_coils, H, W)
            mask: Optional real-valued sampling mask (B, 1 or C, H, W)
            **kwargs: Additional parameters

        Returns:
            Complex k-space tensor (B, C_coils or C, H, W)

        """
        if img.ndim > 4:
            raise RuntimeError(f"Input image must have at most 4 dimensions, got {img.ndim}")

        if smaps is not None:
            # Multi-coil forward model: image * coil_sens -> FFT
            img_coil = img * smaps
        else:
            img_coil = img

        if self.center:
            norm = "ortho"
            kspace = torch.fft.fftshift(
                torch.fft.fft2(img_coil, norm=norm),
                dim=(-2, -1),
            )
        else:
            norm = "ortho"
            kspace = torch.fft.fft2(img_coil, norm=norm)

        if mask is not None:
            kspace = kspace * mask

        return kspace

    def adjoint(
        self,
        kspace: torch.Tensor,
        smaps: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        combine_rss: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Adjoint operator: k-space -> image.

        Args:
            kspace: Complex k-space tensor (B, C_coils or C, H, W)
            smaps: Optional complex sensitivity maps (B, C_coils, H, W)
            mask: Optional real-valued sampling mask (B, 1 or C, H, W)
            combine_rss: If True and smaps provided, return RSS
                combined magnitude
            **kwargs: Additional parameters

        Returns:
            Real or complex image tensor (B, 1 or C, H, W)

        """
        if kspace.ndim > 4:
            raise RuntimeError(f"Input k-space must have at most 4 dimensions, got {kspace.ndim}")

        if mask is not None:
            kspace = kspace * mask

        if self.center:
            norm = "ortho"
            img_coil = torch.fft.ifft2(
                torch.fft.ifftshift(kspace, dim=(-2, -1)),
                norm=norm,
            )
        else:
            norm = "ortho"
            img_coil = torch.fft.ifft2(kspace, norm=norm)

        if smaps is not None:
            if combine_rss:
                # Root sum of squares combination
                img = torch.sqrt(
                    torch.sum(torch.abs(img_coil * smaps.conj()) ** 2, dim=1, keepdim=True),
                )
            else:
                # Multi-coil adjoint: IFFT -> sum over coils
                img = torch.sum(img_coil * smaps.conj(), dim=1, keepdim=True)
        else:
            img = img_coil

        return img

    @property
    def img_size(self) -> tuple[int, int]:
        """Get image size (H, W)."""
        return self._img_size

    def get_operator_type(self) -> str:
        """Get string identifier for operator type."""
        return "fft2d"


# Register the operator
register_operator(
    name="fft2d",
    operator_class=FFT2DOperator,
    config={"normalized": True, "center": True},
)


class MultiCoilFFTOperator(BaseForwardOperator):
    r"""Multi-coil FFT operator with sensitivity map support.

    This operator handles multi-coil MRI data by applying sensitivity maps
    during forward and adjoint operations. It supports both SENSE reconstruction
    and RSS coil combination.
    Mathematical Formulation:
    .. math::

        \mathcal{O}_{MultiCoilFFT}(x) = \mathcal{F}(x \odot S) \odot M"""

    def __init__(
        self,
        normalized: bool = True,
        center: bool = True,
        img_size: tuple[int, int] | None = None,
    ):
        """__init__.

        Args:
            normalized (bool): Description.
            center (bool): Description.
            img_size (Optional[tuple[int, int]]): Description.
        """
        super().__init__()
        self.normalized = normalized
        self.center = center
        self._img_size = img_size or (256, 256)  # Default size

    def _validate_setup(self) -> None:
        """Validate operator setup."""
        # No special setup required for multi-coil FFT

    def forward(
        self,
        img: torch.Tensor,
        smaps: torch.Tensor,
        mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Multi-coil forward operator: image -> k-space.

        Args:
            img: Real-valued image tensor (B, 1, H, W)
            smaps: Complex sensitivity maps (B, num_coils, H, W)
            mask: Optional real-valued sampling mask (B, 1, H, W)
            **kwargs: Additional parameters

        Returns:
            Complex k-space tensor (B, num_coils, H, W)

        """
        if smaps is None:
            raise ValueError("MultiCoilFFTOperator requires sensitivity maps (smaps)")

        # Apply sensitivity maps: coil_images = image * smaps
        # img: (B, 1, H, W), smaps: (B, num_coils, H, W)
        coil_images = img * smaps  # Broadcasting: (B, num_coils, H, W)

        # Apply FFT
        if self.center:
            norm = "ortho"
            kspace = torch.fft.fftshift(
                torch.fft.fft2(coil_images, norm=norm),
                dim=(-2, -1),
            )
        else:
            norm = "ortho"
            kspace = torch.fft.fft2(coil_images, norm=norm)

        # Apply sampling mask if provided
        if mask is not None:
            kspace = kspace * mask

        return kspace

    def adjoint(
        self,
        kspace: torch.Tensor,
        smaps: torch.Tensor,
        mask: torch.Tensor | None = None,
        combine_method: str = "sense",
        **kwargs,
    ) -> torch.Tensor:
        """Multi-coil adjoint operator: k-space -> image.

        Args:
            kspace: Complex k-space tensor (B, num_coils, H, W)
            smaps: Complex sensitivity maps (B, num_coils, H, W)
            mask: Optional real-valued sampling mask (B, 1, H, W)
            combine_method: Coil combination method ('sense' or 'rss')
            **kwargs: Additional parameters

        Returns:
            Real-valued image tensor (B, 1, H, W)

        """
        if smaps is None:
            raise ValueError("MultiCoilFFTOperator requires sensitivity maps (smaps)")

        # Apply sampling mask if provided
        if mask is not None:
            kspace = kspace * mask

        # Apply inverse FFT
        if self.center:
            norm = "ortho"
            coil_images = torch.fft.ifft2(
                torch.fft.ifftshift(kspace, dim=(-2, -1)),
                norm=norm,
            )
        else:
            norm = "ortho"
            coil_images = torch.fft.ifft2(kspace, norm=norm)

        # Coil combination
        if combine_method == "rss":
            # Root sum of squares
            combined = torch.sqrt(
                torch.sum(torch.abs(coil_images * smaps.conj()) ** 2, dim=1, keepdim=True)
            )
        elif combine_method == "sense":
            # SENSE combination: sum over coils of conj(smaps) * coil_images
            combined = torch.sum(coil_images * smaps.conj(), dim=1, keepdim=True)
        else:
            raise ValueError(f"Unknown combine_method: {combine_method}")

        return combined

    @property
    def img_size(self) -> tuple[int, int]:
        """Get image size (H, W)."""
        return self._img_size

    def get_operator_type(self) -> str:
        """Get string identifier for operator type."""
        return "multi_coil_fft"


# Register the multi-coil operator
register_operator(
    name="multi_coil_fft",
    operator_class=MultiCoilFFTOperator,
    config={"normalized": True, "center": True},
)
