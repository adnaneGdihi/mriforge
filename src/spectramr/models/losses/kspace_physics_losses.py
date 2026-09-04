"""
Physics-informed loss functions for k-space cold diffusion.

Implements frequency-weighted losses that account for k-space spectral properties
(1/f decay) and enforce high-frequency reconstruction accuracy.
"""

import torch
import torch.nn as nn

from spectramr.models.losses.registry import register_loss


@register_loss("frequency_weighted_l1_kspace")
class FrequencyWeightedL1Loss(nn.Module):
    """
    L1 loss with frequency-dependent weighting for k-space reconstruction.

    Addresses the natural 1/f spectral decay in MRI by upweighting high-frequency
    components. Prevents blur artifacts where standard losses allow high-frequency
    errors to dominate during optimization.

    The weight map W(r) increases radially from DC (weight=1.0) at center to
    edge (weight=1.0+alpha*r_normalized) at domain boundary.

    Args:
        height (int): Height of k-space data.
        width (int): Width of k-space data.
        alpha (float): Weighting exponent. Controls aggressiveness of high-frequency
                      emphasis. Default=2.0 empirically balances robustness.
                      - alpha=1.0: Linear scaling with radius
                      - alpha=2.0: Quadratic scaling (recommended)
                      - alpha=3.0: Aggressive high-frequency emphasis
        reduction (str): 'mean', 'sum', or 'none'. Default='mean'.

    Shape:
        - Input: (B, 2*C, H, W) where 2*C represents stacked real/imaginary
                 OR (B, C, H, W) for complex64/complex128 tensors
        - Target: Same shape as input
        - Output: Scalar loss (if reduction='mean' or 'sum') or (B, 2*C, H, W)

    Physics:
        Natural k-space decay follows approximately 1/r due to distance-dependent
        signal attenuation. This loss compensates by upweighting periphery via:

            W(r) = 1.0 + alpha * (r_norm / r_max)

        where r_norm is radial distance from center and r_max is maximum radius.
        Applied element-wise: loss = mean(W(r) * |pred - target|)

    Example:
        >>> loss_fn = FrequencyWeightedL1Loss(height=256, width=256, alpha=2.0)
        >>> pred = torch.randn(2, 2, 256, 256)  # [B, 2*C, H, W] for complex
        >>> target = torch.randn(2, 2, 256, 256)
        >>> loss = loss_fn(pred, target)
        >>> loss.backward()
    """

    def __init__(
        self,
        height: int = 256,
        width: int = 256,
        alpha: float = 2.0,
        reduction: str = "mean",
    ):
        """Initialize FrequencyWeightedL1Loss.

        Note: height and width are optional defaults. The loss will dynamically
        adapt to input tensor shapes during forward pass.
        """
        super().__init__()
        self.default_height = height
        self.default_width = width
        self.alpha = alpha
        self.reduction = reduction

        # Cache for dynamically computed weight maps (keyed by (H, W))
        self._weight_map_cache: dict = {}

    def _get_weight_map(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Get or create frequency weight map for given dimensions.

        Caches computed weight maps to avoid recomputation on same shapes.

        Args:
            height (int): Height of required weight map.
            width (int): Width of required weight map.
            device (torch.device): Device where weight map should be placed.
            dtype (torch.dtype): Data type for weight map.

        Returns:
            torch.Tensor: Weight map shape (1, 1, height, width).
        """
        cache_key = (height, width)

        # Return cached weight map if available and on correct device
        if cache_key in self._weight_map_cache:
            cached = self._weight_map_cache[cache_key]
            if cached.device == device:
                return cached

        # Compute new weight map
        weight_map = self._create_frequency_weight_map(height, width, self.alpha)
        # Ensure weights are always real, matching precision of input
        real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
        weight_map = weight_map.to(device=device, dtype=real_dtype)

        # Cache it
        self._weight_map_cache[cache_key] = weight_map

        return weight_map

    def _create_frequency_weight_map(
        self,
        height: int,
        width: int,
        alpha: float,
    ) -> torch.Tensor:
        """
        Create radial frequency weight map.

        Computes 2D Gaussian-like radial distance from center, then applies
        linear scaling. Center (DC) has weight ~1.0, edges have weight ~1.0+alpha.

        Args:
            height (int): Height of map.
            width (int): Width of map.
            alpha (float): Scaling exponent.

        Returns:
            torch.Tensor: Weight map shape (1, 1, height, width) normalized to [1, 1+alpha].
        """
        # Create meshgrid centered at (0, 0)
        y = torch.arange(height, dtype=torch.float32) - height / 2.0
        x = torch.arange(width, dtype=torch.float32) - width / 2.0
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        # Radial distance from center
        r = torch.sqrt(xx**2 + yy**2)

        # Normalize to [0, 1]
        r_max = torch.sqrt(
            torch.tensor((height / 2.0) ** 2 + (width / 2.0) ** 2, dtype=torch.float32)
        )
        r_norm = r / (r_max + 1e-8)

        # Linear weight: 1.0 (center) to 1.0+alpha (edges)
        weight = 1.0 + alpha * r_norm

        # Reshape for broadcasting: [1, 1, H, W]
        weight_map = weight.unsqueeze(0).unsqueeze(0)

        # Normalize mean weight to 1.0 for stable training
        weight_map = weight_map / (weight_map.mean() + 1e-8)

        return weight_map

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute frequency-weighted L1 loss.

        Args:
            pred (torch.Tensor): Predicted k-space, shape (B, 2*C, H, W) or (B, C, H, W) for complex.
            target (torch.Tensor): Target k-space, same shape as pred.

        Returns:
            torch.Tensor: Loss value (scalar or per-sample depending on reduction).

        Raises:
            RuntimeError: If shapes mismatch.
        """
        if pred.shape != target.shape:
            raise RuntimeError(f"Pred and target shape mismatch: {pred.shape} vs {target.shape}")

        # Handle both 4D (B,C,H,W) and 5D (B,C,D,H,W) tensors.
        # For 5D inputs (volumetric/multi-slice), collapse depth into batch.
        original_shape = pred.shape
        if pred.dim() == 5:
            B, C, D, H, W = pred.shape
            pred = pred.reshape(B * D, C, H, W)
            target = target.reshape(B * D, C, H, W)
        elif pred.dim() == 4:
            pass  # Already (B, C, H, W)
        else:
            raise RuntimeError(f"Expected 4D or 5D tensor, got {pred.dim()}D: {pred.shape}")

        # Get weight map dynamically based on spatial dims
        height, width = pred.shape[-2], pred.shape[-1]
        weight_map = self._get_weight_map(height, width, pred.device, pred.dtype)

        # Handle both stacked real/imag [B, 2*C, H, W] and complex tensors
        if pred.is_complex():
            # Convert complex to magnitude for weighting (phase-neutral)
            pred_magnitude = torch.abs(pred)
            target_magnitude = torch.abs(target)
            unweighted_loss = torch.nn.functional.l1_loss(
                pred_magnitude, target_magnitude, reduction="none"
            )
        else:
            # Stacked real/imag: [B, 2*C, H, W]
            unweighted_loss = torch.nn.functional.l1_loss(pred, target, reduction="none")

        # Apply frequency weights: broadcast weight_map [1, 1, H, W] to loss [B, C, H, W]
        weighted_loss = unweighted_loss * weight_map

        # Apply reduction
        if self.reduction == "mean":
            return weighted_loss.mean()
        elif self.reduction == "sum":
            return weighted_loss.sum()
        elif self.reduction == "none":
            return weighted_loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f"default_height={self.default_height}, default_width={self.default_width}, "
            f"alpha={self.alpha}, reduction='{self.reduction}', "
            f"(dynamically adapts to input shape)"
        )


@register_loss("sobolev_kspace")
class SobolevKSpaceLoss(nn.Module):
    """Fourier-Domain Sobolev Loss.

    By the derivative theorem of Fourier transforms, spatial derivatives map to
    multiplying k-space by frequency coordinates. This loss enforces:

    L_Sobolev = mean_{kx, ky} (1 + kx^2 + ky^2) * |K_pred(kx, ky) - K_true(kx, ky)|_1

    This natively up-weights high-frequency edge sharpness directly in k-space.
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{SobolevKSpace} = \\mathbb{E}_{k_x, k_y} \\left[ (1 + k_x^2 + k_y^2)^{s/2} | \\hat{k} - k | \right]
    """

    def __init__(self, order: float = 2.0, reduction: str = "mean"):
        """Fourier-domain Sobolev loss of order ``order``.

        Args:
            order: The ``s`` in ``(1 + k_x^2 + k_y^2)^{s/2}``. The default 2.0
                reproduces the exponent this class hardcoded until 2026-08, so
                existing arms are unaffected. ``order=1.0`` is the
                ``sqrt(1 + |k|^2)`` weighting that 56 exp_11 arms asked for via
                the dead ``sobolev_order`` key (#560, #615); note
                ``sobolev_frequency`` is a separate registered loss weighting by
                ``(1 + |k|)`` exactly, which may be the closer intent.
                Immutable after construction — the weight-map cache is keyed on
                shape alone.
            reduction: Reduction passed to the underlying magnitude penalty.
        """
        super().__init__()
        self.order = float(order)
        self.reduction = reduction
        self._weight_map_cache = {}

    def _get_weight_map(
        self, height: int, width: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """_get_weight_map.

        Args:
            height (int): Description.
            width (int): Description.
            device (torch.device): Description.
            dtype (torch.dtype): Description.
        Returns:
            torch.Tensor: Description.
        """
        cache_key = (height, width)
        if cache_key in self._weight_map_cache:
            cached = self._weight_map_cache[cache_key]
            if cached.device == device:
                return cached

        # Create meshgrid with frequencies normalized to [-1, 1]
        y = torch.arange(height, dtype=torch.float32) - height / 2.0
        x = torch.arange(width, dtype=torch.float32) - width / 2.0

        # Normalize to maximum frequency 1.0 at the edge
        y = y / (height / 2.0)
        x = x / (width / 2.0)

        yy, xx = torch.meshgrid(y, x, indexing="ij")

        # W(kx, ky) = (1 + kx^2 + ky^2)^(s/2); s=2 is the historical hardcoded
        # case, so the default `order` leaves every existing arm unchanged.
        weight = (1.0 + xx**2 + yy**2) ** (self.order / 2.0)

        weight_map = weight.unsqueeze(0).unsqueeze(0)
        real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
        weight_map = weight_map.to(device=device, dtype=real_dtype)

        self._weight_map_cache[cache_key] = weight_map
        return weight_map

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            pred (torch.Tensor): Description.
            target (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for SobolevKSpaceLoss.

        Executes PyTorch tensor operations.

        Args:
            pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        if pred.shape != target.shape:
            raise RuntimeError(f"Shape mismatch: {pred.shape} vs {target.shape}")

        # Handle both 4D (B,C,H,W) and 5D (B,C,D,H,W) tensors.
        if pred.dim() == 5:
            B, C, D, H, W = pred.shape
            pred = pred.reshape(B * D, C, H, W)
            target = target.reshape(B * D, C, H, W)
        elif pred.dim() != 4:
            raise RuntimeError(f"Expected 4D or 5D tensor, got {pred.dim()}D: {pred.shape}")

        batch_size, channels, height, width = pred.shape
        weight_map = self._get_weight_map(height, width, pred.device, pred.dtype)

        if pred.is_complex():
            diff_mag = torch.abs(pred - target)
        else:
            if pred.ndim == 4 and pred.shape[1] % 2 == 0:
                # Real-stacked: compute magnitude of diff
                B, C, H, W = pred.shape
                diff = pred - target
                diff_reshaped = diff.permute(0, 2, 3, 1).contiguous().view(B, H, W, C // 2, 2)
                diff_complex = torch.view_as_complex(diff_reshaped).permute(0, 3, 1, 2)
                diff_mag = torch.abs(diff_complex)
                # Ensure weight_map broadcasts to C // 2
            else:
                diff_mag = torch.abs(pred - target)

        # Handle broadcasting if needed due to complex casting
        if diff_mag.shape[1] * 2 == pred.shape[1] and pred.ndim == 4 and not pred.is_complex():
            # We went from C to C/2 channels
            pass

        weighted_loss = diff_mag * weight_map

        if self.reduction == "mean":
            return weighted_loss.mean()
        elif self.reduction == "sum":
            return weighted_loss.sum()
        elif self.reduction == "none":
            return weighted_loss
        else:
            raise RuntimeError(f"Unknown reduction: {self.reduction}")
