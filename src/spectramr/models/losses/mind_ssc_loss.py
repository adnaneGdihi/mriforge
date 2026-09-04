"""MIND-SSC Loss for intensity-invariant anatomical structure matching.

MIND-SSC (Modality Independent Neighbourhood Descriptor with Self-Similarity Context)
computes self-similarity features that are robust to intensity transformations between
different MRI contrasts (e.g., ULF vs HF).

References:
    Heinrich et al. "MIND: Modality independent neighbourhood descriptor for multi-modal
    deformable registration." Medical Image Analysis, 2012.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss


@register_loss("mind_ssc", aliases=["MINDSSCLoss", "anat_lock", "mind_loss"], domain="image")
class MINDSSCLoss(nn.Module):
    """MIND-SSC loss for anatomical structure preservation.

    **DOMAIN**: IMAGE-SPACE ONLY
    **Input**: [B, 1, H, W] or [B, C, H, W] image-space intensity
    **3D Support**: Yes (3D kernel support)

    Intensity-invariant self-similarity descriptors robust to non-linear intensity
    transformations (e.g., ULF vs HF contrasts). Do NOT use with k-space [B, 2, H, W].

    Computes self-similarity features by comparing local patches using
    Gaussian-weighted differences.

    Args:
        patch_size: Size of local neighborhood for descriptor (default: 7)
        search_radius: Radius for self-similarity context (default: 3)
        gaussian_sigma: Sigma for Gaussian weighting (default: 1.0)
        use_self_similarity: Enable self-similarity context (default: True)

    Forward Args:
        pred: Predicted image [B, C, H, W]
        target: Target image [B, C, H, W]

    Returns:
        MIND-SSC loss (L2 distance between descriptors)

    Example:
        >>> loss_fn = MINDSSCLoss(patch_size=7, search_radius=3)
        >>> loss = loss_fn(pred_image, target_image)
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{MIND} = \\| \text{MIND}(\\hat{y}) - \text{MIND}(y) \\|_2^2"""

    def __init__(
        self,
        patch_size: int = 7,
        search_radius: int = 3,
        gaussian_sigma: float = 1.0,
        use_self_similarity: bool = True,
    ):
        """__init__.

        Args:
            patch_size (int): Description.
            search_radius (int): Description.
            gaussian_sigma (float): Description.
            use_self_similarity (bool): Description.
        """
        super().__init__()
        self.patch_size = patch_size
        self.search_radius = search_radius
        self.gaussian_sigma = gaussian_sigma
        self.use_self_similarity = use_self_similarity
        self.padding = patch_size // 2

        # Pre-compute the 8 directional Gaussian kernels for the MIND descriptor.
        # Directions: the full 3x3 neighborhood (excluding the centre). Using the
        # COMPLETE, symmetric neighbourhood is what makes MIND insensitive to
        # orientation. The previous ``directions[:6]`` slice kept an asymmetric
        # subset -- it dropped (1, 0) and (1, 1) while keeping (-1, 0)/(-1, 1) --
        # which biased the descriptor toward the upper-left and broke that
        # rotation-insensitivity.
        self.directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),  # Top row
            (0, -1),
            (0, 1),  # Middle row (skip center)
            (1, -1),
            (1, 0),
            (1, 1),  # Bottom row
        ]

    def _compute_gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """Create 2D Gaussian kernel."""
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2

        g = coords**2
        g = (-g / (2 * sigma**2)).exp()

        kernel = g.unsqueeze(-1) @ g.unsqueeze(0)
        kernel /= kernel.sum()
        return kernel

    def _compute_mind_descriptor(self, img: torch.Tensor) -> torch.Tensor:
        """Compute MIND descriptor for image.

        Args:
            img: Input image [B, C, H, W]

        Returns:
            MIND descriptor [B, C*8, H, W] (8 directional features per channel)
        """
        B, C, H, W = img.shape
        device = img.device

        # Gaussian kernel for weighted differences
        gaussian_kernel = self._compute_gaussian_kernel(self.patch_size, self.gaussian_sigma).to(
            device
        )
        gaussian_kernel = gaussian_kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]

        mind_features = []

        for dx, dy in self.directions:
            # Shift image in direction (dx, dy)
            shifted = torch.roll(img, shifts=(dy, dx), dims=(-2, -1))

            # Compute squared difference
            diff = (img - shifted) ** 2

            # Apply Gaussian weighting via convolution
            # Use groups=C to process each channel independently
            weighted_diff = F.conv2d(
                diff,
                gaussian_kernel.repeat(C, 1, 1, 1),
                padding=self.padding,
                groups=C,
            )

            mind_features.append(weighted_diff)

        # Stack all directional features: [B, C*8, H, W]
        mind = torch.cat(mind_features, dim=1)

        # Variance normalization (self-similarity)
        if self.use_self_similarity:
            # Compute local variance for normalization
            mean_kernel = torch.ones(1, 1, self.patch_size, self.patch_size).to(device)
            mean_kernel /= self.patch_size**2

            local_mean = F.conv2d(
                mind,
                mean_kernel.repeat(mind.shape[1], 1, 1, 1),
                padding=self.padding,
                groups=mind.shape[1],
            )
            local_var = (
                F.conv2d(
                    mind**2,
                    mean_kernel.repeat(mind.shape[1], 1, 1, 1),
                    padding=self.padding,
                    groups=mind.shape[1],
                )
                - local_mean**2
            )

            # Normalize by local variance (add epsilon for stability)
            # [FIX] Clamp variance to be non-negative AND add epsilon before sqrt to prevent NaN gradient at 0
            # Variance computation (E[x^2] - E[x]^2) can be negative due to float precision
            local_var = torch.clamp(local_var, min=1e-6)
            mind = mind / local_var.sqrt()

        return mind

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute MIND-SSC loss between predicted and target images.

        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]

        Returns:
            Scalar loss value

        forward method for MINDSSCLoss.

        Executes PyTorch tensor operations.

        Args:
            pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Compute MIND descriptors for both images
        mind_pred = self._compute_mind_descriptor(pred)
        mind_target = self._compute_mind_descriptor(target)

        # L2 distance between descriptors
        loss = F.mse_loss(mind_pred, mind_target)

        return loss
