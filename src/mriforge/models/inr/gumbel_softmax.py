"""Gumbel-Softmax Categorical Bottleneck for Disentangled Encoding.

Implements the categorical constraint from GLVS (Section 3.1.1 of tasks.md):
    s = E_anat(X), s ∈ {0, 1}^{H' × W' × D' × K}

The Gumbel-Softmax trick enables end-to-end training with discrete outputs
by using a continuous relaxation during training that approaches one-hot
as temperature τ → 0.

Based on:
- Jang et al., "Categorical Reparameterization with Gumbel-Softmax"
- Maddison et al., "The Concrete Distribution"

Follows unit-test.md rules:
- Strict typing with jaxtyping
- NaN/Inf safety checks
- Deterministic mode for testing
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from beartype import beartype
    from jaxtyping import Float, jaxtyped

    JAXTYPING_AVAILABLE = True
except ImportError:
    JAXTYPING_AVAILABLE = False

    def jaxtyped(fn):
        """jaxtyped.

        Args:
            fn (Any): Description.
        Returns:
            Any: Description.
        """
        return fn

    def beartype(fn):
        """beartype.

        Args:
            fn (Any): Description.
        Returns:
            Any: Description.
        """
        return fn


class GumbelSoftmax(nn.Module):
    """Gumbel-Softmax layer for discrete latent variables.

    During training: Returns soft samples using Gumbel-Softmax
    During inference: Returns hard one-hot samples (argmax)

    The temperature τ controls the smoothness:
    - τ → ∞: Uniform distribution
    - τ → 0: One-hot (discrete)
    """

    def __init__(
        self,
        temperature: float = 1.0,
        hard: bool = False,
        eps: float = 1e-10,
    ) -> None:
        """
        Args:
            temperature: Initial temperature (annealed during training)
            hard: If True, return hard one-hot even during training (straight-through)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.temperature = temperature
        self.hard = hard
        self.eps = eps

    @jaxtyped
    @beartype
    def forward(
        self,
        logits: Float[Tensor, "... num_classes"],
    ) -> Float[Tensor, "... num_classes"]:
        """
        Args:
            logits: [..., K] unnormalized log probabilities

        Returns:
            samples: [..., K] soft or hard categorical samples

        forward method for GumbelSoftmax.

        Executes PyTorch tensor operations.

        Args:
            logits (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Float[Tensor, '... num_classes']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.training:
            return self._gumbel_softmax_sample(logits, self.temperature, self.hard)
        else:
            # During inference, always use hard samples
            return self._hard_sample(logits)

    def _gumbel_softmax_sample(
        self,
        logits: Tensor,
        temperature: float,
        hard: bool,
    ) -> Tensor:
        """Sample from Gumbel-Softmax distribution."""
        # Sample Gumbel noise
        gumbels = self._sample_gumbel(logits.shape, logits.device, logits.dtype)

        # Add Gumbel noise and apply softmax with temperature
        y = (logits + gumbels) / temperature
        y_soft = F.softmax(y, dim=-1)

        # NaN safety check
        if not torch.isfinite(y_soft).all():
            # Fallback to regular softmax without noise
            y_soft = F.softmax(logits / temperature, dim=-1)

        if hard:
            # Straight-through estimator: hard sample in forward, soft in backward
            y_hard = self._hard_sample(logits)
            return (y_hard - y_soft).detach() + y_soft

        return y_soft

    def _sample_gumbel(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Sample from Gumbel(0, 1) distribution."""
        U = torch.rand(shape, device=device, dtype=dtype)
        # Clamp to avoid log(0)
        U = U.clamp(self.eps, 1.0 - self.eps)
        return -torch.log(-torch.log(U))

    def _hard_sample(self, logits: Tensor) -> Tensor:
        """Return hard one-hot sample."""
        index = logits.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        return y_hard

    def set_temperature(self, temperature: float) -> None:
        """Update temperature (for annealing)."""
        self.temperature = max(temperature, 0.1)  # Don't go below 0.1


class CategoricalAnatomyEncoder(nn.Module):
    """Anatomy Encoder with Gumbel-Softmax Categorical Bottleneck.

    Implements Section 3.1.1 of tasks.md:
    "By forcing s to be a discrete segmentation mask (softened during
    training via Gumbel-Softmax), we strip away intensity variations
    caused by MRI physics."

    The encoder outputs a spatial tensor s ∈ {0,1}^{H'×W'×D'×K} where K
    is the number of tissue classes. This categorical representation
    cannot encode continuous contrast variations, forcing disentanglement.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 8,  # K tissue types
        base_filters: int = 32,
        spatial_reduction: int = 8,  # Downsample factor
        initial_temperature: float = 1.0,
        hard_samples: bool = True,  # Use straight-through estimator
    ) -> None:
        """
        Args:
            in_channels: Input channels
            num_classes: Number of tissue classes (K)
            base_filters: Base filter count
            spatial_reduction: Spatial downsampling factor
            initial_temperature: Starting temperature for Gumbel-Softmax
            hard_samples: Use straight-through estimator for hard gradients
        """
        super().__init__()

        self.num_classes = num_classes
        self.spatial_reduction = spatial_reduction

        # Encoder backbone (spatial features)
        self.encoder = nn.Sequential(
            # Initial conv
            nn.Conv3d(in_channels, base_filters, 3, padding=1),
            nn.InstanceNorm3d(base_filters),
            nn.LeakyReLU(0.2, inplace=True),
            # Downsample 1: /2
            nn.Conv3d(base_filters, base_filters * 2, 3, stride=2, padding=1),
            nn.InstanceNorm3d(base_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Downsample 2: /4
            nn.Conv3d(base_filters * 2, base_filters * 4, 3, stride=2, padding=1),
            nn.InstanceNorm3d(base_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Downsample 3: /8
            nn.Conv3d(base_filters * 4, base_filters * 4, 3, stride=2, padding=1),
            nn.InstanceNorm3d(base_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Classifier head: features → class logits per voxel
        self.classifier = nn.Conv3d(base_filters * 4, num_classes, 1)

        # Gumbel-Softmax for discrete sampling
        self.gumbel = GumbelSoftmax(
            temperature=initial_temperature,
            hard=hard_samples,
        )

    @jaxtyped
    @beartype
    def forward(
        self,
        x: Float[Tensor, "batch channels depth height width"],
    ) -> Float[Tensor, "batch num_classes d_out h_out w_out"]:
        """
        Args:
            x: [B, C, D, H, W] input volume

        Returns:
            s: [B, K, D', H', W'] categorical spatial tensor
               During training: soft samples
               During inference: hard one-hot per voxel

        forward method for CategoricalAnatomyEncoder.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Float[Tensor, 'batch num_classes d_out h_out w_out']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Extract spatial features
        features = self.encoder(x)  # [B, F, D', H', W']

        # Compute class logits per voxel
        logits = self.classifier(features)  # [B, K, D', H', W']

        # Rearrange: [B, K, D', H', W'] -> [B, D', H', W', K]
        B, K, D, H, W = logits.shape
        logits = logits.permute(0, 2, 3, 4, 1).contiguous()  # [B, D', H', W', K]

        # Apply Gumbel-Softmax
        samples = self.gumbel(logits)  # [B, D', H', W', K]

        # Rearrange back: [B, D', H', W', K] -> [B, K, D', H', W']
        samples = samples.permute(0, 4, 1, 2, 3).contiguous()

        return samples

    def get_tissue_maps(
        self,
        x: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Get both soft probabilities and hard assignments.

        Returns:
            probs: [B, K, D', H', W'] soft class probabilities
            labels: [B, D', H', W'] hard class labels (argmax)
        """
        features = self.encoder(x)
        logits = self.classifier(features)  # [B, K, D', H', W']

        probs = F.softmax(logits, dim=1)
        labels = logits.argmax(dim=1)

        return probs, labels

    def set_temperature(self, temperature: float) -> None:
        """Set Gumbel-Softmax temperature (for annealing schedule)."""
        self.gumbel.set_temperature(temperature)


class CategoricalEntropyLoss(nn.Module):
    """Entropy loss to encourage discrete (one-hot) predictions.

    Implements L_mask from Section 3.3.4 of tasks.md:
        L_mask = Σ -s_ijk * log(s_ijk)

    Minimizing entropy pushes s toward one-hot vectors.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        """__init__.

        Args:
            eps (float): Description.
        """
        super().__init__()
        self.eps = eps

    @jaxtyped
    @beartype
    def forward(
        self,
        s: Float[Tensor, "batch num_classes *spatial"],
    ) -> Float[Tensor, ""]:
        """
        Args:
            s: [B, K, ...] soft categorical predictions

        Returns:
            entropy: Scalar entropy loss

        forward method for CategoricalEntropyLoss.

        Executes PyTorch tensor operations.

        Args:
            s (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Float[Tensor, '']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Clamp to avoid log(0)
        s_clamped = s.clamp(self.eps, 1.0 - self.eps)

        # Compute entropy: -Σ s * log(s)
        entropy = -s_clamped * torch.log(s_clamped)

        # Sum over classes, mean over batch and spatial
        entropy = entropy.sum(dim=1).mean()

        return entropy


class TemperatureScheduler:
    """Annealing schedule for Gumbel-Softmax temperature.

    Gradually reduces temperature during training to approach
    discrete (one-hot) samples.
    """

    def __init__(
        self,
        initial_temp: float = 1.0,
        min_temp: float = 0.1,
        anneal_rate: float = 0.00003,
    ) -> None:
        """
        Args:
            initial_temp: Starting temperature
            min_temp: Minimum temperature
            anneal_rate: Exponential decay rate
        """
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.anneal_rate = anneal_rate
        self.step_count = 0

    def step(self) -> float:
        """Advance schedule and return current temperature."""
        self.step_count += 1
        temp = max(
            self.min_temp,
            self.initial_temp * math.exp(-self.anneal_rate * self.step_count),
        )
        return temp

    def get_temperature(self) -> float:
        """Get current temperature without stepping."""
        return max(
            self.min_temp,
            self.initial_temp * math.exp(-self.anneal_rate * self.step_count),
        )


# Convenience function for integration
def create_categorical_encoder(
    in_channels: int = 1,
    num_classes: int = 8,
    temperature: float = 1.0,
    **kwargs,
) -> CategoricalAnatomyEncoder:
    """Factory function for categorical anatomy encoder."""
    return CategoricalAnatomyEncoder(
        in_channels=in_channels,
        num_classes=num_classes,
        initial_temperature=temperature,
        **kwargs,
    )
