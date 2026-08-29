"""Bloch Projector for MRF-DiPh
==============================

Projects noisy diffusion estimates onto the Bloch manifold using
dictionary matching and signal resynthesis.

The projector ensures that the denoised estimate is a physically
valid MRI signal by:
1. Matching to nearest tissue parameters (T1, T2, PD)
2. Resynthesizing signal via Bloch equations
"""

import torch
import torch.nn as nn

from mriforge.infrastructure.physics.bloch_simulation import DifferentiableBlochSimulator

from .bloch_dictionary import BlochDictionary


class BlochProjector(nn.Module):
    """Projects signals onto the Bloch manifold.

    Given an arbitrary image (e.g., from diffusion denoising), this module:
    1. Estimates tissue parameters via dictionary matching
    2. Resynthesizes a physically valid signal using Bloch equations

    This acts as a "physics regularizer" that prevents the diffusion
    model from generating impossible tissue contrasts.

    Args:
        dictionary: Pre-computed Bloch dictionary
        bloch_sim: Differentiable Bloch simulator
        sequence_type: "SE" or "GRE"
        smooth_matching: Use soft (differentiable) matching
    """

    def __init__(
        self,
        dictionary: BlochDictionary,
        bloch_sim: DifferentiableBlochSimulator | None = None,
        sequence_type: str = "SE",
        smooth_matching: bool = True,
        temperature: float = 0.1,
    ):
        """__init__.

        Args:
            dictionary (BlochDictionary): Description.
            bloch_sim (Optional[DifferentiableBlochSimulator]): Description.
            sequence_type (str): Description.
            smooth_matching (bool): Description.
            temperature (float): Description.
        """
        super().__init__()

        self.dictionary = dictionary
        self.bloch_sim = bloch_sim or DifferentiableBlochSimulator(sequence_type)
        self.sequence_type = sequence_type
        self.smooth_matching = smooth_matching
        self.temperature = temperature

    def forward(
        self,
        signal: torch.Tensor,
        TR: float,
        TE: float,
        alpha: float = 90.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project signal onto Bloch manifold.

        Args:
            signal: Input signal [B, 1, H, W]
            TR: Repetition time in ms
            TE: Echo time in ms
            alpha: Flip angle in degrees

        Returns:
            projected_signal: Physically valid signal [B, 1, H, W]
            tissue_maps: Estimated tissue parameters [B, 3, H, W]

        forward method for BlochProjector.

        Executes PyTorch tensor operations.

        Args:
            signal (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            TR (float): Expected input tensor.
            TE (float): Expected input tensor.
            alpha (float): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Step 1: Dictionary matching to estimate tissue parameters
        if self.smooth_matching:
            tissue_maps = self.dictionary.match_soft(signal, TR, TE, alpha, self.temperature)
        else:
            tissue_maps = self.dictionary.match(signal, TR, TE, alpha)

        # Step 2: Resynthesize signal using Bloch equations
        projected_signal = self.bloch_sim(tissue_maps, TR, TE, alpha)

        return projected_signal, tissue_maps

    def project(
        self,
        signal: torch.Tensor,
        TR: float,
        TE: float,
        alpha: float = 90.0,
    ) -> torch.Tensor:
        """Project signal onto Bloch manifold (returns only signal).

        Convenience method when tissue maps are not needed.
        """
        projected_signal, _ = self.forward(signal, TR, TE, alpha)
        return projected_signal

    def get_tissue_maps(
        self,
        signal: torch.Tensor,
        TR: float,
        TE: float,
        alpha: float = 90.0,
    ) -> torch.Tensor:
        """Get tissue parameter maps from signal.

        Convenience method for pure tissue estimation.
        """
        _, tissue_maps = self.forward(signal, TR, TE, alpha)
        return tissue_maps


class BlendedProjector(nn.Module):
    """Blends diffusion prediction with physics projection.

    This is the core of MRF-DiPh: we don't fully replace the diffusion
    output with the physics projection, but blend them with a learnable
    or fixed weight.

    Output = λ * projected + (1-λ) * original

    Args:
        projector: Bloch projector
        lambda_physics: Blend weight (0 = pure diffusion, 1 = pure physics)
        learnable: Whether lambda is a learnable parameter
    """

    def __init__(
        self,
        projector: BlochProjector,
        lambda_physics: float = 0.5,
        learnable: bool = False,
    ):
        """__init__.

        Args:
            projector (BlochProjector): Description.
            lambda_physics (float): Description.
            learnable (bool): Description.
        """
        super().__init__()

        self.projector = projector
        self.learnable = learnable

        if learnable:
            # Learnable blend weight (sigmoid-constrained to [0, 1])
            self._lambda_logit = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("_lambda_physics", torch.tensor(lambda_physics))

    @property
    def lambda_physics(self) -> torch.Tensor:
        """lambda_physics.

        Returns:
            torch.Tensor: Description.
        """
        if self.learnable:
            return torch.sigmoid(self._lambda_logit)
        return self._lambda_physics

    def forward(
        self,
        signal: torch.Tensor,
        TR: float,
        TE: float,
        alpha: float = 90.0,
    ) -> torch.Tensor:
        """Blend diffusion output with physics projection.

        Args:
            signal: Diffusion-denoised signal [B, 1, H, W]
            TR: Repetition time in ms
            TE: Echo time in ms
            alpha: Flip angle in degrees

        Returns:
            Blended signal [B, 1, H, W]

        forward method for BlendedProjector.

        Executes PyTorch tensor operations.

        Args:
            signal (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            TR (float): Expected input tensor.
            TE (float): Expected input tensor.
            alpha (float): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        projected_signal = self.projector.project(signal, TR, TE, alpha)

        lam = self.lambda_physics
        blended = lam * projected_signal + (1 - lam) * signal

        return blended


def create_bloch_projector(
    sequence_type: str = "SE",
    dictionary_resolution: str = "medium",
    smooth_matching: bool = True,
    device: torch.device | None = None,
) -> BlochProjector:
    """Factory function to create a Bloch projector.

    Args:
        sequence_type: "SE" or "GRE"
        dictionary_resolution: "low", "medium", or "high"
        smooth_matching: Use soft (differentiable) matching
        device: Compute device

    Returns:
        Configured BlochProjector
    """
    from .bloch_dictionary import create_bloch_dictionary

    dictionary = create_bloch_dictionary(
        sequence_type=sequence_type,
        resolution=dictionary_resolution,
        device=device,
    )

    bloch_sim = DifferentiableBlochSimulator(sequence_type)

    return BlochProjector(
        dictionary=dictionary,
        bloch_sim=bloch_sim,
        sequence_type=sequence_type,
        smooth_matching=smooth_matching,
    )
