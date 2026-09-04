"""Parameter extraction from Gaussian clouds."""

import logging

import torch

from spectramr.infrastructure.physics.implementations.gaussian_fourier_op import GaussianCloud

logger = logging.getLogger(__name__)


class OptimizableParameters:
    """Container for optimizable Gaussian parameters."""

    def __init__(
        self,
        means: torch.Tensor | None = None,
        scales: torch.Tensor | None = None,
        rotations: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
        opacity: torch.Tensor | None = None,
    ):
        """__init__.

        Args:
            means (Optional[torch.Tensor]): Description.
            scales (Optional[torch.Tensor]): Description.
            rotations (Optional[torch.Tensor]): Description.
            features (Optional[torch.Tensor]): Description.
            opacity (Optional[torch.Tensor]): Description.
        """
        self.means = means
        self.scales = scales
        self.rotations = rotations
        self.features = features
        self.opacity = opacity

    def to_list(self) -> list[torch.Tensor]:
        """Convert to list of tensors for optimizer."""
        params = []
        if self.means is not None:
            params.append(self.means)
        if self.scales is not None:
            params.append(self.scales)
        if self.rotations is not None:
            params.append(self.rotations)
        if self.features is not None:
            params.append(self.features)
        if self.opacity is not None:
            params.append(self.opacity)
        return params

    def is_valid(self) -> bool:
        """Check if we have at least means for a valid Gaussian."""
        return self.means is not None


class ParameterExtractor:
    """Extracts optimizable parameters from Gaussian clouds.

    Handles multiple Gaussian cloud formats (MedGS, standard GaussianCloud, etc.)
    by attempting various attribute access patterns.
    """

    def __init__(self, device: torch.device):
        """
        Args:
            device: Target device for parameter tensors.
        """
        self.device = device

    def extract(self, gaussians: GaussianCloud) -> OptimizableParameters:
        """Extract optimizable parameters from a Gaussian cloud.

        Args:
            gaussians: Input Gaussian cloud (various formats supported).

        Returns:
            OptimizableParameters container with cloned, optimizable tensors.

        Note:
            All parameters are cloned, moved to device, and set to require gradients.
        """
        params_to_optimize = []

        def make_optimizable(val: torch.Tensor | None) -> torch.Tensor | None:
            """Clone tensor and make it optimizable."""
            if val is None:
                return None
            if isinstance(val, (list, tuple)):
                return None
            if not isinstance(val, torch.Tensor):
                return None
            t = val.clone().detach().to(self.device).requires_grad_(True)
            params_to_optimize.append(t)
            return t

        def get_val(obj, key: str):
            """Try multiple attribute access patterns."""
            for name in [f"get_{key}", f"_{key}", key]:
                val = getattr(obj, name, None)
                if val is not None:
                    return val
            return None

        try:
            # Extract means (MedGS uses _coeffs, others use xyz/means)
            if hasattr(gaussians, "_coeffs"):
                opt_means = make_optimizable(gaussians._coeffs)
            else:
                means_candidate = get_val(gaussians, "xyz")
                if means_candidate is None:
                    means_candidate = getattr(gaussians, "means", None)
                opt_means = make_optimizable(means_candidate)

            # Extract scales
            scales_candidate = get_val(gaussians, "scaling")
            if scales_candidate is None:
                scales_candidate = get_val(gaussians, "scale")
            if scales_candidate is None:
                scales_candidate = getattr(gaussians, "scales", None)
            opt_scales = make_optimizable(scales_candidate)

            # Extract rotations
            rots_candidate = get_val(gaussians, "rotation")
            if rots_candidate is None:
                rots_candidate = get_val(gaussians, "quaternions")
            if rots_candidate is None:
                rots_candidate = getattr(gaussians, "quaternions", None)
            opt_rotations = make_optimizable(rots_candidate)

            # Extract features
            features_candidate = get_val(gaussians, "features")
            if features_candidate is None:
                features_candidate = getattr(gaussians, "features", None)
            opt_features = make_optimizable(features_candidate)

            # Extract opacity
            opacity_candidate = get_val(gaussians, "opacity")
            if opacity_candidate is None:
                opacity_candidate = getattr(gaussians, "opacity", None)
            opt_opacity = make_optimizable(opacity_candidate)

        except Exception as e:
            logger.debug(f"Error extracting optimizable params: {e}")
            opt_means = None
            opt_scales = None
            opt_rotations = None
            opt_features = None
            opt_opacity = None

        return OptimizableParameters(
            means=opt_means,
            scales=opt_scales,
            rotations=opt_rotations,
            features=opt_features,
            opacity=opt_opacity,
        )
