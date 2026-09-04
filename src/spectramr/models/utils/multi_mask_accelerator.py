"""Multi-mask k-space accelerator.

This module provides the MultiMaskAccelerator, which alternates between
different sampling strategies (e.g., horizontal and vertical Cartesian)
to ensure diverse k-space coverage during the diffusion process.
"""

from typing import Any

import torch

from spectramr.infrastructure.physics.sampling import (
    MAX_ACCELERATION,
    KSpaceAccelerator,
    UniformCartesianKSpaceAccelerator,
)


class MultiMaskAccelerator(KSpaceAccelerator):
    """Alternating axis k-space accelerator.

    This accelerator alternates between horizontal (axis 'y') and vertical
    (axis 'x') Cartesian undersampling based on the timestep. This ensures
    that over the course of the diffusion process, k-space is sampled
    more uniformly in 2D than with a static 1D pattern.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        max_acceleration: float = MAX_ACCELERATION,
        center_fraction: float = 0.08,
        seed: int | None = None,
        **kwargs: Any,
    ):
        """Initialize the MultiMaskAccelerator.

        Args:
            num_timesteps: Total number of diffusion timesteps.
            max_acceleration: Maximum acceleration factor (at t=0).
            center_fraction: Fraction of fully sampled center lines.
            seed: Random seed for reproducibility.
            **kwargs: Additional arguments passed to the underlying accelerators.
        """
        super().__init__(num_timesteps, max_acceleration)

        # This accelerator alternates between BOTH axes by construction, so a
        # single declared axis is not something it can honour. Since
        # `mask_direction` became live (issue #948) it arrives here as
        # `line_axis` in **kwargs and collided with the explicit values below,
        # raising an opaque "got multiple values for keyword argument". Say what
        # is actually wrong instead of dropping it silently (pitfall #15).
        if "line_axis" in kwargs or "mask_direction" in kwargs:
            declared = kwargs.get("mask_direction") or kwargs.get("line_axis")
            raise ValueError(
                f"multi_mask alternates between both k-space axes, so it cannot "
                f"honour a single declared direction ({declared!r}). Remove "
                f"`mask_direction` from the undersampling block, or choose an "
                f"acceleration_type that samples along one axis "
                f"(equispaced, variable_density_1d, ...)."
            )

        # Create two sub-accelerators for the different axes
        # We share the seed logic but offset it to ensure independence if needed,
        # though alternating implies we want complementary coverage.

        self.accelerator_y = UniformCartesianKSpaceAccelerator(
            num_timesteps=num_timesteps,
            max_acceleration=max_acceleration,
            center_fraction=center_fraction,
            line_axis="y",
            seed=seed,
            **kwargs,
        )

        self.accelerator_x = UniformCartesianKSpaceAccelerator(
            num_timesteps=num_timesteps,
            max_acceleration=max_acceleration,
            center_fraction=center_fraction,
            line_axis="x",
            seed=seed,  # Use same seed base, internal logic handles offsets
            **kwargs,
        )

    def get_acceleration_mask(
        self,
        kspace_shape: tuple[int, ...],
        timestep: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Generate an acceleration mask, alternating axes by timestep.

        Args:
            kspace_shape: Shape of k-space data.
            timestep: Current diffusion timestep.
            device: Target device.

        Returns:
            Binary mask tensor.
        """
        # Alternate based on parity of timestep
        # Even steps get Y-axis lines (Horizontal lines in image ie Vertical in K-space?
        # - UniformCartesian: axis='y' -> mask[:, selected, :] = True.
        #   Indices are rows. So these are horizontal lines of K-space points.

        if timestep % 2 == 0:
            return self.accelerator_y.get_acceleration_mask(kspace_shape, timestep, device)
        else:
            return self.accelerator_x.get_acceleration_mask(kspace_shape, timestep, device)

    def get_acceleration_factor(self, timestep: int) -> float:
        """Get acceleration factor for given timestep.

        Both sub-accelerators share the same schedule logic.
        """
        # They should match, so just delegate to one
        return self.accelerator_y.get_acceleration_factor(timestep)
