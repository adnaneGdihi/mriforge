from dataclasses import dataclass

import torch


@dataclass
class BiophysicalGaussian:
    """
    Unified representation for Lagrangian Neuro-Splatting (LaNS) and
    Holographic Bloch-Splatting (HoBS).

    Attributes:
        means: (B, N, 3) Centroids in spatial domain (x, y, z).
        scales: (B, N, 3) Log-scales or actual scales.
        quaternions: (B, N, 4) Rotation as quaternions (w, x, y, z).
        opacity: (B, N, 1) Alpha/Density.

        # LaNS Extensions (Dynamics)
        velocity_embedding: (B, N, D) Latent code conditioning the Neural ODE velocity field.

        # HoBS Extensions (Biophysics)
        biophysics: (B, N, 3) Tuple of (M0, T1, T2).
                    M0: Proton Density / Equilibrium Magnetization.
                    T1: Longitudinal Relaxation Time.
                    T2: Transverse Relaxation Time.

        active_mask: (B, N, 1) Binary mask for culled/active gaussians.
    """

    means: torch.Tensor
    scales: torch.Tensor
    quaternions: torch.Tensor
    opacity: torch.Tensor

    velocity_embedding: torch.Tensor | None = None
    biophysics: torch.Tensor | None = None
    active_mask: torch.Tensor | None = None

    @property
    def density(self) -> torch.Tensor:
        """Alias for opacity"""
        return self.opacity

    @property
    def num_points(self) -> int:
        """num_points.

        Returns:
            int: Description.
        """
        return self.means.shape[1]

    @property
    def batch_size(self) -> int:
        """batch_size.

        Returns:
            int: Description.
        """
        return self.means.shape[0]

    @property
    def m0(self) -> torch.Tensor | None:
        """m0.

        Returns:
            Optional[torch.Tensor]: Description.
        """
        if self.biophysics is not None:
            return self.biophysics[..., 0:1]
        return None

    @property
    def t1(self) -> torch.Tensor | None:
        """t1.

        Returns:
            Optional[torch.Tensor]: Description.
        """
        if self.biophysics is not None:
            return self.biophysics[..., 1:2]
        return None

    @property
    def t2(self) -> torch.Tensor | None:
        """t2.

        Returns:
            Optional[torch.Tensor]: Description.
        """
        if self.biophysics is not None:
            return self.biophysics[..., 2:3]
        return None

    def clone(self) -> "BiophysicalGaussian":
        """clone.

        Returns:
            'BiophysicalGaussian': Description.
        """
        return BiophysicalGaussian(
            means=self.means.clone(),
            scales=self.scales.clone(),
            quaternions=self.quaternions.clone(),
            opacity=self.opacity.clone(),
            velocity_embedding=(
                self.velocity_embedding.clone() if self.velocity_embedding is not None else None
            ),
            biophysics=self.biophysics.clone() if self.biophysics is not None else None,
            active_mask=(self.active_mask.clone() if self.active_mask is not None else None),
        )

    def detach(self) -> "BiophysicalGaussian":
        """detach.

        Returns:
            'BiophysicalGaussian': Description.
        """
        return BiophysicalGaussian(
            means=self.means.detach(),
            scales=self.scales.detach(),
            quaternions=self.quaternions.detach(),
            opacity=self.opacity.detach(),
            velocity_embedding=(
                self.velocity_embedding.detach() if self.velocity_embedding is not None else None
            ),
            biophysics=(self.biophysics.detach() if self.biophysics is not None else None),
            active_mask=(self.active_mask.detach() if self.active_mask is not None else None),
        )
