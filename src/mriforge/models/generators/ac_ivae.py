r"""AC-iVAE -- Acquisition-Conditioned identifiable VAE.

Disentangles anatomy from contrast with an *identifiability guarantee*, by using
the continuous acquisition vector :math:`\mathbf u=\boldsymbol\varphi` as the
iVAE auxiliary variable (Khemakhem et al. 2020). The conditionally-factorial
exponential-family prior

.. math::

   p_{\boldsymbol\lambda}(\mathbf z\mid\mathbf u)
     = \prod_i \frac{Q_i(z_i)}{Z_i(\mathbf u)}
       \exp\!\Bigl[\textstyle\sum_j T_{ij}(z_i)\,\lambda_{ij}(\mathbf u)\Bigr]

makes the latent identifiable up to permutation + pointwise transform when the
sufficient-statistic differences across distinct :math:`\mathbf u` have full
column rank. The continuous acquisition manifold supplies that variability once
at least :math:`2m+1` distinct settings appear; a single (constant) acquisition
makes the rank condition fail, collapsing to the unidentifiable VAE that CALAMITI
operates in -- the precise reason multi-acquisition data is necessary rather than
incidental.

This module provides the prior + the variability-rank diagnostic (the
identifiability test) and a registered AC-iVAE with an anatomy/contrast latent
split.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.models.registry import register_model


class ACIVAEPrior(nn.Module):
    r"""Conditional exponential-family prior: :math:`\mathbf u\mapsto\boldsymbol
    \lambda(\mathbf u)`.

    Emits the natural parameters (2 per latent dim: a Gaussian's
    :math:`(\mu/\sigma^2,\,-1/2\sigma^2)`) as a nonlinear function of the
    acquisition vector.

    Args:
        u_dim: Acquisition-vector dimension.
        latent_dim: Latent dimension ``m`` (``n_natural = 2m``).
        hidden: Hidden width.
    """

    def __init__(self, u_dim: int, latent_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.net = nn.Sequential(
            nn.Linear(u_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2 * latent_dim),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Natural parameters ``[B, 2*latent_dim]``."""
        return self.net(u)


def acquisition_variability_rank(
    prior: ACIVAEPrior,
    u_values: torch.Tensor,
    tol: float = 1e-5,
) -> int:
    r"""Rank of the sufficient-statistic-difference matrix across acquisitions.

    Computes :math:`[\boldsymbol\lambda(\mathbf u_i)-\boldsymbol\lambda(\mathbf
    u_0)]_{i\ge1}` and returns its matrix rank. Full rank (:math:`2m`) certifies
    the iVAE variability condition; rank ``0`` (constant :math:`\mathbf u`) is the
    unidentifiable collapse.

    Args:
        prior: The conditional prior.
        u_values: Distinct acquisition settings ``[K, u_dim]`` (``K >= 2m+1``).
        tol: Singular-value tolerance for the rank.

    Returns:
        The rank of the difference matrix.
    """
    with torch.no_grad():
        lambdas = prior(u_values)  # [K, 2m]
    diffs = lambdas[1:] - lambdas[0:1]  # [K-1, 2m]
    return int(torch.linalg.matrix_rank(diffs, tol=tol).item())


def _encoder(in_channels: int, latent_dim: int, width: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, width, 3, stride=2, padding=1),
        nn.GELU(),
        nn.Conv2d(width, width, 3, stride=2, padding=1),
        nn.GELU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(width, 2 * latent_dim),  # (mu, logvar)
    )


@register_model(
    name="ac_ivae",
    training_mode="vae",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
)
class ACIVAE(nn.Module):
    r"""Acquisition-conditioned identifiable VAE with an anatomy/contrast split.

    Args:
        in_channels: Input image channels.
        anatomy_dim: Anatomy latent dimension :math:`\mathbf z_a`.
        contrast_dim: Contrast latent dimension :math:`\mathbf z_c`.
        u_dim: Acquisition-vector dimension (the iVAE auxiliary variable).
        width: Encoder/decoder width.
        image_size: Spatial size used to seed the decoder.
    """

    def __init__(
        self,
        in_channels: int = 1,
        anatomy_dim: int = 16,
        contrast_dim: int = 4,
        u_dim: int = 5,
        width: int = 32,
        image_size: int = 16,
    ) -> None:
        super().__init__()
        self.anatomy_dim = int(anatomy_dim)
        self.contrast_dim = int(contrast_dim)
        latent_dim = anatomy_dim + contrast_dim
        self.image_size = int(image_size)
        self.encoder = _encoder(in_channels, latent_dim, width)
        self.prior = ACIVAEPrior(u_dim=u_dim, latent_dim=latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, width * (image_size // 4) ** 2)
        self.width = width
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(width, width, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(width, in_channels, 4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode, split anatomy/contrast, reparametrise, decode.

        Args:
            x: Input image ``[B, in_channels, H, W]``.
            u: Acquisition vector ``[B, u_dim]`` (auxiliary variable).

        Returns:
            ``{"reconstruction", "z_anatomy", "z_contrast", "mu", "logvar",
            "prior_natural_params"}``.
        """
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        z_anatomy = z[:, : self.anatomy_dim]
        z_contrast = z[:, self.anatomy_dim :]

        side = self.image_size // 4
        dec = self.decoder_fc(z).reshape(-1, self.width, side, side)
        reconstruction = self.decoder(dec)
        return {
            "reconstruction": reconstruction,
            "z_anatomy": z_anatomy,
            "z_contrast": z_contrast,
            "mu": mu,
            "logvar": logvar,
            "prior_natural_params": self.prior(u),
        }


__all__ = ["ACIVAE", "ACIVAEPrior", "acquisition_variability_rank"]
