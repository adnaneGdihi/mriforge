r"""β-TC-VAE-GAN (``beta_vae_gan``).

Composition of the β-TC-VAE [1] with a hinge-GAN discriminator. The
β-TC-VAE up-weights the total-correlation term to enforce disentangled
latents; the discriminator adds a perceptual adversarial signal:

.. math::

    \mathcal{L}_{\text{total}} = \mathcal{L}_{\beta\text{-TC}}
        + \lambda_{\text{adv}}\,
          \mathbb{E}_q[\max(0, 1 - D_\psi(\hat x))].

This entry is primarily a cross-paradigm composition test: ``LossBuilder``
must compose ``beta_tc_vae``, an adversarial term, and a reconstruction
term cleanly. The TC term is computed in-model via the existing
:class:`spectramr.models.losses.beta_tc_vae_loss.BetaTCVAELoss`; the
discriminator is the existing
:class:`spectramr.models.discriminators.patchgan_discriminator.PatchGANDiscriminator`.

Reference:
    [1] T. Q. Chen, X. Li, R. Grosse, D. Duvenaud, "Isolating sources of
        disentanglement in variational autoencoders," NeurIPS 2018.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from spectramr.models.discriminators.patchgan_discriminator import PatchGANDiscriminator
from spectramr.models.losses.beta_tc_vae_loss import BetaTCVAELoss
from spectramr.models.registry import register_model


class _GaussianVAE(nn.Module):
    """Self-contained Gaussian VAE returning ``(x_hat, z, mu, logvar)``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        hidden: int,
        pooled_size: int,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.pooled_size = pooled_size
        feature_dim = hidden * pooled_size * pooled_size

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden // 2, 4, 2, 1),
            nn.GroupNorm(8, hidden // 2),
            nn.SiLU(),
            nn.Conv2d(hidden // 2, hidden, 4, 2, 1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((pooled_size, pooled_size))
        self.fc_mu = nn.Linear(feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(feature_dim, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, feature_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden, hidden // 2, 4, 2, 1),
            nn.GroupNorm(8, hidden // 2),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden // 2, hidden // 2, 4, 2, 1),
            nn.GroupNorm(8, hidden // 2),
            nn.SiLU(),
            nn.Conv2d(hidden // 2, out_channels, 3, 1, 1),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        h = self.pool(self.encoder(x))
        h = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)
        d = self.dec_fc(z).view(-1, self.hidden, self.pooled_size, self.pooled_size)
        x_hat = self.decoder(d)
        if x_hat.shape[-2:] != x.shape[-2:]:
            x_hat = F.interpolate(x_hat, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x_hat, z, mu, logvar


@register_model(
    name="beta_vae_gan",
    training_mode="gan",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=False,
)
class BetaVAEGAN(nn.Module):
    r"""β-TC-VAE with an adversarial head.

    Args:
        in_channels: Input image channels.
        out_channels: Reconstruction channels.
        latent_dim: VAE latent width.
        hidden: VAE trunk width.
        pooled_size: Spatial size pooled to before the latent heads.
        beta: Total-correlation weight (β ≫ 1 enforces disentanglement).
        alpha: Mutual-information weight in the β-TC-VAE decomposition.
        gamma: Dimension-wise KL weight.
        lambda_adv: Adversarial-loss weight.
        ndf: Discriminator base channel count.
        disc_layers: Discriminator depth.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_dim: int = 32,
        hidden: int = 128,
        pooled_size: int = 4,
        beta: float = 6.0,
        alpha: float = 1.0,
        gamma: float = 1.0,
        lambda_adv: float = 0.1,
        ndf: int = 64,
        disc_layers: int = 3,
    ) -> None:
        super().__init__()
        self.lambda_adv = lambda_adv
        self.vae = _GaussianVAE(in_channels, out_channels, latent_dim, hidden, pooled_size)
        self.discriminator = PatchGANDiscriminator(
            in_channels=out_channels, ndf=ndf, n_layers=disc_layers
        )
        self.beta_tc = BetaTCVAELoss(alpha=alpha, beta=beta, gamma=gamma)

    @property
    def name(self) -> str:
        return "beta_vae_gan"

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def hinge_g_loss(d_fake: Tensor) -> Tensor:
        """Generator hinge loss ``E[max(0, 1 - D(x_hat))]``."""
        return F.relu(1.0 - d_fake).mean()

    @staticmethod
    def hinge_d_loss(d_real: Tensor, d_fake: Tensor) -> Tensor:
        """Discriminator hinge loss ``E[relu(1-D(x)) + relu(1+D(x_hat))]``."""
        return F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()

    def beta_tc_term(self, x: Tensor) -> Tensor:
        """The β-TC-VAE disentanglement loss for input ``x`` (scalar).

        Exposed so ``BetaVAEGANStrategy`` can add it to the generator loss
        of the standard GAN two-player loop — i.e. wire the β-TC objective
        into training without the model needing its own discriminator or a
        dict forward. Encodes ``x`` and evaluates the registered
        :class:`BetaTCVAELoss` on the resulting latents.
        """
        x_hat, z, mu, logvar = self.vae(x)
        return self.beta_tc(z, mu, logvar)

    def forward(self, x: Tensor) -> Tensor:
        """Generator forward: return the reconstruction ``x_hat``.

        ``GANTrainingStrategy`` calls ``generator(x)`` and expects an image
        tensor (it drives the adversarial loop with its own configured
        discriminator). The full β-TC / adversarial composition dict — used
        by tests and the cross-paradigm composition story — is available via
        :meth:`forward_full`.
        """
        return self.forward_full(x)["x_hat"]

    def forward_full(self, x: Tensor) -> dict[str, Tensor]:
        """Full forward pass exposing the composition terms.

        Returns:
            dict with ``x_hat``, the β-TC decomposition (``mi``, ``tc``,
            ``dw_kl`` plus the weighted ``beta_tc_loss``), the
            discriminator logits ``d_real`` / ``d_fake``, the generator
            adversarial loss ``adv_loss``, and the composite
            ``total_loss`` (β-TC + λ·adv + reconstruction).
        """
        x_hat, z, mu, logvar = self.vae(x)
        d_real = self.discriminator(x)
        d_fake = self.discriminator(x_hat)

        kl_breakdown = self._kl_decomposition(z, mu, logvar)
        beta_tc_loss = self.beta_tc(z, mu, logvar)
        adv_loss = self.hinge_g_loss(d_fake)
        recon = F.l1_loss(x_hat, x)
        total = beta_tc_loss + self.lambda_adv * adv_loss + recon

        return {
            "x_hat": x_hat,
            "z": z,
            "mu": mu,
            "log_var": logvar,
            "mi": kl_breakdown["mi"],
            "tc": kl_breakdown["tc"],
            "dw_kl": kl_breakdown["dw_kl"],
            "beta_tc_loss": beta_tc_loss,
            "d_real": d_real,
            "d_fake": d_fake,
            "adv_loss": adv_loss,
            "recon_loss": recon,
            "total_loss": total,
        }

    @staticmethod
    def _kl_decomposition(z: Tensor, mu: Tensor, logvar: Tensor) -> dict[str, Tensor]:
        r"""Unweighted MI / TC / dimension-wise-KL components.

        Mirrors the minibatch-weighted estimator in
        :class:`BetaTCVAELoss` so the returned ``tc`` term can be asserted
        positive by the audit probe.
        """
        import math

        B, D = z.shape

        def log_gauss(x: Tensor, m: Tensor, lv: Tensor) -> Tensor:
            return -0.5 * (math.log(2 * math.pi) + lv + (x - m).pow(2) * torch.exp(-lv))

        log_qz_given_x = log_gauss(z, mu, logvar).sum(dim=1)
        mat = log_gauss(z.unsqueeze(1), mu.unsqueeze(0), logvar.unsqueeze(0))  # (B,B,D)
        log_qz = torch.logsumexp(mat.sum(dim=2), dim=1) - math.log(B)
        log_prod_qz = (torch.logsumexp(mat, dim=1) - math.log(B)).sum(dim=1)
        log_pz = (-0.5 * (math.log(2 * math.pi) + z.pow(2))).sum(dim=1)

        mi = (log_qz_given_x - log_qz).mean()
        tc = (log_qz - log_prod_qz).mean()
        dw_kl = (log_prod_qz - log_pz).mean()
        return {"mi": mi, "tc": tc, "dw_kl": dw_kl}


__all__ = ["BetaVAEGAN"]
