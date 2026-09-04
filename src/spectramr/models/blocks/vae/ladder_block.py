r"""Ladder-VAE block with precision-weighted Gaussian merge.

Implements one level of the Ladder VAE of Sønderby *et al.* [1]. Each
level carries a bottom-up deterministic statistic
:math:`(\hat\mu_\ell, \hat\sigma^2_\ell)` and a top-down generative prior
:math:`(\mu_{p,\ell}, \sigma^2_{p,\ell})`. The posterior is the exact
maximum-likelihood combination of the two independent Gaussian
observations of the same latent (the *precision-weighted product*):

.. math::

    \sigma_{q}^{-2} &= \hat\sigma^{-2} + \sigma_{p}^{-2}, \\
    \mu_{q} &= \bigl(\hat\mu\,\hat\sigma^{-2}
              + \mu_{p}\,\sigma_{p}^{-2}\bigr)\,\sigma_{q}^{2}.

Reference:
    [1] C. K. Sønderby, T. Raiko, L. Maaløe, S. K. Sønderby, and
        O. Winther, "Ladder variational autoencoders," NeurIPS 2016.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LadderBlock(nn.Module):
    r"""One level of the ladder VAE.

    The bottom-up path maps a deterministic feature vector to
    :math:`(\hat\mu, \log\hat\sigma^2)`. The top-down path maps a sample
    from the level above to the generative prior
    :math:`(\mu_p, \log\sigma_p^2)`. ``merge`` fuses them into the
    posterior via the precision-weighted product.

    Args:
        feature_dim: Dimensionality of the bottom-up feature vector fed
            to ``bottom_up``.
        latent_dim: Dimensionality of this level's latent.
        above_dim: Dimensionality of the latent of the level above
            (input to ``top_down``). When ``None`` this is the topmost
            block and ``top_down`` returns a standard-normal prior.
        hidden: Hidden width of the two-layer MLP heads.
    """

    def __init__(
        self,
        feature_dim: int,
        latent_dim: int,
        above_dim: int | None = None,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.above_dim = above_dim

        # Bottom-up inference head: feature -> (mu_hat, logvar_hat)
        self.bottom_up_net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.bu_mu = nn.Linear(hidden, latent_dim)
        self.bu_logvar = nn.Linear(hidden, latent_dim)

        # Top-down generative head: z_above -> (mu_p, logvar_p)
        if above_dim is not None:
            self.top_down_net = nn.Sequential(
                nn.Linear(above_dim, hidden),
                nn.LayerNorm(hidden),
                nn.SiLU(),
            )
            self.td_mu = nn.Linear(hidden, latent_dim)
            self.td_logvar = nn.Linear(hidden, latent_dim)
        else:
            self.top_down_net = None
            self.td_mu = None
            self.td_logvar = None

    def bottom_up(self, feature: Tensor) -> tuple[Tensor, Tensor]:
        """Produce the bottom-up Gaussian statistic.

        Args:
            feature: Deterministic feature vector ``(B, feature_dim)``.

        Returns:
            ``(mu_hat, logvar_hat)`` each ``(B, latent_dim)``.
        """
        h = self.bottom_up_net(feature)
        return self.bu_mu(h), self.bu_logvar(h)

    def top_down(self, z_above: Tensor | None) -> tuple[Tensor, Tensor]:
        """Produce the top-down generative prior.

        For the topmost block (``above_dim is None``) the prior is the
        unit Gaussian, returned as zero tensors broadcastable against the
        bottom-up statistic.

        Args:
            z_above: Latent sample from the level above ``(B, above_dim)``
                or ``None`` for the topmost block.

        Returns:
            ``(mu_p, logvar_p)``.
        """
        if self.top_down_net is None or z_above is None:
            # Unit-Gaussian prior; scalar zeros broadcast in ``merge``.
            return (
                torch.zeros(1, self.latent_dim, device=self._device()),
                torch.zeros(1, self.latent_dim, device=self._device()),
            )
        h = self.top_down_net(z_above)
        return self.td_mu(h), self.td_logvar(h)

    def merge(
        self,
        mu_hat: Tensor,
        logvar_hat: Tensor,
        mu_p: Tensor,
        logvar_p: Tensor,
    ) -> tuple[Tensor, Tensor]:
        r"""Precision-weighted Gaussian product.

        Returns the posterior ``(mu_q, logvar_q)`` such that
        :math:`\sigma_q^{-2} = \hat\sigma^{-2} + \sigma_p^{-2}` and
        :math:`\mu_q = (\hat\mu\,\hat\sigma^{-2} + \mu_p\,\sigma_p^{-2})
        \sigma_q^2`.
        """
        prec_hat = (-logvar_hat).exp()
        prec_p = (-logvar_p).exp()
        prec_q = prec_hat + prec_p
        logvar_q = -prec_q.log()
        mu_q = (mu_hat * prec_hat + mu_p * prec_p) / prec_q
        return mu_q, logvar_q

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        """Sample ``z = mu + eps * exp(0.5 * logvar)``."""
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    @staticmethod
    def kl_two_gaussians(
        mu_q: Tensor,
        logvar_q: Tensor,
        mu_p: Tensor,
        logvar_p: Tensor,
    ) -> Tensor:
        r"""KL( N(mu_q, var_q) || N(mu_p, var_p) ), summed over the latent.

        .. math::

            \mathrm{KL} = \tfrac{1}{2}\sum_d \Bigl[
                \log\frac{\sigma_p^2}{\sigma_q^2}
                + \frac{\sigma_q^2 + (\mu_q - \mu_p)^2}{\sigma_p^2}
                - 1 \Bigr].

        Returns the per-sample KL ``(B,)``; broadcasts scalar priors.
        """
        var_q = logvar_q.exp()
        var_p = logvar_p.exp()
        term = logvar_p - logvar_q + (var_q + (mu_q - mu_p).pow(2)) / var_p - 1.0
        return 0.5 * term.sum(dim=-1)

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def extra_repr(self) -> str:
        return (
            f"feature_dim={self.feature_dim}, latent_dim={self.latent_dim}, "
            f"above_dim={self.above_dim}"
        )


__all__ = ["LadderBlock"]
