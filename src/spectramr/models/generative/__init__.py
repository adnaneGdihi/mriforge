"""Generative model implementations.

Since 2026-07-02 (sota_registry retirement) ``spectramr.models.generative``
IS in the ``populate_model_registry`` walk list, so every module's
``@register_model`` decorator fires via auto-discovery. The explicit
import below is kept only for the ``BetaVAEGAN`` re-export.
"""

from .beta_vae_gan import BetaVAEGAN

__all__ = ["BetaVAEGAN"]
