"""Model wrappers — small composable adapters around registered models.

Provides:

- :class:`SpectralNormalizedDenoiser` — wraps any image-domain
  denoiser to enforce a global Lipschitz constraint via
  ``torch.nn.utils.spectral_norm``. Required for PnP-ADMM
  convergence per Sun-Wohlberg-Kamilov 2019 (paradigm-expansion
  PR-9 / M1).
"""

from .spectral_normalized import SpectralNormalizedDenoiser

__all__ = ["SpectralNormalizedDenoiser"]
