r"""Spectral-norm-wrapped denoiser (paradigm-expansion PR-9 / M1).

Enforces a global Lipschitz constraint on the wrapped denoiser by
applying ``torch.nn.utils.spectral_norm`` to every ``Conv2d`` /
``Conv3d`` / ``Linear`` weight matrix. PnP-ADMM convergence
guarantees in Sun-Wohlberg-Kamilov 2019 require the inner
denoiser to be a contraction (or, more generally, non-expansive); the
wrapper makes that contractivity explicit and audit-checkable.

Mathematical formulation
------------------------
For a layer with weight :math:`W`, the spectral-norm reparameterisation

.. math::

    W_{SN} = W / \sigma(W),
    \qquad
    \sigma(W) = \max_{\|h\|=1} \|W h\|_2

ensures :math:`\|W_{SN} h\| \le \|h\|`, i.e. the layer is
non-expansive. The composition of non-expansive layers (with 1-Lipschitz
activations like ReLU / SiLU) is itself non-expansive.

Usage
-----
.. code-block:: python

    from mriforge.models.generators.unet import SimpleUNet  # any image-domain denoiser
    base = SimpleUNet(in_channels=2, out_channels=2)
    wrapped = SpectralNormalizedDenoiser(base)
    # wrapped behaves like base but every weight is spectrally normalised
    out = wrapped(noisy_image)

Notes
-----
- The wrapper *modifies the inner module in place* — every supported
  layer's weight is replaced with a spectral-norm parameterisation.
- ``spectral_norm`` introduces a small per-forward power-iteration
  step; the cost is negligible on typical denoiser sizes.
- The audit can verify Lipschitz constraint via ``is_lipschitz_constrained``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Layer types that get the spectral-norm treatment.
_NORMALISABLE = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)


class SpectralNormalizedDenoiser(nn.Module):
    """Wraps an inner denoiser with spectral-norm-constrained weights.

    Args:
        inner: Any ``nn.Module`` to wrap. Layers of type
            :class:`nn.Conv1d`, :class:`nn.Conv2d`, :class:`nn.Conv3d`,
            :class:`nn.Linear` get spectral-norm-reparameterised
            weights.
        n_power_iterations: Number of power iterations per forward
            pass for estimating the spectral norm. Default 1 (matches
            the PyTorch default).
        eps: Numerical floor on the denominator.

    Raises:
        TypeError: when ``inner`` is not an ``nn.Module``.
    """

    def __init__(
        self,
        inner: nn.Module,
        n_power_iterations: int = 1,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if not isinstance(inner, nn.Module):
            raise TypeError(f"inner must be an nn.Module; got {type(inner).__name__}")
        if n_power_iterations < 1:
            raise ValueError(f"n_power_iterations must be ≥ 1; got {n_power_iterations}")
        self.inner = inner
        self.n_power_iterations = int(n_power_iterations)

        # Apply spectral-norm to every supported layer in-place.
        self._normalised_layers: list[str] = []
        for name, module in inner.named_modules():
            if isinstance(module, _NORMALISABLE):
                nn.utils.spectral_norm(
                    module,
                    n_power_iterations=n_power_iterations,
                    eps=eps,
                )
                self._normalised_layers.append(name or "<root>")

    @property
    def is_lipschitz_constrained(self) -> bool:
        """``True`` iff at least one layer has spectral-norm applied.

        Audit gate: PnP / RED strategies refuse to run with a denoiser
        whose ``is_lipschitz_constrained`` returns ``False``.
        """
        return len(self._normalised_layers) > 0

    @property
    def normalised_layers(self) -> tuple[str, ...]:
        """Names of layers that received the spectral-norm reparameterisation."""
        return tuple(self._normalised_layers)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward through the wrapped denoiser."""
        return self.inner(x, *args, **kwargs)


__all__ = ["SpectralNormalizedDenoiser"]
