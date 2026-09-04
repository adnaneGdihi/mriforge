r"""Spectral-norm Lipschitz certificate utilities (LCAH core).

A model whose linear/convolutional layers are spectral-normalised and whose
activations are 1-Lipschitz (ReLU/GELU/LeakyReLU) is :math:`L`-Lipschitz with
:math:`L \le \prod_\ell \sigma_{\max}(W_\ell)`. LCAH uses this to attach a
finite, computable extrapolation certificate to acquisition-parameter
conditioning: with the hypernetwork :math:`h_\psi` and target :math:`f_\theta`
both spectral-normalised,

.. math::

   \sup_{\mathbf x}\bigl\|f_{h_\psi(\boldsymbol\varphi)}(\mathbf x)
       - f_{h_\psi(\boldsymbol\varphi^\ast)}(\mathbf x)\bigr\|
   \;\le\; L_w L_h\,\bigl\|\boldsymbol\varphi - \boldsymbol\varphi^\ast\bigr\|,

with :math:`L_h \le \prod_\ell\sigma_{\max}(W^h_\ell)` and :math:`L_w \le
\prod_\ell\sigma_{\max}(W^f_\ell)`. The certificate is zero at a training point
(consistency) and grows at most linearly with distance to the training convex
hull -- a computable deployment guarantee.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def layer_spectral_norm(layer: nn.Module) -> float:
    r"""Largest singular value :math:`\sigma_{\max}(W)` of a linear/conv layer.

    Conv weights ``[out, in, *k]`` are reshaped to ``[out, in*prod(k)]`` (the
    standard upper bound on the conv operator norm).

    Raises:
        TypeError: if ``layer`` carries no ``weight`` matrix.
    """
    weight = getattr(layer, "weight", None)
    if weight is None:
        raise TypeError(f"{type(layer).__name__} has no weight to bound.")
    w = weight.detach()
    if w.ndim > 2:
        w = w.reshape(w.shape[0], -1)
    return float(torch.linalg.matrix_norm(w, ord=2).item())


def spectral_norm_product(module: nn.Module) -> float:
    r"""Lipschitz upper bound :math:`\prod_\ell \sigma_{\max}(W_\ell)`.

    Multiplies the spectral norms of every ``Linear`` / ``Conv{1,2,3}d`` leaf in
    ``module``. Valid as a Lipschitz bound when the interleaved activations are
    1-Lipschitz (ReLU/GELU/LeakyReLU/tanh/sigmoid).
    """
    product = 1.0
    linear_types = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
    for sub in module.modules():
        if isinstance(sub, linear_types):
            product *= layer_spectral_norm(sub)
    return product


def extrapolation_radius(
    lipschitz: float,
    phi: torch.Tensor,
    phi_train: torch.Tensor,
) -> torch.Tensor:
    r"""Certified drift radius :math:`L\,\min_i\lVert\varphi-\varphi_i\rVert`.

    Args:
        lipschitz: The composed Lipschitz constant :math:`L = L_w L_h`.
        phi: Query acquisition vector ``[d]``.
        phi_train: Training acquisition vectors ``[N, d]``.

    Returns:
        Scalar tensor: the certified worst-case prediction drift at ``phi``. Zero
        when ``phi`` coincides with a training acquisition.
    """
    distances = torch.linalg.vector_norm(phi_train - phi.unsqueeze(0), dim=-1)
    return lipschitz * distances.min()


__all__ = [
    "extrapolation_radius",
    "layer_spectral_norm",
    "spectral_norm_product",
]
