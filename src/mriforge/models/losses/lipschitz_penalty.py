r"""Differentiable Lipschitz spectral penalty (A-8.3 *CertRob*, component A).

A training-time regulariser that drives every weight layer's largest singular
value :math:`\sigma_i` down toward a target (default 1), so the network's global
Lipschitz upper bound :math:`\prod_i\sigma_i` — certified at eval time by
:class:`~mriforge.core.metrics.lipschitz_bound_metrics.ComposedSpectralNormBound`
— shrinks. A reconstructor with a small Lipschitz constant cannot amplify a
worst-case input perturbation, the failure mode of Antun et al. (PNAS 2020).

.. math::

    \mathcal{L}_{\mathrm{Lip}} \;=\; \sum_i \operatorname{ReLU}\!\big(
        \hat{\sigma}_i - \sigma_{\mathrm{target}}\big),

which is **0 iff every** :math:`\hat\sigma_i \le \sigma_{\mathrm{target}}` (a
1-Lipschitz network for the default target 1) and positive otherwise — a hinge
that only penalises layers that exceed the budget, so it does not crush the
capacity of already-contractive layers.

**Why power iteration, not SVD (the honest engineering).** The companion *metric*
computes :math:`\sigma_i` by an exact SVD — accurate, run rarely, not
differentiated. This *loss* runs every step and must be differentiated through
:math:`W`; a full SVD is slow per step and its gradient is unstable near a
degenerate spectrum. So the penalty uses a few **power-iteration** steps with the
singular vectors :math:`u,v` *detached*, estimating
:math:`\hat\sigma = u^{\top} W v` — differentiable in :math:`W`, cheap, exactly
the estimator :func:`torch.nn.utils.spectral_norm` uses. The estimate is a slight
*under*-approximation early on (so the penalty is conservative); the metric is the
ground-truth certificate. For a ``Conv`` layer the weight is reshaped to 2-D
``[out, -1]`` (the Miyato/SN-GAN surrogate; see the metric's docstring for the
exact-vs-surrogate boundary).

**Anti-facade contract (CLAUDE.md #16).** This is a *parameter* penalty, not a
``(prediction, target)`` loss. Invoked without the live module it **raises** — it
cannot be dropped into a YAML ``losses`` block (the ``LossBuilder`` would call it
``loss(pred, target)`` and it would silently return 0, the inert-mechanism trap).
Wire it through a training strategy that passes ``model=`` (the certified-
robustness strategy), which also applies the ``lambda_lipschitz`` weight and
stamps it into provenance (#15).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from mriforge.models.losses.registry import register_loss

# Weight-bearing linear maps the penalty sums over — kept in step with
# ``core.metrics.lipschitz_bound_metrics`` (the eval-time certificate). Norm /
# activation / attention layers are intentionally excluded (separate Lipschitz
# behaviour); the penalty regularises the conv/linear backbone.
_WEIGHTED_LAYERS = (
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
)


def _power_iteration_sigma(weight: torch.Tensor, n_iters: int, eps: float) -> torch.Tensor:
    r"""Differentiable estimate of ``sigma_max(weight.reshape(out, -1))``.

    Power iteration with detached singular vectors, deterministically seeded
    (``v = 1/sqrt(in)``; no global RNG, per the training-loop perf rule). Returns
    a 0-d tensor that carries gradient through ``weight`` only (via
    :math:`u^{\top} W v`). Handles real and complex weights.
    """
    w = weight.reshape(weight.shape[0], -1)
    in_dim = w.shape[1]
    with torch.no_grad():
        v = torch.ones(in_dim, device=w.device, dtype=w.dtype)
        v = v / (v.norm() + eps)
        u = w @ v
        for _ in range(max(1, n_iters)):
            u = u / (u.norm() + eps)
            wt = w.conj().transpose(-2, -1) if w.is_complex() else w.transpose(-2, -1)
            v = wt @ u
            v = v / (v.norm() + eps)
            u = w @ v
        u = (u / (u.norm() + eps)).detach()
        v = v.detach()
    # Rayleigh quotient u^H W v — differentiable in w, with u, v frozen.
    wv = w @ v
    sigma = torch.vdot(u, wv).real if w.is_complex() else torch.dot(u, wv)
    return sigma


@register_loss(
    name="lipschitz_spectral_penalty",
    aliases=["spectral_lipschitz_penalty", "lip_penalty", "lipschitz_penalty"],
    domain="agnostic",
)
class LipschitzSpectralPenalty(nn.Module):
    r"""Hinge penalty :math:`\sum_i\operatorname{ReLU}(\hat\sigma_i-\sigma_{\mathrm{target}})`.

    **DOMAIN**: AGNOSTIC — a parameter penalty, independent of the data domain.

    Args:
        target_sigma: Per-layer spectral-norm budget. Default ``1.0`` (drive the
            backbone toward 1-Lipschitz). The penalty is 0 iff all layers comply.
        n_power_iterations: Power-iteration steps for the differentiable
            ``sigma`` estimate (default ``1``; raise for a tighter estimate).
        eps: Numerical floor for vector normalisation.

    Returns a 0-d tensor: the summed hinge penalty (UN-weighted — the strategy
    multiplies by ``lambda_lipschitz``). Raises if called without ``model=``.
    """

    def __init__(
        self,
        target_sigma: float = 1.0,
        n_power_iterations: int = 1,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if target_sigma <= 0:
            raise ValueError(f"target_sigma must be > 0, got {target_sigma}")
        self.target_sigma = float(target_sigma)
        self.n_power_iterations = int(n_power_iterations)
        self.eps = float(eps)

    def forward(
        self,
        prediction: Any = None,
        target: Any = None,
        *,
        model: nn.Module | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # Resolve the module: explicit model=, else a positional nn.Module (a
        # strategy may pass it first-positionally). NEVER fall back to a (pred,
        # target) reading — raise so a YAML-losses-block misuse fails loudly.
        m = model if isinstance(model, nn.Module) else None
        if m is None and isinstance(prediction, nn.Module):
            m = prediction
        if m is None:
            raise ValueError(
                "LipschitzSpectralPenalty is a PARAMETER penalty and needs the live "
                "nn.Module via model=...; it cannot be used as a (prediction, target) "
                "loss. Wire it through the certified-robustness training strategy, not a "
                "YAML losses block (CLAUDE.md pitfall #16 — inert mechanism)."
            )

        total: torch.Tensor | None = None
        target_sigma = float(kwargs.get("target_sigma", self.target_sigma))
        for mod in m.modules():
            if isinstance(mod, _WEIGHTED_LAYERS):
                w = getattr(mod, "weight", None)
                if w is None or w.numel() == 0:
                    continue
                sigma = _power_iteration_sigma(w, self.n_power_iterations, self.eps)
                hinge = torch.relu(sigma - target_sigma)
                total = hinge if total is None else total + hinge

        if total is None:
            # No weight-bearing layers: a real module with nothing to penalise.
            # Anchor to the module's device (if any) so the term is well-formed.
            ref = next(m.parameters(), None)
            dev = ref.device if ref is not None else None
            return torch.zeros((), device=dev)
        return total
