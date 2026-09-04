r"""Conditional Flow Matching (CFM) and Rectified Flow training losses.

Both are direct regression objectives on a *velocity field* that
transports a base distribution :math:`\mathbf{x}_0\sim p_0` to the data
distribution :math:`\mathbf{x}_1\sim p_{\text{data}}`. They differ
only in how the (training) endpoints :math:`(\mathbf{x}_0,\mathbf{x}_1)`
are paired and what the conditioning variable :math:`t` parameterises.

* **Conditional flow matching** (Lipman *et al.* 2023): for each
  example sample :math:`t\sim U(0,1)` and a base point
  :math:`\mathbf{x}_0\sim\mathcal{N}(0,\mathbf{I})`. The conditional
  probability path is the straight line
  :math:`\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1`, whose
  velocity is :math:`\mathbf{x}_1-\mathbf{x}_0`. The loss is

  .. math::
      \mathcal{L}_{\text{CFM}} = \mathbb{E}_{t,\mathbf{x}_0,\mathbf{x}_1}\,
        \|v_{\boldsymbol{\theta}}(\mathbf{x}_t,t)-(\mathbf{x}_1-\mathbf{x}_0)\|^2.

* **Rectified flow** (Liu, Gong, Liu 2023): same loss form but with
  :math:`(\mathbf{x}_0,\mathbf{x}_1)` taken from a previously-trained
  flow's deterministic ODE pairing rather than independent draws —
  *straightens* the trajectories, enabling 1- or 2-step generation.
  The rectified-flow loss therefore behaves identically *given* the
  paired sample, so we re-export the same implementation under a
  separate registry entry to make the choice explicit in YAML.

References:
    [35] Y. Lipman *et al.*, ICLR 2023.
    [36] A. Tong *et al.*, "Improving and generalizing flow-based
        generative models with minibatch optimal transport," TMLR, 2024.
    [37] X. Liu, C. Gong, Q. Liu, ICLR 2023.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss


def _sample_t(batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Uniform t ~ U(0, 1) of shape (B, 1, 1, 1) (4D) or (B, 1, 1, 1, 1) (5D)
    is left to the caller to broadcast — we return shape (B,)."""
    return torch.rand(batch, device=device, dtype=dtype)


def _broadcast_t(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    shape = (t.shape[0],) + (1,) * (like.dim() - 1)
    return t.view(shape)


def _flow_matching_loss(
    velocity_pred: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
    norm: str,
) -> torch.Tensor:
    """Generic v-regression at the linear-interpolant midpoint."""
    target_v = x1 - x0  # straight-line velocity field
    if velocity_pred.shape != target_v.shape:
        raise ValueError(
            f"velocity prediction shape {tuple(velocity_pred.shape)} != "
            f"target velocity shape {tuple(target_v.shape)}"
        )
    _ = t  # t already used by caller to construct x_t
    if norm == "l1":
        return F.l1_loss(velocity_pred, target_v)
    return F.mse_loss(velocity_pred, target_v)


@register_loss(
    name="conditional_flow_matching",
    aliases=["cfm", "CFMLoss", "flow_matching"],
    domain="image",
)
class ConditionalFlowMatchingLoss(nn.Module):
    r"""Velocity-regression loss for conditional flow matching.

    Forward expects:

    * ``prediction``: velocity prediction :math:`v_{\boldsymbol{\theta}}
      (\mathbf{x}_t, t)`. Same shape as the data tensor.
    * ``target``: the data sample :math:`\mathbf{x}_1`. Same shape.
    * ``context["x0"]``: optional base sample :math:`\mathbf{x}_0`. If
      missing, drawn from :math:`\mathcal{N}(0,\mathbf{I})` matching
      ``target`` shape.
    * ``context["t"]``: optional time scalar. If missing, sampled
      :math:`t\sim U(0,1)` per example.

    Args:
        norm: ``"l2"`` (default, paper) or ``"l1"``.
    """

    def __init__(self, norm: str = "l2") -> None:
        super().__init__()
        if norm not in ("l1", "l2"):
            raise ValueError(f"norm must be l1 or l2, got {norm!r}")
        self.norm = norm

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        context: dict | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        ctx = context or {}
        x0 = ctx.get("x0")
        if x0 is None:
            x0 = torch.randn_like(target)
        t = ctx.get("t")
        if t is None:
            t = _sample_t(target.shape[0], target.device, target.dtype)
        return _flow_matching_loss(prediction, x0, target, t, self.norm)


@register_loss(
    name="rectified_flow",
    aliases=["RectifiedFlowLoss", "reflow"],
    domain="image",
)
class RectifiedFlowLoss(nn.Module):
    r"""Rectified-flow velocity loss (paper-identical to CFM regression
    but with caller-supplied straight-line pairs).

    The strategy is responsible for sampling :math:`(\mathbf{x}_0,
    \mathbf{x}_1)` from the previous flow's deterministic ODE and
    passing them via context. Falls back to independent draws if not
    supplied (in which case it equals CFM).

    Args:
        norm: ``"l2"`` (default) or ``"l1"``.
    """

    def __init__(self, norm: str = "l2") -> None:
        super().__init__()
        if norm not in ("l1", "l2"):
            raise ValueError(f"norm must be l1 or l2, got {norm!r}")
        self.norm = norm

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        context: dict | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        ctx = context or {}
        x0 = ctx.get("x0")
        if x0 is None:
            x0 = torch.randn_like(target)
        t = ctx.get("t")
        if t is None:
            t = _sample_t(target.shape[0], target.device, target.dtype)
        return _flow_matching_loss(prediction, x0, target, t, self.norm)


def _hungarian_pair(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Find an optimal-transport pairing of (x0, x1) within the batch.

    Returns a permutation ``perm`` such that ``x1[perm]`` is the
    minimum-cost pairing under squared-Euclidean cost. Uses the
    Hungarian algorithm via ``scipy.optimize.linear_sum_assignment``
    on the detached cost matrix (the loss itself remains
    differentiable through the resulting straight-line velocity).
    """
    from scipy.optimize import linear_sum_assignment  # local import — keep startup cheap

    flat_x0 = x0.reshape(x0.shape[0], -1)
    flat_x1 = x1.reshape(x1.shape[0], -1)
    cost = torch.cdist(flat_x0, flat_x1).pow(2).detach().cpu().numpy()
    _, col = linear_sum_assignment(cost)
    return torch.as_tensor(col, device=x0.device, dtype=torch.long)


@register_loss(
    name="ot_conditional_flow_matching",
    aliases=["ot_cfm", "OTCFMLoss", "ot_flow_matching"],
    domain="image",
)
class OTConditionalFlowMatchingLoss(nn.Module):
    r"""CFM with minibatch optimal-transport pairing of (x_0, x_1).

    Identical loss form to plain CFM, but the noise samples
    :math:`\mathbf{x}_0` are re-paired with the data samples
    :math:`\mathbf{x}_1` using the Hungarian algorithm on the
    minibatch cost matrix :math:`C_{ij} = \|x_{0,i}-x_{1,j}\|^2`.
    The OT pairing produces *straighter* probability paths, which in
    practice lets ODE samplers integrate in many fewer steps.

    Forward expects:

    * ``prediction``: velocity prediction at the (re-paired)
      :math:`\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1`. The
      loss does **not** itself rebuild :math:`\mathbf{x}_t` — the
      strategy must build it after the OT pairing is returned via the
      auxiliary ``permutation`` (see Notes). This is the canonical
      stateless form: the strategy gets ``permutation``, indexes
      :math:`\mathbf{x}_0` accordingly, builds :math:`\mathbf{x}_t`,
      runs the network, and hands the result back as ``prediction``.
    * ``target``: data sample :math:`\mathbf{x}_1`.
    * ``context["x0"]``: optional :math:`\mathbf{x}_0` (else drawn).

    Notes:
        For convenience, when ``context["return_permutation"] = True``
        the loss attaches the chosen permutation to
        ``context["permutation"]`` so the strategy can reuse it. The
        loss assumes the strategy has already built
        :math:`\mathbf{x}_t` using the *same* permutation that the OT
        solve returns — when ``context`` provides a pre-computed
        ``permutation`` we honour it; otherwise we Hungarian-pair on
        the fly and the strategy is expected to have done the same.

    Args:
        norm: ``"l2"`` (default) or ``"l1"``.
    """

    def __init__(self, norm: str = "l2") -> None:
        super().__init__()
        if norm not in ("l1", "l2"):
            raise ValueError(f"norm must be l1 or l2, got {norm!r}")
        self.norm = norm

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        context: dict | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        ctx = context if context is not None else {}
        x0 = ctx.get("x0")
        if x0 is None:
            x0 = torch.randn_like(target)
        perm = ctx.get("permutation")
        if perm is None:
            perm = _hungarian_pair(x0, target)
            if context is not None and ctx.get("return_permutation"):
                ctx["permutation"] = perm
        x0_paired = x0[perm]
        t = ctx.get("t")
        if t is None:
            t = _sample_t(target.shape[0], target.device, target.dtype)
        return _flow_matching_loss(prediction, x0_paired, target, t, self.norm)


# Keep helper visible
_ = _broadcast_t
