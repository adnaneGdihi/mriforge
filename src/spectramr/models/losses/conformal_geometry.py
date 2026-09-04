"""Conformal-geometry losses for the SFC / Beltrami / Teichmüller
directions of ``TODO/audit_plan_novel.md`` §§194-344.

Registered losses:
    * ``conformal_modulus`` — an **image-content distortion penalty** over
      sampled quadrilateral regions (§3 / §5 of the plan). NB: despite the
      name it does not compute a conformal modulus — the quad geometry never
      enters the harmonic solve; see :class:`ConformalModulusLoss`.
    * ``beltrami_motion_residual`` — plain L1 data-consistency residual for a
      Beltrami-parameterised motion field (the Beltrami math lives upstream in
      the solver, not in this loss) (§4).
    * ``teichmuller_geodesic`` — schedule-smoothness term that
      penalises non-geodesic Teichmüller schedules in cold diffusion
      (§6).

References (numbers match the plan):
    [7] N. Papamichael & N. Stylianopoulos, *Numerical Conformal
        Mapping*, World Scientific, 2010.
    [8] L. M. Lui et al., "Teichmüller mapping (T-map)", *SIAM J.
        Imag. Sci.*, 7(1), 2014.
    [9] F. P. Gardiner & N. Lakic, *Quasiconformal Teichmüller
        Theory*, AMS, 2000.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.infrastructure.physics.conformal_geometry import (
    conformal_modulus_quadrilateral,
    teichmuller_dilatation,
)
from spectramr.models.losses.registry import register_loss


@register_loss("conformal_modulus", domain="image")
class ConformalModulusLoss(nn.Module):
    """Conformal-modulus distortion penalty over quadrilateral regions.

    Since 2026-07 :func:`~spectramr.infrastructure.physics.conformal_geometry.conformal_modulus_quadrilateral`
    computes a **genuine** conformal modulus: the Dirichlet energy of the
    harmonic field solved on the *physical* quad by isoparametric FE, so the
    value depends on the quad shape (a square → ``≈1``, a 3:1 rectangle →
    ``≈3``). It is then scaled by the mean image-graph metric
    ``w = sqrt(1 + |∇I|²)`` over the quad, so it also tracks image content.

    .. math::
        \\mathcal{L}_{\\text{mod}}
        = \\mathbb{E}_{Q}\\Bigl|\\log\\, \\mathrm{mod}(R_\\theta(x)(Q))
        - \\log\\, \\mathrm{mod}(x^{\\text{HF}}(Q))\\Bigr|

    The penalty is computed **per image**: the modulus is scaled by each image's
    graph metric sampled on the quad region, so the two terms genuinely differ
    when the reconstruction distorts anatomy — a differentiable distortion
    penalty. (Before 2026-06 both were computed on the same canonical square
    ignoring the images, so the residual was identically zero — an inert loss,
    pitfall #20; the 2026-07 change additionally makes the base modulus
    shape-aware rather than a fixed canonical value.)

    Quadrilateral templates (cyclic corners in normalised ``[-1, 1]`` image
    coordinates) may be supplied via ``context['quad_corners']`` as ``[K, 4, 2]``;
    each template is applied to every image in the batch. When none are supplied
    the loss falls back to a built-in multi-scale set (the full frame plus four
    quadrants), so it is non-degenerate without any caller wiring.
    """

    _EPS = 1.0e-8

    def __init__(self, weight: float = 1.0, *, grid_size: int = 16) -> None:
        super().__init__()
        self.weight = weight
        self.grid_size = grid_size

    @staticmethod
    def _default_quads(ref: torch.Tensor) -> torch.Tensor:
        """Built-in multi-scale quad templates in normalised ``[-1, 1]`` coords."""
        templates = [
            [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],  # full frame
            [[-1.0, -1.0], [0.0, -1.0], [0.0, 0.0], [-1.0, 0.0]],  # top-left
            [[0.0, -1.0], [1.0, -1.0], [1.0, 0.0], [0.0, 0.0]],  # top-right
            [[-1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [-1.0, 1.0]],  # bottom-left
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],  # bottom-right
        ]
        return torch.tensor(templates, dtype=ref.dtype, device=ref.device)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **context: object,
    ) -> dict[str, torch.Tensor]:
        quads = context.get("quad_corners")
        if isinstance(quads, torch.Tensor):
            templates = quads if quads.dim() == 3 else quads.unsqueeze(0)
        else:
            templates = self._default_quads(pred)
        batch = pred.shape[0]
        residuals = []
        for k in range(templates.shape[0]):
            qk = templates[k].unsqueeze(0).expand(batch, 4, 2).contiguous()
            mod_pred = conformal_modulus_quadrilateral(qk, grid_size=self.grid_size, image=pred)
            mod_tgt = conformal_modulus_quadrilateral(qk, grid_size=self.grid_size, image=target)
            residuals.append(
                (mod_pred.clamp_min(self._EPS).log() - mod_tgt.clamp_min(self._EPS).log()).abs()
            )
        residual = torch.stack(residuals).mean()
        return {
            "loss": self.weight * residual,
            "conformal_modulus": residual,
        }


@register_loss("beltrami_motion_residual", domain="image")
class BeltramiMotionResidual(nn.Module):
    """Data-consistency residual for a Beltrami-parameterised motion field.

    Given a moving image ``pred = x ∘ φ`` (where ``φ`` is the
    quasi-conformal map recovered from a Beltrami coefficient ``μ``
    via :class:`LinearBeltramiSolver`) and a fixed reference
    ``target``, return the L1 residual. Beltrami-constraint
    enforcement is delegated to the solver / activation, not the
    loss.

    NB: this loss itself is **plain L1** — it contains no Beltrami or conformal
    math; the name reflects the field it is paired with, not the loss content.
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **context: object,
    ) -> dict[str, torch.Tensor]:
        residual = F.l1_loss(pred, target)
        return {
            "loss": self.weight * residual,
            "beltrami_motion_residual": residual,
        }


@register_loss(
    "teichmuller_geodesic",
    domain="latent",
    compatible_with=["image", "kspace", "complex"],
)
class TeichmullerGeodesicRegularisation(nn.Module):
    """Constant-speed Teichmüller geodesic schedule regulariser (§6).

    .. math::
        \\mathcal{L}_{\\text{teich}}
        = \\int_0^T \\bigl(\\tfrac{d}{dt}\\log K(t)\\bigr)^2\\,dt

    where :math:`K(t) = (1+r(t))/(1-r(t))` is the maximal dilatation
    of the Beltrami coefficient at diffusion step ``t``. Caller
    supplies ``r_schedule`` as a 1-D tensor sampled densely along
    ``[0, T]``.
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **context: object,
    ) -> dict[str, torch.Tensor]:
        r = context.get("r_schedule")
        if not isinstance(r, torch.Tensor) or r.numel() < 3:
            # Fail loud: returning a zero dict makes this registered loss an
            # inert no-op whenever the schedule is not threaded through (a
            # facade — pitfall #16). The caller MUST supply a dense 1-D
            # ``r_schedule`` (>= 3 samples along [0, T]); if a run genuinely has
            # no schedule, do not add this loss term.
            raise ValueError(
                "TeichmullerGeodesicRegularisation requires a 1-D 'r_schedule' "
                "tensor with >= 3 samples in context; got "
                f"{type(r).__name__ if not isinstance(r, torch.Tensor) else f'numel={r.numel()}'}."
            )
        r_clamped = r.clamp(0.0, 0.999)
        log_k = ((1.0 + r_clamped) / (1.0 - r_clamped)).log()
        dlog_k = log_k[1:] - log_k[:-1]
        residual = dlog_k.pow(2).mean()
        return {
            "loss": self.weight * residual,
            "teichmuller_geodesic": residual,
        }


def schedule_dilatation(r_schedule: torch.Tensor) -> torch.Tensor:
    """Convenience: per-step dilatation of a Teichmüller schedule."""
    return teichmuller_dilatation(r_schedule.unsqueeze(-1).unsqueeze(-1) + 0j)


__all__ = [
    "BeltramiMotionResidual",
    "ConformalModulusLoss",
    "TeichmullerGeodesicRegularisation",
    "schedule_dilatation",
]
