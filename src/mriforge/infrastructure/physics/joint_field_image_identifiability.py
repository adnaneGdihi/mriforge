r"""Jointly-identifiable field-image rank certificate (A-3.2 *JIFI*).

Joint reconstruction estimates the **image** and the **fields** (B0 / B1 / trajectory)
together. The danger is *confounding*: if a field perturbation can be absorbed by
an image change that produces the same measurements, the two are not separately
identifiable and the joint inverse is ill-posed. JIFI certifies that they are
jointly identifiable from the stacked forward Jacobian
:math:`J=[\,J_{\text{img}}\;|\;J_{\text{field}}\,]`
(:math:`J_{\text{img}}=\partial y/\partial(\text{image})`,
:math:`J_{\text{field}}=\partial y/\partial(\text{field})`).

Two complementary certificates:

* **Full column rank** — :math:`\operatorname{rank}(J)=n_{\text{img}}+n_{\text{field}}`
  (equivalently :math:`\sigma_{\min}(J)>0`) is exactly joint identifiability: no
  direction in parameter space is invisible to the measurement.
* **Field-image confound angle** — the *minimum principal angle*
  :math:`\theta_{\min}` between :math:`\operatorname{col}(J_{\text{img}})` and
  :math:`\operatorname{col}(J_{\text{field}})`. :math:`\theta_{\min}=0` means a
  field direction lies in the image column space (fully confounded);
  :math:`\theta_{\min}=\pi/2` means the two are orthogonal (cleanly separable).
  This is the *graded* identifiability the binary rank cannot express — and it
  explains why **multi-echo multi-flip** acquisitions help: extra echoes give the
  field a distinct phase evolution that rotates :math:`\operatorname{col}(J_{\text{field}})`
  away from :math:`\operatorname{col}(J_{\text{img}})`, raising :math:`\theta_{\min}`.

A **certificate primitive** over forward Jacobians (from the Bloch / multi-echo
simulator or a learned forward), not a training paradigm — the honest scope, like
the sibling FEINF certs.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor


def _orthonormal_basis(j: Tensor, tol: float = 1e-8) -> Tensor:
    """Orthonormal basis ``Q`` of ``col(J)`` (``[n_obs, rank]``) via thin SVD."""
    m = j
    if m.is_complex():
        m = torch.cat([m.real, m.imag], dim=0)
    u, s, _ = torch.linalg.svd(m.double(), full_matrices=False)
    keep = s > (tol * (float(s.max()) if s.numel() else 1.0))
    return u[:, keep]


def field_image_confound_angle(jacobian_image: Tensor, jacobian_field: Tensor) -> float:
    r"""Minimum principal angle (radians) between the image and field column spaces.

    ``0`` ⇒ fully confounded (a field direction lives in the image span);
    :math:`\pi/2` ⇒ orthogonal (cleanly separable). Computed as
    :math:`\arccos(\sigma_{\max}(Q_{\text{img}}^\top Q_{\text{field}}))`.
    """
    qi = _orthonormal_basis(jacobian_image)
    qf = _orthonormal_basis(jacobian_field)
    if qi.shape[1] == 0 or qf.shape[1] == 0:
        return float("nan")
    cross = qi.transpose(-2, -1) @ qf
    smax = float(torch.linalg.svdvals(cross).max().clamp(max=1.0))
    return float(math.acos(smax))


def joint_identifiability_rank(
    jacobian_image: Tensor, jacobian_field: Tensor, tol: float = 1e-8
) -> dict[str, Any]:
    r"""Rank certificate of the stacked joint Jacobian ``[J_img | J_field]``.

    Returns ``{rank, n_params, full_rank, min_singular_value}``. ``full_rank``
    (rank == total params) certifies joint identifiability.
    """
    ji, jf = jacobian_image, jacobian_field
    if ji.is_complex() or jf.is_complex():
        ji = torch.cat([ji.real, ji.imag], dim=0) if ji.is_complex() else torch.cat([ji, ji * 0], 0)
        jf = torch.cat([jf.real, jf.imag], dim=0) if jf.is_complex() else torch.cat([jf, jf * 0], 0)
    joint = torch.cat([ji.double(), jf.double()], dim=1)  # [n_obs, n_img + n_field]
    n_params = joint.shape[1]
    sv = torch.linalg.svdvals(joint)
    smin = float(sv.min()) if sv.numel() else 0.0
    rank = int((sv > tol * (float(sv.max()) if sv.numel() else 1.0)).sum())
    return {
        "rank": rank,
        "n_params": n_params,
        "full_rank": rank == n_params,
        "min_singular_value": smin,
    }


def joint_condition_number(jacobian_image: Tensor, jacobian_field: Tensor) -> float:
    r"""Condition number :math:`\sigma_{\max}/\sigma_{\min}` of ``[J_img | J_field]``.

    ``+inf`` when rank-deficient (non-identifiable); near 1 is well-conditioned.
    """
    ji, jf = jacobian_image, jacobian_field
    if ji.is_complex():
        ji = torch.cat([ji.real, ji.imag], dim=0)
    if jf.is_complex():
        jf = torch.cat([jf.real, jf.imag], dim=0)
    joint = torch.cat([ji.double(), jf.double()], dim=1)
    sv = torch.linalg.svdvals(joint)
    smin = float(sv.min())
    if smin <= 0.0:
        return float("inf")
    return float(sv.max()) / smin


class JointFieldImageIdentifiabilityCert:
    """Joint field-image identifiability certificate bundle (A-3.2 JIFI).

    Reports the rank certificate, the field-image confound angle (graded
    identifiability), the joint condition number, and a boolean ``identifiable``
    verdict (full rank AND the confound angle above a tolerance).
    """

    def __init__(self, angle_tol_rad: float = 1e-3) -> None:
        self.angle_tol_rad = float(angle_tol_rad)

    def certify(self, jacobian_image: Tensor, jacobian_field: Tensor) -> dict[str, Any]:
        """Return ``{rank, full_rank, min_principal_angle, condition_number, identifiable}``."""
        rank_info = joint_identifiability_rank(jacobian_image, jacobian_field)
        angle = field_image_confound_angle(jacobian_image, jacobian_field)
        cond = joint_condition_number(jacobian_image, jacobian_field)
        identifiable = (
            bool(rank_info["full_rank"])
            and (math.isnan(angle) is False)
            and (angle > self.angle_tol_rad)
        )
        return {
            "rank": rank_info["rank"],
            "full_rank": rank_info["full_rank"],
            "min_principal_angle": angle,
            "condition_number": cond,
            "identifiable": identifiable,
        }


__all__ = [
    "JointFieldImageIdentifiabilityCert",
    "field_image_confound_angle",
    "joint_condition_number",
    "joint_identifiability_rank",
]
