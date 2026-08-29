r"""Equivariant motion-state resolution (A-4.3 *EMR*).

A motion-resolving / motion-correcting operator :math:`R` should treat every motion
state the same way: resolving an object and *then* moving it must equal moving it
and *then* resolving — i.e. :math:`R` must be **equivariant** to the motion group
:math:`\mathcal{G}` (rigid translations / rotations),

.. math::

    R(T_g\,x) \;=\; T_g\,R(x)\qquad\forall g\in\mathcal{G}.

EMR quantifies the violation as the **motion-equivariance defect**

.. math::

    \mathrm{EMR}(R)=\frac{1}{|\mathcal{G}|}\sum_{g}
        \frac{\lVert R(T_g x) - T_g R(x)\rVert}{\lVert R(x)\rVert},

which is ``0`` for a genuinely motion-equivariant operator (e.g. a convolution is
*exactly* translation-equivariant under circular shift) and grows for one that
bakes in a fixed spatial frame. This is B-1.4's steerable-equivariance idea applied
to the *motion* group; it reuses the commutator viewpoint of A-4.4 CCT
(:math:`R(T_g x)-T_g R(x)=[R,T_g]x`) specialised to group actions.

A certificate primitive over arbitrary operator + transform callables (composes
with the framework's SE(2)/SE(3) motion machinery), not a training paradigm.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

Operator = Callable[[Tensor], Tensor]


def translation(dy: int, dx: int) -> Operator:
    """Rigid (circular) translation by ``(dy, dx)`` pixels — a motion-group action."""
    return lambda x: torch.roll(x, shifts=(int(dy), int(dx)), dims=(-2, -1))


def rotation_90(k: int) -> Operator:
    """Rigid rotation by ``k*90`` degrees — a motion-group action (exact on the grid)."""
    return lambda x: torch.rot90(x, k=int(k), dims=(-2, -1))


def motion_equivariance_defect(
    operator: Operator, transforms: list[Operator], image: Tensor, eps: float = 1e-12
) -> float:
    r"""Mean relative motion-equivariance defect of ``operator`` over ``transforms``.

    :math:`\operatorname{mean}_g \lVert R(T_g x)-T_g R(x)\rVert/\lVert R(x)\rVert`.
    ``0`` ⇒ motion-equivariant; larger ⇒ more motion-state-dependent. Scale-invariant
    in ``image``.
    """
    if not transforms:
        raise ValueError("transforms must be a non-empty list of motion-group actions")
    rx = operator(image)
    denom = float(rx.norm()) + eps
    total = 0.0
    for t in transforms:
        lhs = operator(t(image))  # R(T_g x)
        rhs = t(rx)  # T_g R(x)
        total += float((lhs - rhs).norm()) / denom
    return total / len(transforms)


def is_motion_equivariant(
    operator: Operator, transforms: list[Operator], image: Tensor, tol: float = 1e-5
) -> bool:
    """True iff the motion-equivariance defect is within ``tol``."""
    return motion_equivariance_defect(operator, transforms, image) <= tol


class MotionEquivarianceResolver:
    """Certify a motion-resolving operator's equivariance to a motion group (A-4.3 EMR).

    Given an operator and a set of motion transforms (translations / rotations),
    reports the per-transform defects, the mean defect, and a boolean
    ``equivariant`` verdict — the certificate that the operator resolves motion
    states consistently.
    """

    def __init__(self, tol: float = 1e-5) -> None:
        self.tol = float(tol)

    def certify(
        self, operator: Operator, transforms: list[Operator], image: Tensor
    ) -> dict[str, Any]:
        """Return ``{per_transform_defect, mean_defect, equivariant}``."""
        rx = operator(image)
        denom = float(rx.norm()) + 1e-12
        per = [float((operator(t(image)) - t(rx)).norm()) / denom for t in transforms]
        mean_defect = sum(per) / len(per) if per else float("nan")
        return {
            "per_transform_defect": per,
            "mean_defect": mean_defect,
            "equivariant": bool(mean_defect <= self.tol),
        }


__all__ = [
    "MotionEquivarianceResolver",
    "is_motion_equivariant",
    "motion_equivariance_defect",
    "rotation_90",
    "translation",
]
