r"""Commuting-correction theorem (A-4.4 *CCT*).

MRI reconstruction applies several *corrections* in sequence — motion correction,
B0 / off-resonance unwarping, gradient-nonlinearity correction, coil combination.
Each is an operator on the image (or k-space). **Does the order matter?** CCT
makes this precise and measurable through the operator **commutator**

.. math::

    [A, B]\,x \;=\; A\big(B(x)\big) - B\big(A(x)\big).

**The theorem (exact, not a bound).** The error incurred by applying two
corrections in the *wrong order* is *exactly* the commutator norm:
:math:`\lVert (A\circ B - B\circ A)x\rVert = \lVert [A,B]x\rVert`. Hence two
corrections **commute** — their order is irrelevant, and a pipeline may reorder or
fuse them freely — *iff* :math:`[A,B]=0`; otherwise the commutator quantifies the
order-dependence, and (for a composed correction) bounds the worst-case error a
wrong ordering can introduce: :math:`\lVert[A,B]\rVert\,\lVert x\rVert`.

**Why it bites in MRI.** Pointwise corrections (intensity / phase scaling) commute
with each other but generally **not** with geometric ones: a spatial shift
:math:`S` and a pointwise phase ramp :math:`P` satisfy
:math:`S(P\,x)\neq P(S\,x)` (shifting then multiplying differs from multiplying
then shifting), so motion and B0 correction are order-dependent — exactly the
regime where joint, rather than sequential, correction is justified.

This is an **analytical primitive** (operators are arbitrary callables), validated
on synthetic operator pairs and reusable with the framework's real correction
operators (``KinematicForwardOperator``, ``epi_forward``,
``DifferentiableBlochLayer``, …). It is not a training paradigm.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import Tensor

Operator = Callable[[Tensor], Tensor]


def commutator(op_a: Operator, op_b: Operator, x: Tensor) -> Tensor:
    r"""The commutator :math:`[A,B]x = A(B(x)) - B(A(x))`."""
    return op_a(op_b(x)) - op_b(op_a(x))


def order_swap_error(op_a: Operator, op_b: Operator, x: Tensor) -> float:
    r"""Exact error from swapping correction order: :math:`\lVert(A\circ B - B\circ A)x\rVert`.

    Equal to :math:`\lVert[A,B]x\rVert` by definition — the *theorem*: the
    order-dependence error IS the commutator norm.
    """
    return float(commutator(op_a, op_b, x).norm())


def commuting_defect(op_a: Operator, op_b: Operator, x: Tensor, eps: float = 1e-12) -> float:
    r"""Relative order-dependence :math:`\lVert[A,B]x\rVert / \lVert x\rVert`.

    ``0`` iff the corrections commute on ``x`` (order irrelevant); grows with the
    order-dependence. Scale-invariant in ``x``.
    """
    denom = float(x.norm()) + eps
    return order_swap_error(op_a, op_b, x) / denom


def corrections_commute(op_a: Operator, op_b: Operator, x: Tensor, tol: float = 1e-6) -> bool:
    """True iff the relative commuting defect is within ``tol`` (order irrelevant)."""
    return commuting_defect(op_a, op_b, x) <= tol


class CommutingCorrectionAnalyzer:
    """Analyse whether two correction operators commute (A-4.4 CCT).

    Given two correction operators and a probe signal, reports the commuting
    defect, whether they commute within tolerance, the exact order-swap error, and
    a numerical antisymmetry check (``[A,B] = -[B,A]``). Use it to decide whether a
    correction pipeline may reorder/fuse two stages (commute) or must apply them
    jointly (do not commute).
    """

    def __init__(self, tol: float = 1e-6) -> None:
        self.tol = float(tol)

    def analyze(self, op_a: Operator, op_b: Operator, x: Tensor) -> dict[str, Any]:
        """Return ``{defect, commute, order_swap_error, antisymmetry_residual}``."""
        c_ab = commutator(op_a, op_b, x)
        c_ba = commutator(op_b, op_a, x)
        defect = float(c_ab.norm()) / (float(x.norm()) + 1e-12)
        # [A,B] should equal -[B,A] exactly (definitional) — a numerical sanity check.
        antisym = float((c_ab + c_ba).norm()) / (float(c_ab.norm()) + 1e-12)
        return {
            "defect": defect,
            "commute": defect <= self.tol,
            "order_swap_error": float(c_ab.norm()),
            "antisymmetry_residual": antisym,
        }


__all__ = [
    "CommutingCorrectionAnalyzer",
    "commutator",
    "commuting_defect",
    "corrections_commute",
    "order_swap_error",
]
