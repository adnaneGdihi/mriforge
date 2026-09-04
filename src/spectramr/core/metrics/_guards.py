"""Shared validation guards for field/trajectory metrics.

The single anti-facade primitive for the field-domain metrics: a tested
predicate that decides whether an *estimate* and a *reference* are the same
physical quantity and may therefore be graded against one another. It blocks
the two pitfall-#16 facades the VF program is exposed to:

* an image (a corrected bSSFP frame) graded as a B0 **field** in Hz, and
* a Cartesian per-readout-line ``Δk`` graded as a **spiral** ``Δk(t)``.

A metric that grades incomparable quantities smoke-PASSes while measuring
nothing. Centralising the precondition here means one audited guard protects
``b0_field_rmse``, ``k_space_trajectory_rmse``, and every future field/trajectory
metric. Callers grade only when the verdict is ``ok``; a metric returns
``float("nan")`` and a strategy scoring seam returns ``{}`` otherwise — a
graceful skip, never a silent grade (the facade) and never a hard raise (which
would break unrelated validation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ComparabilityVerdict:
    """Whether (estimate, reference) may be graded against one another.

    ``ok`` is the gate; ``reason`` is a non-empty human-readable explanation
    (logged on a skip so a silent ``nan``/``{}`` is always traceable).
    """

    ok: bool
    reason: str


def field_comparability(
    estimate: Any | None,
    reference: Any | None,
    *,
    estimate_kind: str,
    reference_kind: str,
    estimate_units: str,
    reference_units: str,
) -> ComparabilityVerdict:
    """Decide whether an estimate and reference are the same physical quantity.

    Implements the spec predicate ``comparable(u, v)`` = (shape(u) == shape(v))
    AND (kind_u == kind_v) AND (units_u == units_v): both tensors present,
    identical declared *kind* (``"field"`` / ``"image"`` / ``"trajectory"`` ...),
    identical declared *units* (``"Hz"`` / ``"cycles_per_fov"`` ...), and
    identical tensor shape
    (absolute metrics like an RMSE require an exact element-wise correspondence,
    so the caller must bring both tensors into the same shape *before* grading).

    Returns a :class:`ComparabilityVerdict`; never raises on a mismatch — the
    caller skips gracefully.
    """
    if estimate is None or reference is None:
        which = "estimate" if estimate is None else "reference"
        return ComparabilityVerdict(False, f"missing {which}")
    if estimate_kind != reference_kind:
        return ComparabilityVerdict(
            False, f"kind mismatch: {estimate_kind!r} vs {reference_kind!r}"
        )
    if estimate_units != reference_units:
        return ComparabilityVerdict(
            False, f"units mismatch: {estimate_units!r} vs {reference_units!r}"
        )
    est_shape = tuple(getattr(estimate, "shape", ()))
    ref_shape = tuple(getattr(reference, "shape", ()))
    if est_shape != ref_shape:
        return ComparabilityVerdict(False, f"shape mismatch: {est_shape} vs {ref_shape}")
    return ComparabilityVerdict(True, "comparable")
