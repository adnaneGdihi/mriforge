"""Pathology Recall Certificate (ULF-PR-23 / R4).

Plan: ``TODO/backlog_ulf_radiologist_acceptance_roadmap.md`` ULF-PR-23.

Implements Theorem R4 — the Hoeffding lower bound on recall:

.. math::

    r \\;\\geq\\; \\hat r - \\sqrt{\\frac{\\ln(1/\\delta)}{2n}}

with probability at least :math:`1 - \\delta` over the test set of
size ``n`` and empirical recall :math:`\\hat r`. The certificate is
the *guaranteed* lower bound; clinical claims must use this number,
not the raw empirical recall.

Operational reality (from the spec doc): at :math:`\\delta = 0.05`,
:math:`n = 200`, the Hoeffding gap is roughly :math:`0.087`. To
certify "recall ≥ 0.90 at :math:`\\delta = 0.05`" the empirical
recall must be ≥ 0.987 — this is the *right* bar for clinical claims
and the certificate refuses to weaken it.

The frozen-detector orchestration (depends on paradigm-expansion
roadmap PR-5 ``IDownstreamModelService``) is deferred. This module
provides the math + dataclass + audit hook that the use-case will
consume once it lands.

Reference: Hoeffding (1963), *Probability Inequalities for Sums of
Bounded Random Variables*, JASA.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


def hoeffding_recall_lower_bound(
    empirical_recall: float,
    n: int,
    delta: float,
) -> float:
    """Return the one-sided Hoeffding lower bound on recall.

    Args:
        empirical_recall: Observed recall on the test set, in ``[0, 1]``.
        n: Test-set size (positive samples only — Hoeffding bounds the
            estimator of the *positive-class* recall).
        delta: Confidence parameter — the bound holds with probability
            at least ``1 - delta``. Typical clinical value: ``0.05``.

    Returns:
        The guaranteed lower bound, clipped to ``[0, 1]``.

    Raises:
        ValueError: If inputs are out of range.
    """
    if not 0.0 <= empirical_recall <= 1.0:
        raise ValueError(f"empirical_recall must be in [0, 1], got {empirical_recall}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    gap = math.sqrt(math.log(1.0 / delta) / (2.0 * n))
    return max(0.0, empirical_recall - gap)


def minimum_n_for_target_recall(
    target_recall: float,
    empirical_recall: float,
    delta: float,
) -> int:
    """Return the smallest ``n`` that certifies ``target_recall`` at ``delta``.

    Solves :math:`\\hat r - \\sqrt{\\ln(1/\\delta)/(2n)} \\ge r^*`
    for ``n``. Used by the Tier-1 audit
    ``pathology_test_set_size_sufficient`` to refuse certification when
    the test set is too small for the desired claim.

    Args:
        target_recall: The recall threshold to certify (e.g. ``0.90``).
        empirical_recall: Observed recall (the gap shrinks as this grows).
        delta: Confidence parameter (e.g. ``0.05``).

    Returns:
        Smallest integer ``n`` such that the bound certifies
        ``target_recall``. Returns ``math.inf`` if the empirical recall
        is too low to ever certify the target.
    """
    if empirical_recall <= target_recall:
        return math.inf  # type: ignore[return-value]
    gap_max = empirical_recall - target_recall
    # ln(1/delta) / (2n) <= gap_max**2  ==>  n >= ln(1/delta) / (2 * gap_max**2)
    n_required = math.log(1.0 / delta) / (2.0 * gap_max**2)
    return int(math.ceil(n_required))


@dataclass(frozen=True)
class PathologyRecallCertificate:
    """Per-class pathology-recall certificate.

    The full bundle ULF-PR-27 (regulatory CLI) emits one entry per
    pathology class. ``certified=True`` is the only flag clinical
    deployment may rely on.
    """

    pathology_class: str
    n: int
    empirical_recall: float
    delta: float
    target_recall: float
    certified_lower_bound: float
    certified: bool
    minimum_n_for_target: int

    def to_dict(self) -> dict:
        return asdict(self)


def build_pathology_recall_certificate(
    pathology_class: str,
    *,
    empirical_recall: float,
    n: int,
    delta: float = 0.05,
    target_recall: float = 0.90,
) -> PathologyRecallCertificate:
    """Compute the Hoeffding-bound certificate for one pathology class.

    Args:
        pathology_class: Class name (e.g. ``"lesion"``, ``"tumor"``).
        empirical_recall: Observed recall.
        n: Test-set size for this class.
        delta: Confidence (default 0.05 — clinical standard).
        target_recall: Threshold the certificate must clear (default 0.90).

    Returns:
        Structured certificate with the certified lower bound and a
        ``certified`` flag.
    """
    bound = hoeffding_recall_lower_bound(empirical_recall, n, delta)
    min_n = minimum_n_for_target_recall(target_recall, empirical_recall, delta)
    return PathologyRecallCertificate(
        pathology_class=pathology_class,
        n=n,
        empirical_recall=empirical_recall,
        delta=delta,
        target_recall=target_recall,
        certified_lower_bound=bound,
        certified=bound >= target_recall,
        minimum_n_for_target=min_n if isinstance(min_n, int) else -1,
    )


@dataclass(frozen=True)
class TestSetSizeAudit:
    """Outcome of the Tier-1 ``pathology_test_set_size_sufficient`` audit."""

    passed: bool
    pathology_class: str
    n: int
    required_n: int
    message: str


def pathology_test_set_size_sufficient(
    pathology_class: str,
    empirical_recall: float,
    n: int,
    target_recall: float = 0.90,
    delta: float = 0.05,
) -> TestSetSizeAudit:
    """Tier-1 audit — refuses certification when ``n`` is too small.

    Catches the operational pitfall described in the spec doc: at
    ``delta=0.05, n=200``, certifying ``recall >= 0.90`` requires
    empirical recall ≥ ~0.987. A test set that's too small can't
    certify the target no matter how perfect the detector — the audit
    surfaces this loudly.
    """
    required = minimum_n_for_target_recall(target_recall, empirical_recall, delta)
    if isinstance(required, float):  # math.inf — empirical_recall <= target
        return TestSetSizeAudit(
            passed=False,
            pathology_class=pathology_class,
            n=n,
            required_n=-1,
            message=(
                f"{pathology_class}: empirical recall {empirical_recall:.4f} ≤ "
                f"target {target_recall:.4f}; certificate impossible at any n."
            ),
        )
    passed = n >= required
    return TestSetSizeAudit(
        passed=passed,
        pathology_class=pathology_class,
        n=n,
        required_n=required,
        message=(
            f"{pathology_class}: n={n}, required={required} for "
            f"recall ≥ {target_recall:.2f} at δ={delta:.2f}. "
            f"{'PASS' if passed else 'FAIL — increase test-set size'}."
        ),
    )


def write_certificates_json(
    certificates: list[PathologyRecallCertificate],
    out_path: str | Path,
) -> Path:
    """Write the per-class certificate list as JSON for the regulatory bundle.

    Produces the artifact ULF-PR-27's regulatory CLI expects at
    ``reports/<exp>/pathology_certificates.json``.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([c.to_dict() for c in certificates], indent=2))
    return out


__all__ = [
    "PathologyRecallCertificate",
    "TestSetSizeAudit",
    "build_pathology_recall_certificate",
    "hoeffding_recall_lower_bound",
    "minimum_n_for_target_recall",
    "pathology_test_set_size_sufficient",
    "write_certificates_json",
]
