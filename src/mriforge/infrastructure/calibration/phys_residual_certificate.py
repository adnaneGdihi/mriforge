r"""Physics-Residual Conformal hallucination certificate (PR-CC).

A split-conformal certificate whose non-conformity score is the *physics
residual*: re-render the observed contrast from the estimated tissue-parameter
map through the Bloch forward and score :math:`s(v)=|\mathcal A(\hat{\boldsymbol
\xi};\boldsymbol\varphi,B_0)(v)-\mathbf y(v)|`. Because :math:`\mathcal A` (the
Bloch relation) holds at every field, the score is anchored to a *field-invariant*
model, so its validity as a conformal score requires only residual
exchangeability rather than image-distribution exchangeability -- the property
that lets the certificate remain meaningful across fields.

Procedure (Vovk-Gammerman-Shafer split conformal):

1. Compute physics residuals on a held-out calibration split.
2. Take the finite-sample :math:`\lceil(n+1)(1-\alpha)\rceil`-th order statistic
   :math:`\hat q` (delegated to :class:`ConformalCalibrator`, which honestly
   returns :math:`+\infty` when :math:`n` is too small to certify).
3. At test time flag the *consistent* set :math:`C(\mathbf y)=\{v:s(v)\le\hat q\}`
   and the complementary hallucination map :math:`\{v:s(v)>\hat q\}`, with
   :math:`\mathbb P(s(v_{\mathrm{test}})\le\hat q)\ge 1-\alpha`.

When an unseen field shifts the residual distribution (e.g. via SNR scaling),
marginal coverage degrades. The remedy is **Mondrian (field-stratified)
calibration**: fit one calibrator per :math:`B_0` group (``stratify_by="b0"``),
restoring group-conditional coverage.

References
----------
Vovk, Gammerman, Shafer (2005), *Algorithmic Learning in a Random World*.
Angelopoulos & Bates (2021), *A gentle introduction to conformal prediction*.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from mriforge.infrastructure.calibration.conformal import ConformalCalibrator
from mriforge.infrastructure.calibration.scores import physics_residual


@dataclass(frozen=True)
class CertificateResult:
    """Outcome of certifying one test unit against the fitted certificate.

    Attributes:
        coverage: Fraction of voxels with ``s(v) <= q`` (should be ``>= 1-alpha``).
        quantile: The fitted band radius ``q`` for the unit's field group.
        flagged: Boolean hallucination map ``s(v) > q`` (same shape as the score).
        score: The per-voxel physics residual ``s(v)``.
    """

    coverage: float
    quantile: float
    flagged: torch.Tensor
    score: torch.Tensor


class PhysResidualConformalCertificate:
    """Split-conformal hallucination certificate on the physics residual.

    Args:
        alpha: Mis-coverage rate ``alpha in (0, 1)``.
        acquisition: Sequence parameters ``{"TR","TE","TI"}`` (ms) for the
            re-render. Required -- a physics residual without acquisition is
            undefined (no silent fallback).
        contrast_type: Contrast label routed to the Bloch forward.
        stratify_by: ``None`` for a single marginal calibrator, or ``"b0"`` for
            Mondrian (per-field) calibration.

    Raises:
        ValueError: ``alpha`` outside ``(0, 1)``; missing ``acquisition``;
            ``stratify_by`` not in ``{None, "b0"}``.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        *,
        acquisition: Mapping[str, float] | None = None,
        contrast_type: str = "spin_echo",
        stratify_by: str | None = None,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}.")
        if acquisition is None:
            raise ValueError(
                "PhysResidualConformalCertificate requires `acquisition` "
                "(TR/TE/TI in ms) for the physics re-render."
            )
        if stratify_by not in (None, "b0"):
            raise ValueError(f"stratify_by must be None or 'b0', got {stratify_by!r}.")
        self._alpha = float(alpha)
        self._acquisition = dict(acquisition)
        self._contrast_type = str(contrast_type)
        self._stratify_by = stratify_by
        self._calibrators: dict[Hashable, ConformalCalibrator] = {}

    # ------------------------------------------------------------------
    def _score(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return physics_residual(
            prediction,
            target,
            acquisition=self._acquisition,
            contrast_type=self._contrast_type,
        )

    def _group_key(self, b0: Any) -> Hashable:
        return b0 if self._stratify_by == "b0" else None

    @property
    def is_calibrated(self) -> bool:
        return bool(self._calibrators)

    def calibrate(
        self,
        units: Iterable[tuple[torch.Tensor, torch.Tensor, Any]],
    ) -> PhysResidualConformalCertificate:
        """Fit the (per-group) calibrator(s) from ``(prediction, target, b0)`` units.

        Args:
            units: Iterable of ``(prediction, target, b0)``. ``b0`` is the field
                strength used as the Mondrian group key when ``stratify_by="b0"``
                (ignored otherwise).

        Returns:
            Self (chainable).

        Raises:
            ValueError: when no calibration units are supplied.
        """
        groups: dict[Hashable, list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
        for prediction, target, b0 in units:
            groups[self._group_key(b0)].append((prediction, target))
        if not groups:
            raise ValueError("calibrate() needs at least one (pred, target, b0) unit.")
        for key, pairs in groups.items():
            calibrator = ConformalCalibrator(score_fn=self._score, alpha=self._alpha)
            calibrator.fit(pairs)
            self._calibrators[key] = calibrator
        return self

    def quantile(self, b0: Any = None) -> float:
        """Fitted band radius ``q`` for the field group (raises if unfit/unseen)."""
        return self._calibrator_for(b0).quantile

    def _calibrator_for(self, b0: Any) -> ConformalCalibrator:
        if not self._calibrators:
            raise RuntimeError(
                "PhysResidualConformalCertificate is not calibrated. Call calibrate(...) first."
            )
        key = self._group_key(b0)
        calibrator = self._calibrators.get(key)
        if calibrator is None:
            raise KeyError(
                f"No calibrator for field group {key!r}. Under Mondrian "
                f"stratification each B0 must be calibrated before certify; "
                f"available: {sorted(map(repr, self._calibrators))}."
            )
        return calibrator

    def certify(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        b0: Any = None,
    ) -> CertificateResult:
        """Certify a test unit: physics-residual coverage + hallucination map.

        Args:
            prediction: Estimated ``(rho, T1, T2)`` parameter map ``[B, 3, ...]``.
            target: Observed image ``[B, 1, ...]``.
            b0: Field strength selecting the Mondrian group (when stratified).

        Returns:
            A :class:`CertificateResult`.

        Raises:
            RuntimeError: certify before calibrate.
            KeyError: unseen field group under Mondrian stratification.
        """
        calibrator = self._calibrator_for(b0)
        q = calibrator.quantile
        score = self._score(prediction, target)
        coverage = float((score <= q).float().mean().item())
        flagged = score > q
        return CertificateResult(coverage=coverage, quantile=q, flagged=flagged, score=score)


__all__ = ["CertificateResult", "PhysResidualConformalCertificate"]
