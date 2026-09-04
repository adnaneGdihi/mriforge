r"""Conformal trust calibration for cold-diffusion reconstructions (C7).

The papers' C7 corollary: the per-image trust scores (κ, η^null) only become
*decisions* through a calibrated threshold — an empirical
:math:`\lceil(n+1)(1-\alpha)\rceil`-th order statistic on clean validation
scores, valid under exchangeability. This module packages that threshold as a
durable artifact so the deployment-side flag is auditable:

- :class:`TrustCalibrationArtifact` — the frozen record (q̂_α, the clean
  η^null quantile law, n, α, the DKW band, and the σ/seed the sampler ran
  with — σ is part of the exchangeability contract, A9).
- :func:`fit_trust_calibration` — fits via the existing
  :class:`~spectramr.infrastructure.calibration.conformal.ConformalCalibrator`
  (no fourth conformal implementation) and REFUSES undersized calibration
  sets (n < ⌈1/α⌉ − 1 ⇒ the honest band is infinite; shipping a finite
  threshold there would advertise a guarantee that does not hold).
- :func:`confident_fabrication_flag` — the deployment signature: κ within
  the calibrated band (the reconstruction *explains the data*) while η^null
  sits above the clean-law tail (it *invented* unmeasured content anyway).
- :func:`write_trust_certificate_json` — serialisation next to the other
  certificate writers.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from spectramr.infrastructure.calibration.chd import dkw_slack
from spectramr.infrastructure.calibration.conformal import ConformalCalibrator

_ETA_QUANTILES = (0.50, 0.90, 0.95, 0.99)


@dataclass(frozen=True)
class TrustCalibrationArtifact:
    """Frozen record of one trust calibration run.

    Attributes:
        score_name: registry name of the calibrated score (e.g.
            ``"kappa_residual"``).
        alpha: mis-coverage rate the band was fitted at.
        n: number of calibration images.
        q_hat_alpha: the conformal band radius — scores ≤ q̂_α are "within
            the clean law" at level 1−α.
        eta_null_clean_quantiles: empirical quantile law of η^null on the
            SAME clean calibration images (keys ``"q50"``/``"q90"``/…).
        dkw_eps: DKW slack ``sqrt(ln(2/α)/(2n))`` — the finite-sample band
            around every empirical quantile above; report thresholds WITH it.
        sigma: the ``sampler_sigma`` the reconstructions were drawn at. A
            deployment σ different from this voids exchangeability (A9) —
            refit rather than reuse.
        seed: the ``sampler_seed`` used (``None`` = unseeded).
    """

    score_name: str
    alpha: float
    n: int
    q_hat_alpha: float
    eta_null_clean_quantiles: dict[str, float] = field(default_factory=dict)
    dkw_eps: float = 0.0
    sigma: float = 0.0
    seed: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, doc: dict) -> TrustCalibrationArtifact:
        return cls(**doc)


def _empirical_quantile(ordered: list[float], q: float) -> float:
    """Nearest-rank empirical quantile of an ascending-sorted list."""
    n = len(ordered)
    rank = min(max(math.ceil(q * n), 1), n)
    return ordered[rank - 1]


def fit_trust_calibration(
    scores: Sequence[float] | torch.Tensor,
    eta_null_clean: Sequence[float] | torch.Tensor,
    alpha: float = 0.1,
    *,
    score_name: str = "kappa_residual",
    sigma: float = 0.0,
    seed: int | None = None,
) -> TrustCalibrationArtifact:
    """Fit the C7 trust calibration on clean validation images.

    Args:
        scores: per-image trust scores (κ) on a clean calibration split.
        eta_null_clean: per-image η^null on the SAME split — its quantile
            law is what the fabrication flag compares against.
        alpha: mis-coverage rate.
        score_name / sigma / seed: provenance recorded in the artifact.

    Raises:
        ValueError: empty inputs, or ``n < ⌈1/α⌉ − 1`` (the finite-sample
            band is infinite — a finite threshold would be anti-conservative).
    """
    score_tensor = torch.as_tensor(list(scores), dtype=torch.float64).flatten()
    eta_tensor = torch.as_tensor(list(eta_null_clean), dtype=torch.float64).flatten()
    n = int(score_tensor.numel())
    if n == 0 or eta_tensor.numel() == 0:
        raise ValueError("fit_trust_calibration needs non-empty scores and eta_null values.")

    # Reuse the SSOT split-conformal machinery: identity score over the
    # already-computed per-image trust scores. Its +∞ branch is exactly the
    # sizing guard we must surface as a hard error here.
    calibrator = ConformalCalibrator(score_fn=lambda pred, _target: pred, alpha=alpha)
    calibrator.fit([(score_tensor, score_tensor)])
    if math.isinf(calibrator.quantile):
        needed = math.ceil(1.0 / alpha) - 1
        raise ValueError(
            f"Calibration set too small: n={n} < {needed} required for "
            f"alpha={alpha}. The honest conformal band is infinite — collect "
            "more clean calibration images instead of shipping a threshold "
            "with no guarantee."
        )

    eta_sorted = sorted(float(v) for v in eta_tensor)
    quantiles = {f"q{int(q * 100)}": _empirical_quantile(eta_sorted, q) for q in _ETA_QUANTILES}
    return TrustCalibrationArtifact(
        score_name=score_name,
        alpha=float(alpha),
        n=n,
        q_hat_alpha=float(calibrator.quantile),
        eta_null_clean_quantiles=quantiles,
        dkw_eps=dkw_slack(n, alpha),
        sigma=float(sigma),
        seed=None if seed is None else int(seed),
    )


def confident_fabrication_flag(
    kappa: float,
    eta_null: float,
    artifact: TrustCalibrationArtifact,
    eta_quantile: str = "q99",
) -> bool:
    """The papers' confident-fabrication signature.

    Fires when κ is WITHIN the calibrated band (the reconstruction explains
    the acquired data as well as clean ones do) while η^null exceeds the
    clean-law tail quantile (it put anomalously much energy where nothing
    was measured). High κ alone is ordinary inconsistency — visible in the
    residual; this pairing is the failure that residuals cannot see.
    """
    if eta_quantile not in artifact.eta_null_clean_quantiles:
        raise KeyError(
            f"Unknown eta_quantile {eta_quantile!r}; artifact carries "
            f"{sorted(artifact.eta_null_clean_quantiles)}."
        )
    return (
        kappa <= artifact.q_hat_alpha and eta_null > artifact.eta_null_clean_quantiles[eta_quantile]
    )


def write_trust_certificate_json(artifact: TrustCalibrationArtifact, out_path: str | Path) -> Path:
    """Serialise the artifact (creating parent dirs); returns the path."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact.to_dict(), indent=2, sort_keys=True))
    return path


__all__ = [
    "TrustCalibrationArtifact",
    "confident_fabrication_flag",
    "fit_trust_calibration",
    "write_trust_certificate_json",
]
