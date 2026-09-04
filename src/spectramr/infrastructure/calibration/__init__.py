"""Distribution-free conformal calibration utilities (PR-3 / H1).

Implements split-conformal prediction for image-to-image regression
following Vovk-Gammerman-Shafer 2005 and Angelopoulos et al. 2022.

The package exposes:

- :class:`ConformalCalibrator` — fit on a held-out calibration set,
  produce ``(y_hat, lower, upper)`` per pixel with the documented
  marginal coverage guarantee.
- :mod:`scores` — pluggable non-conformity score functions
  (absolute residual, quantile regression, conformalised quantile
  regression / CQR).
- :class:`IConformalCalibrator` — duck-typed protocol used by the
  reporting / strategy layer (re-exported here for convenience).

See ``TODO/backlog_paradigm_expansion_roadmap.md`` PR-3 for the
broader roadmap context.
"""

from .chd import CalibratedHallucinationDetector, dkw_slack
from .conformal import ConformalCalibrator, IConformalCalibrator
from .dice_risk_certificate import (
    DiceRiskCertificate,
    DiceRiskReport,
    hoeffding_risk_upper_bound,
    rcps_threshold,
)
from .pac_bayes_certificate import (
    PACBayesCertificate,
    PACBayesReport,
    mcallester_complexity_term,
    pac_bayes_bound,
)
from .pathology_recall_certificate import (
    PathologyRecallCertificate,
    PathologyRecallReport,
    hoeffding_recall_lower_bound,
    required_empirical_recall,
)
from .runner import CalibrationReport, ConformalCalibrationRunner
from .score_registry import (
    CALIBRATION_SCORE_REGISTRY,
    CalibrationScoreFn,
    list_calibration_scores,
    register_calibration_score,
    resolve_calibration_score,
)
from .scores import absolute_residual, conformalised_quantile_regression, quantile_regression
from .vf_residual_score import (
    marker_basis_condition_number,
    vf_residual_score,
)

__all__ = [
    "CALIBRATION_SCORE_REGISTRY",
    "CalibratedHallucinationDetector",
    "CalibrationReport",
    "CalibrationScoreFn",
    "ConformalCalibrationRunner",
    "ConformalCalibrator",
    "DiceRiskCertificate",
    "DiceRiskReport",
    "IConformalCalibrator",
    "PACBayesCertificate",
    "PACBayesReport",
    "PathologyRecallCertificate",
    "PathologyRecallReport",
    "absolute_residual",
    "conformalised_quantile_regression",
    "dkw_slack",
    "hoeffding_recall_lower_bound",
    "hoeffding_risk_upper_bound",
    "list_calibration_scores",
    "marker_basis_condition_number",
    "mcallester_complexity_term",
    "pac_bayes_bound",
    "quantile_regression",
    "rcps_threshold",
    "register_calibration_score",
    "required_empirical_recall",
    "resolve_calibration_score",
    "vf_residual_score",
]
