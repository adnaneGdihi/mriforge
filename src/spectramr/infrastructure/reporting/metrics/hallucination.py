r"""Radiomics-based feature-preservation profile + hallucination index.

Per `TODO/report_step` Part B (radiomics). Generalises the prostate-DLR
analysis: for each radiomic feature ``f``, compute the Concordance
Correlation Coefficient (CCC) between ``f(real)`` and ``f(synth)``
across the test cohort. Report:

* fraction of features with ``CCC >= 0.85`` (paper-quality threshold),
* median CCC by IBSI family,
* per-feature Bland–Altman bias.

The hallucination index is the fraction of *texture* features whose
deviation exceeds K test–retest standard deviations (uses a hold-out
dual-acquisition subset when supplied).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeaturePreservationReport:
    n_features: int
    median_ccc: float
    fraction_ccc_above_085: float
    family_median_ccc: dict[str, float]
    per_feature_ccc: pd.Series  # name -> ccc
    per_feature_bias: pd.Series  # name -> mean(synth - real)


def concordance_correlation_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    """Lin's CCC."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return float("nan")
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(ddof=0), b.var(ddof=0)
    cov = ((a - mu_a) * (b - mu_b)).mean()
    denom = var_a + var_b + (mu_a - mu_b) ** 2
    if denom < 1e-12:
        return float("nan")
    return float(2 * cov / denom)


def _family_of(feature_name: str) -> str:
    """Heuristic IBSI family classifier from PyRadiomics naming."""
    name = feature_name.lower()
    for fam in ("shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"):
        if fam in name:
            return fam
    if any(w in name for w in ("wavelet", "log-sigma")):
        return "filtered"
    return "other"


def feature_preservation_profile(
    real_features: pd.DataFrame,
    synth_features: pd.DataFrame,
    *,
    ccc_threshold: float = 0.85,
) -> FeaturePreservationReport:
    """Compute the radiomics CCC profile.

    Args:
        real_features: rows = subjects, columns = feature names.
        synth_features: same shape and column order as ``real_features``.
        ccc_threshold: Defensible threshold from the radiomics literature.
    """
    if real_features.shape != synth_features.shape:
        raise ValueError(f"shape mismatch {real_features.shape} vs {synth_features.shape}")
    cccs: dict[str, float] = {}
    biases: dict[str, float] = {}
    for col in real_features.columns:
        a = real_features[col].to_numpy()
        b = synth_features[col].to_numpy()
        cccs[col] = concordance_correlation_coefficient(a, b)
        biases[col] = float(np.nanmean(b - a))
    ser_ccc = pd.Series(cccs)
    ser_bias = pd.Series(biases)
    family_med: dict[str, float] = {}
    for fam, group in ser_ccc.groupby(_family_of):
        family_med[fam] = float(group.median())
    return FeaturePreservationReport(
        n_features=len(ser_ccc),
        median_ccc=float(ser_ccc.median()),
        fraction_ccc_above_085=float((ser_ccc >= ccc_threshold).mean()),
        family_median_ccc=family_med,
        per_feature_ccc=ser_ccc,
        per_feature_bias=ser_bias,
    )


def hallucination_index(
    real_features: pd.DataFrame,
    synth_features: pd.DataFrame,
    *,
    test_retest_std: pd.Series | None = None,
    texture_families: Iterable[str] = ("glcm", "glrlm", "glszm", "gldm", "ngtdm"),
    k: float = 2.0,
) -> float:
    r"""Fraction of texture features whose deviation exceeds K test–retest σ.

    Args:
        real_features, synth_features: per-subject radiomic feature tables.
        test_retest_std: Series mapping feature name → σ_test–retest.
            If None, falls back to the *between-subject* σ on the real
            cohort (more lenient — emit a warning).
        texture_families: which IBSI families count as texture.
        k: deviation multiplier for the threshold.
    """
    if real_features.shape != synth_features.shape:
        raise ValueError(f"shape mismatch {real_features.shape} vs {synth_features.shape}")
    if test_retest_std is None:
        import warnings

        warnings.warn(
            "hallucination_index: no test_retest_std supplied; "
            "falling back to between-subject σ — interpret as upper bound.",
            UserWarning,
            stacklevel=2,
        )
        test_retest_std = real_features.std(ddof=0)

    texture_cols = [c for c in real_features.columns if _family_of(c) in tuple(texture_families)]
    if not texture_cols:
        return float("nan")
    excess = 0
    total = 0
    for col in texture_cols:
        sigma = float(test_retest_std.get(col, 0.0))
        if sigma <= 0:
            continue
        diff = (synth_features[col] - real_features[col]).abs()
        excess += int((diff > k * sigma).sum())
        total += len(diff)
    return float(excess / max(total, 1))
