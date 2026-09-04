"""FSPD content residualization (critique §4 repair)."""

from __future__ import annotations

import torch

from spectramr.core.metrics.meta_evaluation.rankers.fspd import (
    FSPDConfig,
    FSPDRanker,
    _build_response_matrix,
)
from spectramr.core.metrics.meta_evaluation.types import (
    DegradationSample,
    MetricEvaluationDataset,
    MetricSet,
)

# Unbalanced content composition across severity: low theta is all "hi"-baseline
# content, high theta all "lo"-baseline content. A pure content-offset metric then
# *looks* severity-aligned unless content is residualised first.
_PLAN = [
    (0.0, "hi"),
    (0.5, "hi"),
    (0.5, "lo"),
    (1.0, "lo"),
]
_BASELINE = {"hi": 100.0, "lo": 0.0}


def _mse(pred, target):  # placeholder; FSPD reads precomputed values, never calls this
    return 0.0


def _dataset() -> MetricEvaluationDataset:
    samples = []
    content_track: list[float] = []
    sev_track: list[float] = []
    neutral: list[float] = []
    for k, (theta, content) in enumerate(_PLAN):
        samples.append(
            DegradationSample(
                clean=torch.ones(4),
                degraded=torch.ones(4) * (1.0 - theta),
                degradation_family="f",
                severity={"theta": theta},
                content_id=content,
                seed=k,
            )
        )
        content_track.append(_BASELINE[content])  # pure content offset
        sev_track.append(10.0 * theta)  # genuine severity signal
        neutral.append(1.0 + 0.01 * k)
    return MetricEvaluationDataset(
        samples=samples,
        metric_values={
            "content_tracker": content_track,
            "sev_tracker": sev_track,
            "neutral": neutral,
        },
    )


def test_residualization_flattens_pure_content_metric() -> None:
    ds = _dataset()
    raw = _build_response_matrix(ds, "content_tracker", ["f"], [0.0, 0.5, 1.0])
    res = _build_response_matrix(
        ds, "content_tracker", ["f"], [0.0, 0.5, 1.0], residualize_content=True
    )
    # Raw surface varies strongly with severity (100 -> 50 -> 0): the confound.
    assert raw.max() - raw.min() > 50.0
    # Residualised surface is ~flat (each content's offset removed).
    assert torch.nan_to_num(res).abs().max().item() < 1e-9


def test_residualization_preserves_true_severity_metric() -> None:
    ds = _dataset()
    res = _build_response_matrix(
        ds, "sev_tracker", ["f"], [0.0, 0.5, 1.0], residualize_content=True
    )
    curve = res.flatten()
    # Still monotone increasing across severity after residualisation.
    assert curve[0] < curve[1] < curve[2]


def test_residualization_demotes_content_tracker_score() -> None:
    ds = _dataset()
    metric_set = MetricSet(
        metrics={"content_tracker": _mse, "sev_tracker": _mse, "neutral": _mse}
    )
    out_plain = FSPDRanker(FSPDConfig(residualize_content=False)).rank(metric_set, ds)
    out_res = FSPDRanker(FSPDConfig(residualize_content=True)).rank(metric_set, ds)
    # After residualisation the genuine severity tracker is not beaten by the
    # pure content metric, and the content metric's score drops.
    assert out_res.scores["sev_tracker"] >= out_res.scores["content_tracker"]
    assert abs(out_res.scores["content_tracker"]) <= abs(out_plain.scores["content_tracker"]) + 1e-9


def _symmetric_dataset() -> MetricEvaluationDataset:
    # All metrics have a SYMMETRIC (V-shaped) response in severity, so the leading
    # FPCA mode is orthogonal to the (anti-symmetric) severity ramp.
    sevs = [0.0, 0.25, 0.5, 0.75, 1.0]
    samples = []
    a, b, c = [], [], []
    for k, theta in enumerate(sevs):
        samples.append(
            DegradationSample(
                clean=torch.ones(4),
                degraded=torch.ones(4) * (1.0 - theta),
                degradation_family="f",
                severity={"theta": theta},
                content_id="c0",
                seed=k,
            )
        )
        v = abs(theta - 0.5)  # symmetric in severity
        a.append(1.0 * v)
        b.append(2.0 * v)
        c.append(0.5 * v)
    return MetricEvaluationDataset(
        samples=samples, metric_values={"a": a, "b": b, "c": c}
    )


def test_fspd_orthogonal_leading_mode_reports_no_signal() -> None:
    # When mode 1 does not track severity (here, a symmetric response → mode 1 is
    # orthogonal to the ramp), the FSPD score's sign is arbitrary; the ranker must
    # report no severity signal (scores 0) rather than an arbitrarily-signed score
    # (critique M5).
    ds = _symmetric_dataset()
    metric_set = MetricSet(metrics={"a": _mse, "b": _mse, "c": _mse})
    out = FSPDRanker(FSPDConfig(residualize_content=False)).rank(metric_set, ds)
    assert out.diagnostics["mode1_severity_aligned"] is False
    assert abs(out.diagnostics["mode1_severity_cosine"]) < 0.05
    assert all(v == 0.0 for v in out.scores.values())
