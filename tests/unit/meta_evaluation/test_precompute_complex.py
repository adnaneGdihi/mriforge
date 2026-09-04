"""``precompute_metric_values`` honours the ``requires_complex`` flag (pitfall #15).

M4Raw is complex k-space. A metric declaring ``requires_complex`` must receive the
complex sample when one is available, and on a magnitude-only cohort (the simulated
clean references are phaseless pseudo-GT) it must record NaN with a *logged* reason
rather than crash-to-NaN silently.
"""

from __future__ import annotations

import logging
import math

import torch

from spectramr.core.metrics.meta_evaluation.simulator import precompute_metric_values
from spectramr.core.metrics.meta_evaluation.types import DegradationSample


def _sample(clean: torch.Tensor, degraded: torch.Tensor) -> DegradationSample:
    return DegradationSample(
        clean=clean,
        degraded=degraded,
        degradation_family="noise",
        severity={"theta": 0.5},
        content_id="c0",
        seed=0,
    )


def test_requires_complex_skipped_on_magnitude_cohort(caplog) -> None:
    mag = _sample(torch.ones(1, 8, 8), torch.ones(1, 8, 8) * 0.5)
    seen: dict[str, bool] = {}

    def cplx(pred, target):  # would crash on real input in reality
        seen["ran"] = True
        return 1.0

    def l1(pred, target):
        return float((pred - target).abs().mean())

    with caplog.at_level(logging.WARNING):
        out = precompute_metric_values(
            [mag], {"complex_hfen": cplx, "l1": l1},
            complex_metric_keys=frozenset({"complex_hfen"}),
        )
    # The complex metric is skipped (never invoked) and recorded as NaN...
    assert "ran" not in seen
    assert len(out["complex_hfen"]) == 1 and math.isnan(out["complex_hfen"][0])
    # ...with a logged, diagnosable reason (not a silent swallow).
    assert any("requires complex" in r.message.lower() for r in caplog.records)
    # The ordinary metric still computes normally.
    assert out["l1"] == [0.5]


def test_requires_complex_runs_when_sample_is_complex() -> None:
    # Forward-compatible: if the cohort carries complex tensors, the metric runs.
    cplx_clean = torch.complex(torch.ones(1, 8, 8), torch.ones(1, 8, 8) * 0.3)
    cplx_deg = torch.complex(torch.ones(1, 8, 8) * 0.5, torch.zeros(1, 8, 8))
    seen: dict[str, bool] = {}

    def cplx(pred, target):
        seen["pred_complex"] = pred.is_complex()
        return float(pred.abs().mean())

    out = precompute_metric_values(
        [_sample(cplx_clean, cplx_deg)], {"complex_hfen": cplx},
        complex_metric_keys=frozenset({"complex_hfen"}),
    )
    assert seen["pred_complex"] is True
    assert math.isfinite(out["complex_hfen"][0])


def test_no_complex_keys_is_unchanged_behavior() -> None:
    # Default (no complex_metric_keys) preserves the original contract exactly.
    mag = _sample(torch.zeros(1, 4, 4), torch.zeros(1, 4, 4))
    out = precompute_metric_values([mag], {"l1": lambda p, t: float((p - t).abs().mean())})
    assert out["l1"] == [0.0]
