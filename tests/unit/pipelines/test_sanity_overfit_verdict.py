"""Unit tests for the overfit-single-batch sanity verdict.

The ``sanity_check`` command repeats one batch forever; a correctly-wired
model must memorise it (loss collapses). These tests pin the pass/fail
contract of :func:`_evaluate_sanity_overfit`, including the k-space-phase
gate that catches the experiment_11 "DC blob" failure mode (magnitude
learned, phase stuck).
"""

from __future__ import annotations

import pytest

from spectramr.pipelines.train import (
    SANITY_MIN_ITERS,
    _evaluate_sanity_overfit,
)


def _decreasing(n: int, start: float = 1.0, end: float = 0.01) -> list[float]:
    """Linearly-interpolated decreasing trace of length ``n``."""
    if n == 1:
        return [start]
    step = (start - end) / (n - 1)
    return [start - step * i for i in range(n)]


def _flat(n: int, value: float = 1.0) -> list[float]:
    return [value] * n


class TestLossCollapse:
    def test_collapsing_loss_passes(self):
        passed, report = _evaluate_sanity_overfit(_decreasing(100), [])
        assert passed is True
        assert report["final_loss"] < report["initial_loss"]
        assert report["loss_ratio"] < 0.5

    def test_flat_loss_fails(self):
        passed, report = _evaluate_sanity_overfit(_flat(100, 0.8), [])
        assert passed is False
        assert "collapse" in report["reason"]

    def test_slightly_decreasing_but_above_ratio_fails(self):
        # 1.0 -> 0.6: ratio 0.6 > 0.5 threshold -> cannot overfit one batch
        passed, _ = _evaluate_sanity_overfit(_decreasing(100, 1.0, 0.6), [])
        assert passed is False

    def test_already_tiny_loss_passes_via_floor(self):
        # Flat but essentially zero: the absolute floor saves it.
        passed, _ = _evaluate_sanity_overfit(_flat(100, 1e-6), [])
        assert passed is True


class TestDivergenceAndShortRuns:
    def test_nonfinite_loss_fails(self):
        trace = _decreasing(100)
        trace[50] = float("nan")
        passed, report = _evaluate_sanity_overfit(trace, [])
        assert passed is False
        assert "non-finite" in report["reason"]

    def test_inf_loss_fails(self):
        trace = _decreasing(100)
        trace[10] = float("inf")
        passed, _ = _evaluate_sanity_overfit(trace, [])
        assert passed is False

    def test_too_few_iters_is_inconclusive_pass(self):
        passed, report = _evaluate_sanity_overfit(_flat(SANITY_MIN_ITERS - 1), [])
        assert passed is True
        assert report.get("inconclusive") is True


class TestPhaseGate:
    def test_collapsing_loss_but_stuck_phase_fails(self):
        # The exact "DC blob" signature: total loss falls (magnitude learned)
        # but the phase metric plateaus near its random-init value.
        passed, report = _evaluate_sanity_overfit(
            _decreasing(100), _flat(100, 2.0)
        )
        assert passed is False
        assert "phase" in report["reason"]
        assert report["phase_ratio"] >= 0.7

    def test_collapsing_loss_and_improving_phase_passes(self):
        passed, report = _evaluate_sanity_overfit(
            _decreasing(100), _decreasing(100, 2.0, 0.1)
        )
        assert passed is True
        assert report["phase_final"] < report["phase_initial"]

    def test_phase_gate_skipped_when_no_phase_metric(self):
        passed, report = _evaluate_sanity_overfit(_decreasing(100), [])
        assert passed is True
        assert "phase_initial" not in report

    def test_short_phase_trace_does_not_block_pass(self):
        # Loss collapses; phase trace too short to judge -> phase gate skipped.
        passed, _ = _evaluate_sanity_overfit(
            _decreasing(100), _flat(SANITY_MIN_ITERS - 1, 2.0)
        )
        assert passed is True


def test_report_is_json_serialisable():
    import json

    _, report = _evaluate_sanity_overfit(_decreasing(100), _decreasing(100, 2.0, 0.1))
    # Must not raise — the report is stamped into the run result / provenance.
    json.dumps(report)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
