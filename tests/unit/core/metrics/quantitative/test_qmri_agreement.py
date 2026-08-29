"""Tests for the qMRI agreement battery (Bland-Altman / ICC(3,1) / CoV).

Resolves the recurring ``deferred_source_backlog.json`` item: "run the qMRI
metrics battery on (pred_field, field)" for the VF / multi-acquisition arms.
"""

from __future__ import annotations

import torch

from mriforge.core.metrics.quantitative.qmri_agreement import qmri_agreement_metrics


def test_perfect_agreement_icc_one_bias_zero():
    field = torch.randn(4, 1, 16, 16)
    m = qmri_agreement_metrics(field.clone(), field.clone())
    assert abs(m["bland_altman_bias"]) < 1e-6
    assert m["icc_3_1"] > 0.999
    assert m["mae"] < 1e-6
    assert m["coefficient_of_variation"] < 1e-6


def test_constant_offset_shows_in_bias_not_spread():
    field = torch.randn(4, 1, 16, 16)
    pred = field + 0.5  # systematic +0.5 bias, no random scatter
    m = qmri_agreement_metrics(pred, field)
    assert abs(m["bland_altman_bias"] - 0.5) < 1e-4
    # zero scatter ⇒ LoA collapse onto the bias, ICG perfect consistency.
    assert abs(m["limits_of_agreement_upper"] - 0.5) < 1e-3
    assert abs(m["limits_of_agreement_lower"] - 0.5) < 1e-3
    assert m["icc_3_1"] > 0.999  # consistency ICC is invariant to a constant offset


def test_uncorrelated_estimate_has_low_icc():
    torch.manual_seed(0)
    ref = torch.randn(2048)
    pred = torch.randn(2048)  # independent ⇒ no agreement
    m = qmri_agreement_metrics(pred, ref)
    assert m["icc_3_1"] < 0.2


def test_limits_of_agreement_bracket_the_bias():
    torch.manual_seed(1)
    ref = torch.randn(1024)
    pred = ref + 0.1 + 0.3 * torch.randn(1024)
    m = qmri_agreement_metrics(pred, ref)
    assert m["limits_of_agreement_lower"] < m["bland_altman_bias"] < m["limits_of_agreement_upper"]
    assert m["coefficient_of_variation"] > 0


def test_empty_input_is_safe():
    m = qmri_agreement_metrics(torch.empty(0), torch.empty(0))
    assert m["icc_3_1"] == 0.0 and m["bland_altman_bias"] == 0.0


def test_reference_is_broadcast_to_pred():
    # a per-sample scalar ref broadcasts against a field pred (no shape crash).
    pred = torch.randn(3, 1, 8, 8)
    ref = torch.zeros(3, 1, 8, 8)
    m = qmri_agreement_metrics(pred, ref)
    assert "icc_3_1" in m and abs(m["bland_altman_bias"] - float(pred.mean())) < 1e-4
