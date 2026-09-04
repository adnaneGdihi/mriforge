"""Regression: campaign multiple-comparison correction collapsed to Bonferroni.

2026-06-11 infrastructure audit. ``_apply_correction`` used a binary if/else that
mapped every method except the literal ``"fdr"`` to Bonferroni — so
``holm_bonferroni`` / ``fdr_bh`` / ``benjamini_hochberg`` silently got the most
conservative correction. It now routes through ``correct_p_values`` (the SSOT
dispatch that raises on unknown methods).
"""

from __future__ import annotations

import inspect

from spectramr.core.metrics.statistical_tests import StatisticalTests
from spectramr.infrastructure.orchestration import campaign_evaluator


def test_apply_correction_uses_the_ssot_dispatch():
    src = inspect.getsource(campaign_evaluator.CampaignEvaluator._apply_correction)
    assert "correct_p_values" in src
    # The broken binary fallback must be gone.
    assert "bonferroni_correction" not in src


def test_bh_and_holm_differ_from_bonferroni():
    pvals = [0.01, 0.02, 0.03, 0.04]
    bonf = StatisticalTests.correct_p_values(pvals, method="bonferroni")
    bh = StatisticalTests.correct_p_values(pvals, method="benjamini_hochberg")
    holm = StatisticalTests.correct_p_values(pvals, method="holm_bonferroni")
    # If BH/Holm had collapsed to Bonferroni these would be identical.
    assert bh["corrected_p_values"] != bonf["corrected_p_values"]
    assert holm["corrected_p_values"] != bonf["corrected_p_values"]
