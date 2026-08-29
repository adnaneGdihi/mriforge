"""Regression: HPO best-trial selection must (a) key on a metric the evaluator
actually emits and (b) respect the optimization direction.

The pre-fix grid/random search did ``metrics.get("loss", -inf)`` and kept the
trial with the LARGEST value. Two bugs:

* the standard evaluator (``ablation.train_and_score``) returns
  ``final_loss`` / ``<metric>_best`` — never a bare ``loss`` key — so every
  trial resolved to ``-inf`` and ``best_trial`` stayed ``None``;
* ``> best_metric`` maximizes, so even with a real loss key it crowned the
  WORST (highest-loss) trial.

``_better_metric`` is the direction-aware selection seam both loops now use.
"""

from __future__ import annotations

import math

from mriforge.pipelines.hpo import _better_metric, _resolve_metric


def test_minimize_prefers_lower_loss():
    # Fresh incumbent (+inf for a minimize sweep): any finite loss is better.
    assert _better_metric(0.5, math.inf, minimize=True) is True
    # A higher loss than the incumbent is NOT better.
    assert _better_metric(0.6, 0.5, minimize=True) is False
    # A lower loss IS better.
    assert _better_metric(0.4, 0.5, minimize=True) is True


def test_maximize_prefers_higher_metric():
    assert _better_metric(30.0, -math.inf, minimize=False) is True
    assert _better_metric(29.0, 30.0, minimize=False) is False
    assert _better_metric(31.0, 30.0, minimize=False) is True


def test_missing_metric_is_never_better():
    # A trial that did not produce the metric (resolve → None) must never win.
    assert _better_metric(None, math.inf, minimize=True) is False
    assert _better_metric(None, -math.inf, minimize=False) is False


def test_resolve_metric_matches_evaluator_keys():
    # The real evaluator emits final_loss / <metric>_best. The default HPO
    # metric ("final_loss") now resolves exactly — the pre-fix code keyed on a
    # bare ``loss`` via ``dict.get`` (exact key), which never matched and left
    # every trial at the sentinel.
    evaluator_out = {"val_psnr_best": 31.2, "final_loss": 0.08, "success": True}
    assert _resolve_metric(evaluator_out, "final_loss") == 0.08
    assert _resolve_metric(evaluator_out, "val_psnr_best") == 31.2
    # A genuinely-absent metric resolves to None (so the trial cannot win).
    assert _resolve_metric(evaluator_out, "dice") is None
