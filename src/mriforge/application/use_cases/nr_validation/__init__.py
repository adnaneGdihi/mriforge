"""Multi-tier validation harness for the no-reference (NR) metric battery.

Implements the spec §8 label-free validation ladder as composable scorers, each
returning a per-metric :class:`TierResult`. The orchestrator
(``NRMetricValidationUseCase``) runs the tiers and folds them into a
:class:`MetricCard` per metric (pass/fail across tiers).

Tiers
-----
- §8.1 Tier 1 — physics-anchored concordance        :func:`cards.score_concordance`
- §8.2 Tier 2 — full-reference proxy agreement       :func:`tier2_fr_proxy.score_fr_proxy`
- §8.3 Tier 3 — channelized-Hotelling-observer task  :func:`tier3_cho.score_cho_concordance`
- §8.4 Tier 4 — construct validity + redundancy gate :func:`tier4_construct.score_construct_validity`
- §8.5 Tier 5 — ranking/defensibility is the meta-evaluation pipeline itself.
- §8.7 Aggregator — ``nr_quality_index``             :func:`aggregator.fit_nr_quality_index`

All scorers are CPU-canonical and reproducible, return ``dict[metric -> TierResult]``
(folded into per-metric :class:`MetricCard` by :func:`build_cards`), never raise on a
bad metric (crash→NaN), and lazy-import + gate heavy/optional deps (scikit-learn).
"""

from __future__ import annotations

from .aggregator import (
    NRQualityIndexModel,
    fit_nr_quality_index,
    register_nr_quality_index,
)
from .cards import (
    MetricCard,
    TierResult,
    build_cards,
    eval_metric,
    kendall_tau,
    score_concordance,
    spearman,
)
from .report import (
    render_cards_markdown,
    render_nr_validation_figures,
    write_cards_table,
    write_nr_validation_report,
)
from .tier2_fr_proxy import (
    DEFAULT_FR_PROXIES,
    score_fr_proxy,
    variance_inflation_factors,
)
from .tier3_cho import score_cho_concordance
from .tier4_construct import (
    icc_2_1,
    redundancy_gate,
    score_construct_validity,
    score_invariance,
    score_test_retest,
)

__all__ = [
    "DEFAULT_FR_PROXIES",
    "MetricCard",
    "NRQualityIndexModel",
    "TierResult",
    "build_cards",
    "eval_metric",
    "fit_nr_quality_index",
    "icc_2_1",
    "kendall_tau",
    "redundancy_gate",
    "register_nr_quality_index",
    "render_cards_markdown",
    "render_nr_validation_figures",
    "score_cho_concordance",
    "score_concordance",
    "score_construct_validity",
    "score_fr_proxy",
    "score_invariance",
    "score_test_retest",
    "spearman",
    "variance_inflation_factors",
    "write_cards_table",
    "write_nr_validation_report",
]
