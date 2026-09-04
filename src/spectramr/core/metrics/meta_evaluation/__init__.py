"""Meta-evaluation framework for MRI image-quality metrics.

Five novel rankers — CDSCR, LGDR, SFA, ESD, FSPD — evaluate any callable
``IMetric`` along complementary structural-mathematical axes and combine
into an asymmetric defensibility flag. See module-level docstrings for the
mathematical justification of each ranker.

Public API
----------

* ``MetaEvaluationPipeline.from_defaults()`` builds a pipeline that runs
  the simulator, computes the per-metric values, runs every ranker, and
  aggregates.
* ``MetricSet`` bundles user-supplied callables plus their ``higher_is_better``
  / ``differentiable`` flags.
* Individual rankers (e.g. ``CDSCRRanker``) can be instantiated and called
  directly when only one ranking is needed.
"""

from .aggregation import (
    AggregatorConfig,
    KemenyConfig,
    aggregate,
    calibrate_defensibility,
    kemeny_consensus,
)
from .betting import (
    BettingPartialOrder,
    betting_confidence_sequence,
    betting_wealth,
    certify_betting_order,
    evalue_tail_mass,
    kendall_concordance_increments,
)
from .comparison import (
    compare_summaries,
    render_comparison_markdown,
    spearman_rank_correlation,
)
from .conformal import (
    ConformalRiskResult,
    conformal_metric_selection,
    crc_admissible,
    crc_risk_bound,
    select_conformal_threshold,
)
from .iqa_recalibration import (
    RecalibrationFit,
    RecalibrationResult,
    accept_recalibration,
    fit_recalibration,
    kendall_tau,
    recalibrate,
)
from .krcps import KRCPSResult, krcps_calibrate
from .leaderboard import (
    DEFAULT_MIN_SAMPLES,
    CellKey,
    ConsensusResult,
    LeaderboardCell,
    LeaderboardCube,
    LeaderboardEntry,
    build_cell,
)
from .metric_adapter import build_safe_metric_set, safe_metric_call, wrap_metric
from .paired_comparison import PairedDelta, lesion_vs_control, money_table, spearman_rho
from .pipeline import (
    EVAL_MODES,
    MetaEvaluationOutput,
    MetaEvaluationPipeline,
    resolve_eval_mode,
)
from .pool_hygiene import PoolScreening, screen_pool
from .powered import (
    PoweredPartialOrder,
    certify_partial_order,
    min_samples_for_gap,
    min_samples_for_pool,
    powered_gap_threshold,
)
from .regional import (
    RegionalEvaluationBundle,
    build_regional_bundles,
    region_ids_in,
)
from .transfer import (
    TransferCertificate,
    pmean_diff_bound,
    tv_distance,
    tv_distance_joint_samples,
    tv_distance_samples,
    tv_transfer_certificate,
)


def render_figures(output, out_dir, *, strict: bool = True):
    """Lazy import of figures.render_all (avoids matplotlib at import time).

    Strict by default: a renderer that raises fails the run rather than leaving a
    silently-missing figure behind.
    """
    from .figures import render_all

    return render_all(output, out_dir, strict=strict)


def write_tables(output, out_dir, *, strict: bool = True):
    """Lazy import of tables.write_all_tables (keeps csv import out of init).

    Strict by default: a table that fails to write fails the run rather than
    silently thinning the results directory.
    """
    from .tables import write_all_tables

    return write_all_tables(output, out_dir, strict=strict)


from .rankers import (
    BaseRanker,
    CDSCRConfig,
    CDSCRRanker,
    ESDConfig,
    ESDRanker,
    FSPDConfig,
    FSPDRanker,
    LGDRConfig,
    LGDRRanker,
    RankerSpec,
    SFAConfig,
    SFARanker,
    Sim2RankConfig,
    Sim2RankRanker,
    available_rankers,
    check_data_dependencies,
    get_ranker,
    has_ranker,
    list_rankers,
    register_ranker,
)
from .simulator import (
    DEGRADATION_LIBRARY,
    SimulatorConfig,
    precompute_metric_values,
    run_simulator,
)
from .types import (
    AggregateRankingResult,
    DefensibilityCalibration,
    DegradationSample,
    MetricEvaluationDataset,
    MetricSet,
    RankingResult,
)

__all__ = [
    # Region-conditional evaluation (the region axis)
    "RegionalEvaluationBundle",
    "build_regional_bundles",
    "region_ids_in",
    # Leaderboard cube
    "CellKey",
    "ConsensusResult",
    "LeaderboardCell",
    "LeaderboardCube",
    "LeaderboardEntry",
    "build_cell",
    "DEFAULT_MIN_SAMPLES",
    # The money table: lesion vs size-matched control
    "PairedDelta",
    "lesion_vs_control",
    "money_table",
    "spearman_rho",
    # Pipeline
    "MetaEvaluationPipeline",
    "MetaEvaluationOutput",
    "EVAL_MODES",
    "resolve_eval_mode",
    # Rankers
    "BaseRanker",
    "CDSCRRanker",
    "CDSCRConfig",
    "LGDRRanker",
    "LGDRConfig",
    "SFARanker",
    "SFAConfig",
    "ESDRanker",
    "ESDConfig",
    "FSPDRanker",
    "FSPDConfig",
    "Sim2RankRanker",
    "Sim2RankConfig",
    # Unified ranker registry (one registry, one BaseRanker contract, all gens)
    "RankerSpec",
    "register_ranker",
    "get_ranker",
    "list_rankers",
    "available_rankers",
    "has_ranker",
    "check_data_dependencies",
    # Aggregation
    "AggregatorConfig",
    "aggregate",
    "calibrate_defensibility",
    "DefensibilityCalibration",
    # L3⁺ Kemeny (Condorcet-consistent) aggregation
    "KemenyConfig",
    "kemeny_consensus",
    # Powered certification (finite-sample gate)
    "PoweredPartialOrder",
    "certify_partial_order",
    "min_samples_for_gap",
    "min_samples_for_pool",
    "powered_gap_threshold",
    # L2⁺ betting / anytime-valid confidence
    "BettingPartialOrder",
    "betting_wealth",
    "evalue_tail_mass",
    "betting_confidence_sequence",
    "certify_betting_order",
    "kendall_concordance_increments",
    # L5 conformal risk control
    "ConformalRiskResult",
    "crc_admissible",
    "crc_risk_bound",
    "select_conformal_threshold",
    "conformal_metric_selection",
    # K-RCPS entrywise conformal risk control
    "KRCPSResult",
    "krcps_calibrate",
    # T5 in-domain IQA recalibration
    "RecalibrationFit",
    "RecalibrationResult",
    "accept_recalibration",
    "fit_recalibration",
    "kendall_tau",
    "recalibrate",
    # Cross-arm comparison (real vs betting, by eval_mode)
    "compare_summaries",
    "render_comparison_markdown",
    "spearman_rank_correlation",
    # L4 sim-to-real transfer certificate
    "TransferCertificate",
    "tv_distance",
    "tv_distance_samples",
    "tv_distance_joint_samples",
    "pmean_diff_bound",
    "tv_transfer_certificate",
    # Pool hygiene (clone + broken-metric screening)
    "PoolScreening",
    "screen_pool",
    # Data structures
    "MetricSet",
    "MetricEvaluationDataset",
    "DegradationSample",
    "RankingResult",
    "AggregateRankingResult",
    # Simulator
    "SimulatorConfig",
    "DEGRADATION_LIBRARY",
    "run_simulator",
    "precompute_metric_values",
    # Figures (lazy)
    "render_figures",
    # Tables (lazy)
    "write_tables",
    # Metric adapter (handles registry quirks)
    "build_safe_metric_set",
    "safe_metric_call",
    "wrap_metric",
]
