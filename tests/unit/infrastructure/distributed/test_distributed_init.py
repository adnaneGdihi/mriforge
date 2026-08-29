from mriforge.infrastructure.distributed import (
    DistributedAuditResult,
    DistributedLossAudit,
    DistributedMetricsCollector,
    DistributedMetricsSnapshot,
    DistributedTrainingContext,
    RankUtility,
    synchronize_across_ranks,
    wrap_model_distributed,
)


def test_distributed_init_exports():
    """Verify that all required distributed training exports are available."""
    assert RankUtility is not None
    assert DistributedLossAudit is not None
    assert DistributedMetricsCollector is not None
    assert DistributedTrainingContext is not None
    assert DistributedAuditResult is not None
    assert DistributedMetricsSnapshot is not None
    assert wrap_model_distributed is not None
    assert synchronize_across_ranks is not None
