"""Experiment Campaign Orchestration.

Automates the lifecycle of multi-experiment campaigns:
define → submit → track → evaluate → compare.
"""

from spectramr.infrastructure.orchestration.campaign_orchestrator import (
    CampaignOrchestrator,
)
from spectramr.infrastructure.orchestration.campaign_state import (
    CampaignState,
    ExperimentStatus,
)

__all__ = [
    "CampaignOrchestrator",
    "CampaignState",
    "ExperimentStatus",
]
