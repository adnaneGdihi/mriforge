"""Distributed Training Infrastructure (multi-GPU + FSDP + PEFT).

Public surface — these names are part of the dynamic-dispatch contract
and stay reachable even when production paths under ``src/`` don't
currently import them. Distributed training is a *feature gate* (R3 in
``TODO/audit/CHANGES_RATIONALE.md`` §0): the absence of a current
caller does not justify trimming the surface.

- :mod:`distributed_training` — rank utilities, loss / metrics audit,
  multi-GPU wrapping and synchronization.
- :mod:`fsdp_wrap` — wrap an ``nn.Module`` with FSDP under the
  configured sharding + mixed-precision policy.
- :mod:`peft_inject` — inject LoRA / Adapter / Prefix modules.
- :mod:`launcher` — campaign-aware launcher
  (``dispatch_for_campaign``) + single-arm helper (``launch_distributed``).
"""

from .distributed_training import (
    DistributedAuditResult,
    DistributedLossAudit,
    DistributedMetricsCollector,
    DistributedMetricsSnapshot,
    DistributedTrainingContext,
    RankUtility,
    synchronize_across_ranks,
    wrap_model_distributed,
)
from .fsdp_wrap import maybe_wrap_with_fsdp
from .launcher import dispatch_for_campaign, launch_distributed
from .peft_inject import LoRAAdapter, inject_peft

__all__ = [
    "DistributedAuditResult",
    "DistributedLossAudit",
    "DistributedMetricsCollector",
    "DistributedMetricsSnapshot",
    "DistributedTrainingContext",
    "LoRAAdapter",
    "RankUtility",
    "dispatch_for_campaign",
    "inject_peft",
    "launch_distributed",
    "maybe_wrap_with_fsdp",
    "synchronize_across_ranks",
    "wrap_model_distributed",
]
