"""Downstream-task evaluation pipeline (PR-5 / H3).

Plan: TODO/backlog_paradigm_expansion_roadmap.md §PR-5.

Walks every reconstruction in a dataloader through one or more
downstream-task adapters, aggregates per-sample metrics, and returns a
single :class:`DownstreamMetricResult` per adapter.

Wires into the campaign-evaluation step via
``DownstreamEvaluationPipeline.run_for_campaign``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch

from mriforge.domain.interfaces.i_downstream_model_service import IDownstreamModelService


@dataclass
class DownstreamMetricResult:
    """Aggregated downstream-task metric across a dataloader."""

    task_name: str
    n_samples: int
    metrics: dict[str, float]
    per_sample: list[dict[str, float]] = field(default_factory=list)

    def summarise(self) -> dict[str, Any]:
        return {
            "task": self.task_name,
            "n_samples": self.n_samples,
            "metrics": self.metrics,
        }


class DownstreamEvaluationPipeline:
    """Run a battery of downstream tasks on a reconstruction stream."""

    def __init__(self, adapters: Iterable[IDownstreamModelService]) -> None:
        self._adapters = list(adapters)
        if not self._adapters:
            raise ValueError(
                "DownstreamEvaluationPipeline received zero adapters — "
                "configure at least one IDownstreamModelService."
            )

    @torch.no_grad()
    def evaluate(
        self,
        reconstructions: Iterable[torch.Tensor],
        ground_truths: Iterable[torch.Tensor],
        *,
        keep_per_sample: bool = False,
    ) -> dict[str, DownstreamMetricResult]:
        """Aggregate metrics across all reconstruction / ground-truth pairs."""
        results: dict[str, DownstreamMetricResult] = {
            a.task_name: DownstreamMetricResult(task_name=a.task_name, n_samples=0, metrics={})
            for a in self._adapters
        }
        agg: dict[str, dict[str, float]] = {a.task_name: {} for a in self._adapters}
        n: dict[str, int] = {a.task_name: 0 for a in self._adapters}

        for recon, gt in zip(reconstructions, ground_truths, strict=False):
            for adapter in self._adapters:
                m = adapter.compute_metric(recon, gt)
                if keep_per_sample:
                    results[adapter.task_name].per_sample.append(dict(m))
                for k, v in m.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        agg[adapter.task_name][k] = agg[adapter.task_name].get(k, 0.0) + float(v)
                n[adapter.task_name] += 1

        for tn, summed in agg.items():
            count = max(n[tn], 1)
            results[tn].metrics = {k: v / count for k, v in summed.items()}
            results[tn].n_samples = n[tn]
        return results

    def run_for_campaign(
        self,
        reconstructions: Iterable[torch.Tensor],
        ground_truths: Iterable[torch.Tensor],
    ) -> dict[str, dict[str, Any]]:
        """Run the suite and return a JSON-serializable summary."""
        per_task = self.evaluate(reconstructions, ground_truths, keep_per_sample=False)
        return {tn: r.summarise() for tn, r in per_task.items()}
