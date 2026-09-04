"""ESD Monte-Carlo seeding tied to sample identity (critique §7 / L5 repair)."""

from __future__ import annotations

import pytest
import torch

from spectramr.core.metrics.meta_evaluation.rankers.esd import ESDConfig, ESDRanker
from spectramr.core.metrics.meta_evaluation.types import (
    DegradationSample,
    MetricEvaluationDataset,
    MetricSet,
)

_SEEDS = [11, 22, 33, 44]


def _phantom(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (0.5 + 0.3 * torch.randn((1, 16, 16), generator=g)).float()


def _mse(pred, target):
    return float(((pred.float() - target.float()) ** 2).mean().item())


def _l1(pred, target):
    return float((pred.float() - target.float()).abs().mean().item())


def _dataset(order: list[int]) -> MetricEvaluationDataset:
    samples = []
    for k in order:
        clean = _phantom(k)
        samples.append(
            DegradationSample(
                clean=clean,
                degraded=clean + 0.1,
                degradation_family="gaussian_noise",
                severity={"theta": 0.5},
                content_id=f"c{k}",
                seed=k,
            )
        )
    return MetricEvaluationDataset(
        samples=samples, metric_values={"mse": [0.0] * len(order), "l1": [0.0] * len(order)}
    )


def _metric_set() -> MetricSet:
    return MetricSet(
        metrics={"mse": _mse, "l1": _l1}, higher_is_better={"mse": False, "l1": False}
    )


def test_seed_by_sample_is_order_invariant() -> None:
    cfg = ESDConfig(seed_by_sample=True, max_pairs=10, n_haar_samples=2)
    forward = ESDRanker(cfg).rank(_metric_set(), _dataset(_SEEDS))
    shuffled = ESDRanker(cfg).rank(_metric_set(), _dataset(list(reversed(_SEEDS))))
    # Each cell seeds from its own sample.seed, so re-ordering the pool cannot
    # change the per-metric scores (the §7 reproducibility guarantee).
    for name in forward.scores:
        assert forward.scores[name] == pytest.approx(shuffled.scores[name], rel=1e-9, abs=1e-9)


def test_seed_by_sample_repeatable() -> None:
    cfg = ESDConfig(seed_by_sample=True, max_pairs=10, n_haar_samples=2)
    a = ESDRanker(cfg).rank(_metric_set(), _dataset(_SEEDS))
    b = ESDRanker(cfg).rank(_metric_set(), _dataset(_SEEDS))
    assert a.scores == b.scores


def test_global_seed_path_still_runs() -> None:
    # Default (global-seed) path remains functional and deterministic per-run.
    cfg = ESDConfig(seed_by_sample=False, max_pairs=10, n_haar_samples=2)
    out = ESDRanker(cfg).rank(_metric_set(), _dataset(_SEEDS))
    assert set(out.scores) == {"mse", "l1"}
