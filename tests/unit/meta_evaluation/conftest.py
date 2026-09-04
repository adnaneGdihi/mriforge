"""Shared fixtures for meta-evaluation tests.

Each ranker is exercised on the same synthetic dataset: 4 phantoms x
8 degradation families x 4 severities. The metric set contains:

* ``mse``: known good distortion metric (positive on every axis).
* ``noise``: returns Gaussian noise — should rank dead last.
* ``constant``: returns 0.0 for every input — orthogonal to everything.

Tests check that ``mse`` consistently outranks the bad metrics. We do not
assert exact numeric scores because each ranker uses different
estimators with finite-sample variance.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.core.metrics.meta_evaluation import (
    MetricSet,
    SimulatorConfig,
    precompute_metric_values,
    run_simulator,
)
from spectramr.core.metrics.meta_evaluation.types import MetricEvaluationDataset


def _phantom(seed: int, size: int = 32) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, size),
        torch.linspace(-1, 1, size),
        indexing="ij",
    )
    r = torch.sqrt(xx**2 + yy**2)
    base = torch.exp(-r**2 / 0.5)
    speckle = 0.05 * torch.randn((size, size), generator=gen)
    return (base + speckle).unsqueeze(0).float()


def _mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(((pred.float() - target.float()) ** 2).mean().item())


def _mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred.float() - target.float()).abs().mean().item())


def _constant(pred: torch.Tensor, target: torch.Tensor) -> float:
    return 0.0


def _noise_metric(pred: torch.Tensor, target: torch.Tensor) -> float:
    # Pure noise — completely uncorrelated with reality.
    return float(torch.randn(1).item())


@pytest.fixture
def metric_set() -> MetricSet:
    return MetricSet(
        metrics={
            "mse": _mse,
            "mae": _mae,
            "constant": _constant,
            "noise": _noise_metric,
        },
        higher_is_better={
            "mse": False,
            "mae": False,
            "constant": False,
            "noise": False,
        },
        differentiable={
            "mse": True,
            "mae": True,
            "constant": False,
            "noise": False,
        },
    )


@pytest.fixture
def clean_volumes() -> list[tuple[str, torch.Tensor]]:
    return [(f"subj_{i}", _phantom(seed=10 + i)) for i in range(4)]


@pytest.fixture
def small_dataset(clean_volumes, metric_set) -> MetricEvaluationDataset:
    cfg = SimulatorConfig(
        families=[
            "gaussian_noise",
            "gaussian_blur",
            "gamma",
            "kspace_undersample",
        ],
        n_severities_per_family=4,
        seed=7,
    )
    samples = run_simulator(clean_volumes, cfg)
    values = precompute_metric_values(samples, metric_set.metrics)
    return MetricEvaluationDataset(samples=samples, metric_values=values)
