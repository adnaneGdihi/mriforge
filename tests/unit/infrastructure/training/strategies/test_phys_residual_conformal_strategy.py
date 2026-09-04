r"""Unit tests for the PR-CC calibration strategy + its science helper.

Targets ``spectramr.infrastructure.training.strategies.phys_residual_conformal_strategy``.

The strategy is a no-gradient calibration pass (mirrors B-1 equivariance_conformal):
it freezes a tissue-parameter estimator, runs ``run_phys_residual_calibration``
over the val loader, and emits the split-conformal quantile + empirical coverage.
The science lives in the module-level helper so it is testable without a full
training environment.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit

_ACQ = {"TR": 3000.0, "TE": 90.0, "TI": 0.0}


def _params(h: int = 16, w: int = 16) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(0.4, 1.0, h), torch.linspace(0.5, 1.0, w), indexing="ij"
    )
    rho = (0.6 + 0.4 * xx).unsqueeze(0).unsqueeze(0)
    t1 = (600.0 + 800.0 * yy).unsqueeze(0).unsqueeze(0)
    t2 = (60.0 + 80.0 * xx).unsqueeze(0).unsqueeze(0)
    return torch.cat([rho, t1, t2], dim=1)


def _se_render(params: torch.Tensor) -> torch.Tensor:
    from spectramr.infrastructure.physics.multi_physics_bloch import (
        MultiPhysicsBlochLayer,
    )

    layer = MultiPhysicsBlochLayer(learnable_flip_angle=False)
    return layer(
        {"rho": params[:, 0:1], "t1": params[:, 1:2], "t2": params[:, 2:3]},
        {"TR": 3000.0, "TE": 90.0, "TI": 0.0, "contrast_type": 0},
    )


def test_run_phys_residual_calibration_emits_coverage() -> None:
    from spectramr.infrastructure.training.strategies.phys_residual_conformal_strategy import (
        run_phys_residual_calibration,
    )

    torch.manual_seed(0)
    params = _params()
    clean = _se_render(params)
    batches = [
        {"input": clean, "target": clean + 0.05 * torch.randn_like(clean), "field_strength": 0.3}
        for _ in range(40)
    ]

    def param_map_fn(x: torch.Tensor) -> torch.Tensor:
        return params  # a frozen "estimator" that returns the parameter map

    report = run_phys_residual_calibration(
        param_map_fn,
        batches,
        alpha=0.1,
        acquisition=_ACQ,
        contrast_type="spin_echo",
        stratify_by="none",
        calibration_fraction=0.5,
        device="cpu",
    )
    assert report["coverage_at_alpha"] >= 0.85
    assert report["empirical_coverage"] == report["coverage_at_alpha"]
    assert report["quantile"] > 0.0
    assert report["n_calibration"] == 20
    assert report["score_fn"] == "physics_residual"
    assert report["stratify_by"] == "none"
    assert 0.0 <= report["flagged_fraction"] <= 1.0


def test_mondrian_stratifies_by_field() -> None:
    from spectramr.infrastructure.training.strategies.phys_residual_conformal_strategy import (
        run_phys_residual_calibration,
    )

    torch.manual_seed(1)
    params = _params()
    clean = _se_render(params)

    def make(b0: float, sigma: float, n: int) -> list:
        return [
            {"input": clean, "target": clean + sigma * torch.randn_like(clean), "field_strength": b0}
            for _ in range(n)
        ]

    batches = make(0.064, 0.16, 40) + make(3.0, 0.02, 40)

    def param_map_fn(x: torch.Tensor) -> torch.Tensor:
        return params

    report = run_phys_residual_calibration(
        param_map_fn,
        batches,
        alpha=0.1,
        acquisition=_ACQ,
        contrast_type="spin_echo",
        stratify_by="b0",
        calibration_fraction=0.5,
        device="cpu",
    )
    # Mondrian emits a per-field quantile map and a (pooled) coverage that meets target.
    assert report["stratify_by"] == "b0"
    assert isinstance(report["per_field_quantile"], dict)
    assert len(report["per_field_quantile"]) == 2
    assert report["coverage_at_alpha"] >= 0.85


def test_batch_extraction_raises_on_unknown_shape() -> None:
    from spectramr.infrastructure.training.strategies.phys_residual_conformal_strategy import (
        _extract_input_target_b0,
    )

    with pytest.raises(ValueError):
        _extract_input_target_b0("not a batch", default_b0=0.3)


def test_strategy_resolves_via_factory() -> None:
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    path = TrainingStrategyFactory.STRATEGY_CLASS_PATHS["phys_residual_conformal"]
    assert path.endswith("PhysResidualConformalStrategy")
