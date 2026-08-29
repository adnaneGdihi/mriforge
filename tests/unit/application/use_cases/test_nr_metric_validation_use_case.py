"""Tests for the NR-metric validation use case (spec §8)."""

from __future__ import annotations

import pytest
import torch

from mriforge.application.use_cases.nr_metric_validation_use_case import (
    DEFAULT_NR_BATTERY,
    HARNESS_TIERS,
    NRMetricValidationConfig,
    NRMetricValidationUseCase,
    build_physics_operator_library,
)


def _phantoms(n: int = 2) -> list[tuple[str, torch.Tensor]]:
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32), indexing="ij"
    )
    base = ((xx**2 + yy**2 < 0.5).float() * 0.8).unsqueeze(0)
    return [(f"s{i}", base + 0.01 * i) for i in range(n)]


def test_default_battery_excludes_gated_metrics() -> None:
    # Prior-geometry / cross-contrast-predictor metrics need trained assets and
    # must NOT be in the default label-free battery.
    gated = {"pmmd", "dpst", "fitc", "ksdp", "pfld", "hallucination_index", "ncpd", "ccpr"}
    assert not (set(DEFAULT_NR_BATTERY) & gated)


def test_build_physics_operator_library_signature() -> None:
    lib = build_physics_operator_library(["rician", "gibbs"])
    assert set(lib) == {"rician", "gibbs"}
    clean = torch.rand(1, 32, 32)
    gen = torch.Generator().manual_seed(3)
    out = lib["rician"](clean, 0.5, gen)
    assert out.shape == clean.shape
    # deterministic for a fixed (theta, generator-seed)
    gen2 = torch.Generator().manual_seed(3)
    out2 = lib["rician"](clean, 0.5, gen2)
    assert torch.allclose(out, out2)


def test_build_physics_operator_library_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown degradation families"):
        build_physics_operator_library(["not_a_real_degradation"])


def test_use_case_runs_and_ranks() -> None:
    def phantom(seed: int) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, 48), torch.linspace(-1, 1, 48), indexing="ij"
        )
        return ((xx**2 + yy**2 < 0.5).float() * 0.8).unsqueeze(0)

    vols = [("s0", phantom(0)), ("s1", phantom(1))]
    cfg = NRMetricValidationConfig(
        metrics=("hfrw", "ciea", "gri", "savs"),
        families=("rician", "gibbs"),
        n_severities_per_family=3,
        seed=7,
    )
    summary = NRMetricValidationUseCase(cfg).run(vols)
    assert "final_ranks" in summary
    assert set(summary["final_ranks"]) == set(cfg.metrics)
    assert summary["n_samples"] == 2 * 2 * 3  # subjects x families x severities


def test_config_rejects_unknown_tier() -> None:
    # pitfall #15: a knob is validated against the advertised set and raises.
    with pytest.raises(ValueError, match="unknown NR validation tier"):
        NRMetricValidationConfig(tiers=("concordance", "not_a_tier"))


def test_run_validation_full_harness_emits_cards() -> None:
    cfg = NRMetricValidationConfig(
        metrics=("gri", "csr"),
        families=("rician", "gibbs"),
        n_severities_per_family=3,
        cho_families=("rician_noise",),
        cho_n_severities=3,
        cho_n_ensemble=8,
        seed=11,
    )
    result = NRMetricValidationUseCase(cfg).run_validation(_phantoms(2))

    assert result["research_mode"] is True
    assert result["tiers_run"] == list(HARNESS_TIERS)
    # FR proxies are folded into the sweep for Tier 2 but are NOT metrics under test
    assert "psnr" in result["fr_proxies"]
    assert set(result["nr_metrics"]) == {"gri", "csr"}

    # one card per NR metric, each carrying every run tier
    assert set(result["cards"]) == {"gri", "csr"}
    for card in result["cards"].values():
        assert set(card["tiers"]) == set(HARNESS_TIERS)
        assert isinstance(card["overall_pass"], bool)

    # the meta-evaluation ranking (Tier 5) includes NR + FR columns
    assert set(result["nr_metrics"]) <= set(result["ranking"]["final_ranks"])
    # redundancy gate + aggregator block present
    assert "kept" in result["redundancy"]
    assert result["aggregator"]["backend"] in ("sklearn_ridge", "ridge_closed_form")
    # provenance stamps the resolved knobs (pitfall #15)
    assert result["provenance"]["seed"] == 11
    assert result["provenance"]["thresholds"]["icc"] == cfg.icc_threshold


def test_run_validation_tier_subset() -> None:
    # Only the requested tiers run; the aggregator can be turned off.
    cfg = NRMetricValidationConfig(
        metrics=("gri",),
        families=("rician", "gibbs"),
        n_severities_per_family=3,
        tiers=("concordance", "fr_proxy"),
        fit_aggregator=False,
        run_redundancy_gate=False,
        seed=5,
    )
    result = NRMetricValidationUseCase(cfg).run_validation(_phantoms(2))
    assert result["tiers_run"] == ["concordance", "fr_proxy"]
    assert set(result["cards"]["gri"]["tiers"]) == {"concordance", "fr_proxy"}
    assert result["aggregator"] is None
    assert result["redundancy"] is None
