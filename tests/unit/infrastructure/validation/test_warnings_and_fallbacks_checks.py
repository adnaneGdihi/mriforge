"""Unit tests for the warnings-and-fallbacks Tier-1 additions.

Each test exercises a single check against a synthetic config object
constructed with ``types.SimpleNamespace`` — no full TrainingSettings
import is needed.
"""

from __future__ import annotations

import types
from typing import Any

import pytest

from mriforge.infrastructure.validation.config_health_checker import (
    ConfigHealthChecker,
)

# ``check_legacy_schema_mixing`` tests lived here. Deleted along with the
# check itself (#933): both were structurally dead (all four legacy leaves
# resolve to None on a post-fold TrainingSettings) and the tests pinned
# that dead check as "passing" -- part of the defect, not coverage of it.
# See ``config_health_checker.py`` for the full rationale.


# ── check_early_stopping_metric_compatibility ──────────────────────────


def test_early_stopping_skipped_when_disabled() -> None:
    cfg = types.SimpleNamespace(
        early_stopping=types.SimpleNamespace(enabled=False, metric="val_psnr")
    )
    assert ConfigHealthChecker().check_early_stopping_metric_compatibility(cfg) == []


def test_early_stopping_errors_on_missing_metric() -> None:
    cfg = types.SimpleNamespace(
        early_stopping=types.SimpleNamespace(enabled=True, metric=""),
        training=None,
        data=None,
    )
    results = ConfigHealthChecker().check_early_stopping_metric_compatibility(cfg)
    failures = [r for r in results if not r.passed]
    assert len(failures) == 1
    assert failures[0].severity == "error"


def test_early_stopping_warns_on_cascading_metric_mismatch() -> None:
    cfg = types.SimpleNamespace(
        early_stopping=types.SimpleNamespace(enabled=True, metric="val_psnr"),
        training=types.SimpleNamespace(cascading_super_resolution=True),
        data=types.SimpleNamespace(),
    )
    results = ConfigHealthChecker().check_early_stopping_metric_compatibility(cfg)
    failures = [r for r in results if not r.passed]
    assert len(failures) == 1
    assert failures[0].severity == "warning"
    assert "_2x" in failures[0].fix_hint


def test_early_stopping_passes_with_correct_cascading_suffix() -> None:
    cfg = types.SimpleNamespace(
        early_stopping=types.SimpleNamespace(enabled=True, metric="val_psnr_2x"),
        training=types.SimpleNamespace(cascading_super_resolution=True),
        data=types.SimpleNamespace(),
    )
    results = ConfigHealthChecker().check_early_stopping_metric_compatibility(cfg)
    assert all(r.passed for r in results)


# ── check_metric_channel_compatibility ─────────────────────────────────


def _metrics_cfg(out_channels: int, **flags: bool) -> Any:
    return types.SimpleNamespace(
        model=types.SimpleNamespace(out_channels=out_channels),
        metrics=types.SimpleNamespace(**flags),
    )


def test_metric_channel_pass_when_fid_matches_3() -> None:
    cfg = _metrics_cfg(3, compute_fid=True, compute_lpips=False)
    assert all(r.passed for r in ConfigHealthChecker().check_metric_channel_compatibility(cfg))


def test_metric_channel_error_when_fid_on_8_channel() -> None:
    cfg = _metrics_cfg(8, compute_fid=True, compute_lpips=False)
    results = ConfigHealthChecker().check_metric_channel_compatibility(cfg)
    failures = [r for r in results if not r.passed]
    assert len(failures) == 1
    assert failures[0].severity == "error"
    assert "compute_fid" in " ".join(failures[0].yaml_keys)


def test_metric_channel_pass_when_lpips_on_1_or_3() -> None:
    for n in (1, 3):
        cfg = _metrics_cfg(n, compute_fid=False, compute_lpips=True)
        assert all(r.passed for r in ConfigHealthChecker().check_metric_channel_compatibility(cfg))


def test_metric_channel_lpips_no_longer_strict() -> None:
    """LPIPS is intentionally not in the strict-channel set.

    LPIPS auto-handles arbitrary channel counts at runtime (mean+repeat
    for ``C ∉ {1, 3}``, take[:3] for ``C > 3``), so the audit no longer
    flags it — see the ``_METRIC_CHANNEL_CONSTRAINTS`` docstring in
    ``config_health_checker.py`` for the rationale (previous strict
    check produced ~12 false positives per smoke run). This test
    locks in the current contract so a future "tighten LPIPS again"
    refactor must explicitly update this expectation.
    """
    cfg = _metrics_cfg(2, compute_fid=False, compute_lpips=True)
    failures = [
        r for r in ConfigHealthChecker().check_metric_channel_compatibility(cfg)
        if not r.passed
    ]
    assert failures == []


def test_metric_channel_fid_still_strict_on_2_channel() -> None:
    """FID stays strict — InceptionV3 cannot auto-adapt 2-channel input.

    Replaces the previous LPIPS-strictness test. FID's auto-adapt path
    only handles ``{1, 3}`` (1→repeat-to-3, 3→pass-through); anything
    else falls through to InceptionV3 unchanged and silently produces
    NaN, so the audit must catch it pre-flight.
    """
    cfg = _metrics_cfg(2, compute_fid=True, compute_lpips=False)
    failures = [
        r for r in ConfigHealthChecker().check_metric_channel_compatibility(cfg)
        if not r.passed
    ]
    assert len(failures) == 1
    assert failures[0].category == "metric_channel_compatibility"
    assert "compute_fid" in failures[0].message


# ── check_paradigm_required_fields ─────────────────────────────────────


def test_paradigm_required_pass_when_vae_field_present() -> None:
    cfg = types.SimpleNamespace(
        training=types.SimpleNamespace(
            strategy_class="mriforge.infrastructure.training.strategies.vae.VAEStrategy",
            training_mode="vae",
            vae=types.SimpleNamespace(kl_beta_end=1.0),
        )
    )
    assert all(r.passed for r in ConfigHealthChecker().check_paradigm_required_fields(cfg))


def test_paradigm_required_errors_when_vae_field_missing() -> None:
    cfg = types.SimpleNamespace(
        training=types.SimpleNamespace(
            strategy_class="my.module.VAEStrategy",
            training_mode="vae",
            vae=None,
        )
    )
    failures = [
        r for r in ConfigHealthChecker().check_paradigm_required_fields(cfg)
        if not r.passed
    ]
    assert len(failures) == 1
    assert failures[0].category == "paradigm_required_fields"
    assert "kl_beta_end" in failures[0].fix_hint


def test_paradigm_required_skips_unrelated_strategies() -> None:
    cfg = types.SimpleNamespace(
        training=types.SimpleNamespace(
            strategy_class="ReconstructionTrainingStrategy",
            training_mode="reconstruction",
        )
    )
    results = ConfigHealthChecker().check_paradigm_required_fields(cfg)
    assert all(r.passed for r in results)


# ── check_declared_losses_registered ───────────────────────────────────


def test_declared_losses_skipped_for_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """A loss with enabled=False is not required to be registered."""
    cfg = types.SimpleNamespace(
        losses=types.SimpleNamespace(
            image_losses=[types.SimpleNamespace(name="never_registered_loss", enabled=False)],
            kspace_losses=[],
            complex_losses=[],
        )
    )
    # Stub registry
    chk = ConfigHealthChecker()
    chk._loss_registry = {"l1", "l2"}
    monkeypatch.setattr(chk, "_lazy_load_registries", lambda: None)
    results = chk.check_declared_losses_registered(cfg)
    # No declared (enabled) losses → check returns nothing.
    assert results == []


def test_declared_losses_pass_when_all_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = types.SimpleNamespace(
        losses=types.SimpleNamespace(
            image_losses=[
                types.SimpleNamespace(name="l1", weight=1.0, enabled=True),
                types.SimpleNamespace(name="ssim", weight=0.5, enabled=True),
            ],
            kspace_losses=[],
            complex_losses=[],
        )
    )
    chk = ConfigHealthChecker()
    chk._loss_registry = {"l1", "l2", "ssim"}
    monkeypatch.setattr(chk, "_lazy_load_registries", lambda: None)
    results = chk.check_declared_losses_registered(cfg)
    assert all(r.passed for r in results)


def test_declared_losses_error_on_typo(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = types.SimpleNamespace(
        losses=types.SimpleNamespace(
            image_losses=[
                types.SimpleNamespace(name="complex_l1", weight=1.0, enabled=True),
            ],
            kspace_losses=[],
            complex_losses=[],
        )
    )
    chk = ConfigHealthChecker()
    chk._loss_registry = {"l1", "l2", "ssim"}
    monkeypatch.setattr(chk, "_lazy_load_registries", lambda: None)
    failures = [r for r in chk.check_declared_losses_registered(cfg) if not r.passed]
    assert len(failures) == 1
    assert failures[0].category == "declared_losses_registered"
    assert "complex_l1" in failures[0].message
    assert "image_losses" in " ".join(failures[0].yaml_keys)
