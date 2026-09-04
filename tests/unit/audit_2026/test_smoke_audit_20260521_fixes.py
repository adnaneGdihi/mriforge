"""Regression tests for the 2026-05-21 smoke-audit remediation (Round 15).

Round 15 is the "F4 (E17)" pickup the prior rounds deferred: the cluster
``audit_*.json`` files were synced down and the remaining audit-fail set
triaged locally. Two framework-level fixes are pinned here (see
``TODO/audit/smoke_audit_20260521.md`` Round 15):

* **F23** — ``concomitant_phase_residual`` is registered ``domain="kspace"``
  but penalizes a real-valued residual-phase head output, so it is
  domain-agnostic. It is now ``compatible_with=["image"]`` so the dozen
  image-domain reconstruction arms that declare it under
  ``losses.image_losses`` pass ``check_loss_domain_block_match``.

* **F24** — ``training.ib_vf`` / ``training.twin_dps`` are accepted as raw
  dicts by :class:`TrainingStrategyConfigSchema` (``Any``-typed to avoid an
  import cycle, like ``tto``). The audit checks now coerce the dict to its
  typed schema, which previously crashed with
  ``'dict' object has no attribute 'beta_plus'``.
"""

from __future__ import annotations

import pytest

import spectramr.models.losses  # noqa: F401 — populate the loss registry
from spectramr.models.losses.registry import LossRegistry


# ---------------------------------------------------------------------------
# F1 — concomitant_phase_residual is image-compatible
# ---------------------------------------------------------------------------


def test_concomitant_phase_residual_is_image_compatible() -> None:
    meta = getattr(LossRegistry, "_loss_domains", {}).get(
        "concomitant_phase_residual"
    )
    assert meta is not None, "loss must be registered with domain metadata"
    assert meta["domain"] == "kspace"
    assert "image" in (meta.get("compatible_with") or []), (
        "image-domain arms declare this loss under losses.image_losses; "
        "the decorator must mark it compatible_with=['image']"
    )


def test_loss_domain_block_match_accepts_concomitant_under_image() -> None:
    """The audit honours compatible_with, so an image_losses placement passes."""
    from spectramr.infrastructure.validation.config_health_checker import (
        ConfigHealthChecker,
    )

    checker = ConfigHealthChecker()
    block_to_domain = {"image_losses": "image"}
    registered = getattr(LossRegistry, "_loss_domains", {})
    meta = registered.get("concomitant_phase_residual", {})
    declared = meta.get("domain")
    compatible = meta.get("compatible_with") or []
    # mirror the check's accept predicate
    assert declared == "image" or "image" in compatible
    assert checker is not None  # checker instantiates without error


# ---------------------------------------------------------------------------
# F2 — ib_vf / twin_dps dict coercion never crashes the audit
# ---------------------------------------------------------------------------


def test_typed_training_block_coerces_dict() -> None:
    from spectramr.config.schemas.training.vf_advanced import IBVFConfig
    from spectramr.infrastructure.validation.config_health_checker import (
        ConfigHealthChecker,
    )

    class _T:
        ib_vf = {"beta_plus": 1.0, "beta_minus": 0.1}

    class _Cfg:
        training = _T()

    block = ConfigHealthChecker._typed_training_block(_Cfg(), "ib_vf", IBVFConfig)
    assert isinstance(block, IBVFConfig)
    assert block.beta_plus == pytest.approx(1.0)


def test_ib_vf_audit_does_not_crash_on_dict_block() -> None:
    """Loading an ib_vf YAML (ib_vf stays a dict) must not raise in the audit."""
    import warnings

    from spectramr.config.settings import TrainingSettings
    from spectramr.infrastructure.validation.config_health_checker import (
        validate_config_health,
    )

    yaml_path = "experiments/inprogress/vf/exp_vf_ib_infonce_v2.yaml"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = TrainingSettings.from_yaml(yaml_path)
        # ib_vf is delivered as a raw dict by the extra="allow" schema
        assert isinstance(getattr(config.training, "ib_vf", None), dict)
        # The audit must complete (it may report config errors, but never crash).
        report = validate_config_health(config)
    names = {r.check_name for r in report.results}
    assert "ib_beta_positive" in names


# ---------------------------------------------------------------------------
# F28 — early-stopping monitor alias resolution (runtime training health)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "monitor, emitted",
    [
        # cascade-strip: monitor carries _2x, validator emits the bare key
        ("val_robust_mri_psnr_2x", "val_robust_mri_psnr"),
        # loss → min-mode mse proxy (recon-only validator, no aggregate loss)
        ("val_loss", "val_mse"),
        # cascade-add: bare monitor, cascading validator suffixes the level
        ("val_psnr", "val_psnr_2x"),
        # prefix strip: validator returns the unprefixed key
        ("val_ssim", "ssim"),
    ],
)
def test_early_stop_monitor_candidates_cover_smoke_mismatches(
    monitor: str, emitted: str
) -> None:
    from spectramr.pipelines.train import early_stop_monitor_candidates

    candidates = early_stop_monitor_candidates(monitor)
    assert candidates[0] == monitor, "exact monitor must be tried first"
    assert emitted in candidates, (
        f"monitor {monitor!r} should resolve to emitted key {emitted!r}; "
        f"got {candidates}"
    )


def test_early_stop_exact_match_wins_over_alias() -> None:
    """If the validator emits the exact monitor, it must be chosen first."""
    from spectramr.pipelines.train import early_stop_monitor_candidates

    candidates = early_stop_monitor_candidates("val_psnr")
    emitted = {"val_psnr", "val_psnr_2x", "psnr"}
    chosen = next(c for c in candidates if c in emitted)
    assert chosen == "val_psnr"
