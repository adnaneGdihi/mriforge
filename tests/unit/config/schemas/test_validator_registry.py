"""Startup validators are fed the RESOLVED config, so test them with one.

`tests/unit/config/test_validator_registry_extended.py` covers the registered
rules against hand-built dicts. That is what let this bug live: a hand-built
dict is a second resolver, and it agreed with the validator right up until the
schema moved underneath both of them.

Phase 8 folded `optimization.learning_rate` to
`optimization.optimizer.learning_rate`. `_validate_learning_rate` had its
*message* repointed and its *read* left on the flat key, so
`validate_config_at_startup` failed for every arm in the corpus and no training
run could start. Its two siblings got the dual-read; this one did not.

These tests therefore resolve a real `TrainingSettings` and dump it, which is
exactly what `bootstrap.build_container` hands the registry.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import spectramr.config.schemas as _schemas
from spectramr.config.schemas.validator_registry import (
    _KSPACE_GRADEABLE_METRICS,
    _validate_batch_size_positive,
    _validate_kspace_model_config,
    _validate_learning_rate,
    _validate_loss_weights_positive,
    _validate_num_workers,
)
from spectramr.config.settings import TrainingSettings

# Anchored off the package, not the cwd, so the test runs from any directory.
REFERENCE = Path(str(_schemas.__file__)).parent / "templates" / "v1.0_reference.yaml"


@pytest.fixture(scope="module")
def resolved_dump() -> dict:
    """The dict shape the startup validators actually receive.

    `bootstrap.build_container` calls `validate_config_at_startup(config)` on a
    resolved `TrainingSettings`; anything else is a stand-in that can drift.
    """
    return TrainingSettings.from_yaml(str(REFERENCE)).model_dump(mode="json")


@pytest.mark.unit
class TestValidatorsReadTheCanonicalPaths:
    """Every path-reading validator must survive the phase 8-12 decomposition."""

    def test_learning_rate_reads_the_nested_optimizer_block(self, resolved_dump):
        # Regression: this returned ["... must be positive, got None"] on 25/25
        # sampled arms, because the flat key no longer exists after the fold.
        assert "learning_rate" not in resolved_dump["optimization"]
        assert resolved_dump["optimization"]["optimizer"]["learning_rate"] > 0
        assert _validate_learning_rate(resolved_dump) == []

    def test_batch_size_and_num_workers_stay_clean(self, resolved_dump):
        # The two siblings that were repointed correctly -- pinned so a future
        # move breaks the test rather than the corpus.
        assert _validate_batch_size_positive(resolved_dump) == []
        assert _validate_num_workers(resolved_dump) == []

    @pytest.mark.parametrize(
        "bad_lr, expect_message",
        [(0, True), (-1e-4, True), (1e-4, False)],
    )
    def test_learning_rate_still_rejects_non_positive(self, bad_lr, expect_message):
        """Anti-vacuity: the dual-read must not make the check unfailable."""
        cfg = {"optimization": {"optimizer": {"learning_rate": bad_lr}}}
        assert bool(_validate_learning_rate(cfg)) is expect_message

    def test_legacy_flat_spelling_is_still_accepted(self):
        """`fold` posture means unmigrated callers keep working."""
        assert _validate_learning_rate({"optimization": {"learning_rate": 1e-4}}) == []

    def test_absent_learning_rate_is_reported(self):
        assert _validate_learning_rate({"optimization": {}}) != []


@pytest.mark.unit
class TestLossWeightRangeReadsCanonicalLosses:
    """`_validate_loss_weights_positive` used to read `objectives` -- a block
    `extra='forbid'` rejects and 0/647 `experiments/inprogress` arms declare
    (issue #933) -- so the rule could never fire. Repointed at `losses.*`.
    """

    def test_loss_weight_range_rule_fires_on_a_canonical_out_of_range_lambda(self) -> None:
        """Repointed at losses.*, the rule must reject a lambda outside its declared range."""
        doc = {"losses": {"reconstruction": {"lambda_l1": -1.0}}}
        issues = _validate_loss_weights_positive(doc)
        assert issues, "rule is still reading a block no config can declare"

    def test_in_range_lambda_passes(self) -> None:
        doc = {"losses": {"reconstruction": {"lambda_l1": 10.0}}}
        assert _validate_loss_weights_positive(doc) == []

    def test_the_reference_config_is_clean(self, resolved_dump) -> None:
        assert _validate_loss_weights_positive(resolved_dump) == []


def _kspace_arm(metrics: dict) -> dict:
    """A k-space model (dataset_type kspace) with the given ``metrics`` block."""
    return {
        "model": {"model_type": "kspace_cold_diffusion"},
        "data": {"dataset_type": "kspace"},
        "metrics": metrics,
    }


@pytest.mark.unit
class TestKspaceMetricDomainRule:
    """`kspace_model_config` names the image-intensity metrics under a k-space domain.

    Planted violations (non-negotiable 15): one per shape the declared metric set
    can take -- the legacy ``compute_*`` flags and the ``compute`` list.
    """

    def test_ssim_flag_under_kspace_domain_warns_and_names_it(self):
        msgs = _validate_kspace_model_config(
            _kspace_arm({"domain": "kspace", "compute_ssim": True, "compute_kspace_error": True})
        )
        assert len(msgs) == 1 and "['ssim']" in msgs[0]

    def test_compute_list_under_kspace_domain_warns_and_names_the_offenders(self):
        msgs = _validate_kspace_model_config(
            _kspace_arm({"domain": "kspace", "compute": ["lpips", "psnr", "ssim"]})
        )
        assert len(msgs) == 1 and "['lpips', 'ssim']" in msgs[0]

    def test_kspace_native_flags_pass(self):
        # The diffusion Fourier-bridge lineage's shape: k-space error terms only.
        assert (
            _validate_kspace_model_config(
                _kspace_arm(
                    {
                        "domain": "kspace",
                        "compute_kspace_error": True,
                        "compute_phase_mse": True,
                        "compute_psnr": False,
                        "compute_ssim": False,
                        "compute_advanced_metrics": True,
                    }
                )
            )
            == []
        )

    def test_kspace_native_compute_list_passes(self):
        assert (
            _validate_kspace_model_config(
                _kspace_arm(
                    {"domain": "kspace", "compute": ["kspace_error", "mae", "mse", "phase_mse"]}
                )
            )
            == []
        )

    def test_compute_list_wins_over_flags(self):
        # A non-empty list is the declaration; stale flags beside it select nothing.
        assert (
            _validate_kspace_model_config(
                _kspace_arm({"domain": "kspace", "compute": ["kspace_error"], "compute_ssim": True})
            )
            == []
        )

    @pytest.mark.parametrize("domain", ["image", None])
    def test_image_or_unspecified_domain_passes(self, domain):
        assert (
            _validate_kspace_model_config(_kspace_arm({"domain": domain, "compute_ssim": True}))
            == []
        )

    def test_non_kspace_model_is_out_of_scope(self):
        doc = {
            "model": {"model_type": "unet"},
            "data": {"dataset_type": "image"},
            "metrics": {"domain": "kspace", "compute_ssim": True},
        }
        assert _validate_kspace_model_config(doc) == []

    def test_gradeable_set_is_the_complex_aware_one(self):
        assert {"psnr", "kspace_error", "phase_mse"} <= _KSPACE_GRADEABLE_METRICS
        assert not {"ssim", "lpips", "hfen", "ms_ssim"} & _KSPACE_GRADEABLE_METRICS

    def test_reference_template_passes(self, resolved_dump):
        assert _validate_kspace_model_config(resolved_dump) == []
