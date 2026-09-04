r"""Augmentation Schema Coverage Tests.

PURPOSE:
``AugmentationConfigSchema`` exposes 18 ``enable_*`` flags.  Whether each one
is *wired* is owned by ``tests/utils/augmentation_coverage.py`` and asserted
behaviourally in ``tests/unit/data/transforms/test_augmentation_factory.py``.
This suite no longer answers that question a second way (non-negotiable 17):
it used to compute ``all_flags - CONSUMED_BY_FACTORY`` -- deriving "actually
unconsumed" from the ledger it was auditing, so it could never notice a flag
becoming consumed -- and to score a flag "consumed" for merely appearing in a
``\.enable_(\w+)`` regex over the factory source.  Both ledgers were stale and
both suites were green while eight flags did nothing.

What is left here is what the behavioural census does not cover:

* the deleted ``augmentation_pipeline`` module really is gone (it built a
  torchvision ``T.Compose`` whose per-call randomness would have given
  ``input`` and ``target`` different geometry -- the leak
  ``tests/data_integrity/test_augmentation_leak.py`` exists to catch);
* the flags it used to read are now genuinely wired, closing that record;
* every non-``enable_`` parameter field is paired with a flag or declared an
  intentional orphan.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from tests.utils.augmentation_coverage import (
    NOT_PER_SAMPLE,
    enable_flags,
    unwired_flags,
)

# Flags the deleted AugmentationPipeline used to read.  Kept as a named set so
# the delete stays auditable.  Deleting it did not orphan them -- it revealed
# that they already were, since the module had no production caller.  All but
# one are wired now; the exception is recorded in NOT_PER_SAMPLE.
FORMERLY_CONSUMED_BY_DELETED_PIPELINE = {
    "enable_rician_noise",
    "enable_motion_blur",
    "enable_kspace_undersampling_augmentation",
    "enable_brightness",
    "enable_contrast",
    "enable_gamma",
    "enable_noise",
}


class TestTheDeletedPipelineIsResolved:
    """The delete is real, and the flags it stranded have since been wired."""

    def test_the_deleted_pipeline_is_really_gone(self):
        """Not a dangling import waiting to be re-added."""
        import importlib

        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("spectramr.data.transforms.augmentation_pipeline")

    def test_the_formerly_stranded_flags_are_still_schema_fields(self):
        """Anti-staleness: a renamed field would silently empty this record."""
        assert set(enable_flags()) >= FORMERLY_CONSUMED_BY_DELETED_PIPELINE

    def test_all_but_the_deferred_one_are_now_wired(self):
        """The gap this suite used to xfail is closed.

        ``enable_bias_field`` and ``enable_blur`` -- previously xfailed here
        with "Setting it in YAML has NO effect" -- are wired too, so they are
        covered by the census rather than pinned as known gaps.
        """
        still_unwired = FORMERLY_CONSUMED_BY_DELETED_PIPELINE & unwired_flags()
        assert still_unwired == {"enable_kspace_undersampling_augmentation"}, (
            "flags stranded by the augmentation_pipeline delete that are still "
            f"unwired: {sorted(still_unwired)}"
        )

    def test_the_one_exception_is_a_documented_deferral(self):
        assert "enable_kspace_undersampling_augmentation" in NOT_PER_SAMPLE


# ─────────────────────────────────────────────────────────────────────────────
# 3. Non-enable fields that are schema-orphans (parameter with no enable flag)
# ─────────────────────────────────────────────────────────────────────────────


class TestAugmentationParameterOrphans:
    """Parameter fields (non-enable) must have a corresponding enable_ flag."""

    # All non-enable parameter fields have an enable_* counterpart (paired parameters)
    # or are global control fields — all intentional
    INTENTIONAL_ORPHAN_PARAMS: ClassVar[set[str]] = {
        "enabled",  # top-level on/off switch
        "probability",  # global augmentation probability
        "transforms",  # custom transform dict (arbitrary)
        "intensity_range",  # normalization parameter, not per-augmentation
        # Spatial parameters (paired with enable_rotation, enable_flip, enable_elastic_deformation)
        "rotation_range",
        "prob_rotate",
        "prob_flip",
        "flip_axes",
        "elastic_alpha",
        "elastic_sigma",
        # Intensity parameters (paired with enable_noise, enable_brightness, enable_contrast, enable_gamma)
        "noise_std",
        "brightness_range",
        "contrast_range",
        "gamma_range",
        # MRI-specific parameters (paired with corresponding enable_ flags)
        "rician_noise_level",
        "motion_blur_intensity",
        "b0_max_displacement",
        "b0_prob",
        "bias_field_coefficients",  # parameter for enable_bias_field (schema-only flag)
        "blur_std",  # parameter for enable_blur (schema-only flag)
        "num_ghosts",
        "ghosting_intensity",
        "ghosting_axes",
        "num_spikes",
        "spike_intensity",
        "anisotropy_downsampling",
        "undersampling_factor_augmentation",
        # Advanced augmentation parameters (paired with schema-only flags)
        "mixup_alpha",  # parameter for enable_mixup (schema-only)
        "cutmix_alpha",  # parameter for enable_cutmix (schema-only)
    }

    def test_all_non_enable_fields_classified(self):
        """Every non-enable AugmentationConfigSchema field must be either:
        - Paired with an enable_* flag
        - Listed in INTENTIONAL_ORPHAN_PARAMS
        """
        from spectramr.config.schemas.augmentation import AugmentationConfigSchema

        all_fields = set(AugmentationConfigSchema.model_fields.keys())
        enable_flags = {f for f in all_fields if f.startswith("enable_")}
        non_enable = all_fields - enable_flags

        orphans = non_enable - self.INTENTIONAL_ORPHAN_PARAMS
        assert not orphans, (
            f"Non-enable AugmentationConfigSchema fields with no enable_* counterpart "
            f"and not in INTENTIONAL_ORPHAN_PARAMS:\n  {sorted(orphans)}\n"
            "Either add a corresponding enable_* flag or add to INTENTIONAL_ORPHAN_PARAMS."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
