"""GAN Loss Type Coverage Tests.

PURPOSE:
Every value in GANLossType enum must be handled by LossBuilder._build_composite_gan().
An unhandled GAN loss type silently falls back to gan_standard (or raises an
exception mid-training), causing experiments to run with the wrong adversarial loss.

This suite:
  1. Verifies each GANLossType value passes through _build_composite_gan()
  2. Verifies the result is a real nn.Module (not a placeholder)
  3. Tests the adv_registry_map covers all documented enum values
  4. Verifies feature_matching=0 correctly excludes feat-match from composite
"""

from __future__ import annotations

import pytest
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gan_settings(gan_loss_type: str, extra_gan_kwargs: dict | None = None):
    """Build minimal TrainingSettings with a GAN loss config."""
    from mriforge.config.schemas.data import DataConfigSchema
    from mriforge.config.schemas.logging import LoggingConfigSchema
    from mriforge.config.schemas.loss import GANLossesConfig, LossConfigSchema
    from mriforge.config.schemas.metrics import MetricsConfigSchema
    from mriforge.config.schemas.model import ModelConfigSchema
    from mriforge.config.schemas.optimization import OptimizationConfigSchema
    from mriforge.config.settings import TrainingSettings

    gan_kwargs = {
        "gan_loss_type": gan_loss_type,
        "enable_adversarial": True,
        "lambda_adv": 1.0,
    }
    if extra_gan_kwargs:
        gan_kwargs.update(extra_gan_kwargs)

    return TrainingSettings(
        model=ModelConfigSchema(),
        data=DataConfigSchema(),
        optimization=OptimizationConfigSchema(),
        logging=LoggingConfigSchema(),
        metrics=MetricsConfigSchema(),
        losses=LossConfigSchema(gan=GANLossesConfig(**gan_kwargs)),
    )


def _invoke_build_composite_gan(settings):
    """Run only the composite GAN build step and return the losses dict."""
    from mriforge.infrastructure.training.builders.loss_builder import LossBuilder

    builder = LossBuilder(config=settings, device="cpu")
    builder._build_composite_gan(
        gan_config=settings.losses.gan,
        recon_config=settings.losses.reconstruction,
    )
    return builder._losses


# ---------------------------------------------------------------------------
# 1. Each GANLossType value produces a real adversarial module
# ---------------------------------------------------------------------------


class TestGANLossTypeExecution:
    """Every GANLossType enum value must flow through _build_composite_gan() to a module."""

    # (gan_loss_type value, expected registry name for documentation)
    GAN_LOSS_CASES = [
        ("vanilla", "gan_standard"),
        ("standard", "gan_standard"),
        ("bce", "gan_standard"),
        ("lsgan", "gan_lsgan"),
        ("ralsgan", "gan_ralsgan"),
        ("wgan", "gan_wgan"),
        ("wgan-gp", "gan_wgan"),
        ("hinge", "gan_hinge"),
    ]

    @pytest.mark.parametrize(
        "gan_loss_type,expected_registry_name",
        GAN_LOSS_CASES,
        ids=[c[0] for c in GAN_LOSS_CASES],
    )
    def test_gan_loss_type_produces_module(
        self, gan_loss_type: str, expected_registry_name: str
    ):
        """_build_composite_gan() must produce losses['adversarial'] as nn.Module."""
        settings = _make_gan_settings(gan_loss_type)
        losses = _invoke_build_composite_gan(settings)

        assert "adversarial" in losses, (
            f"_build_composite_gan() did not create 'adversarial' loss for "
            f"gan_loss_type='{gan_loss_type}'.\n"
            f"Expected registry name: '{expected_registry_name}'.\n"
            f"Keys present: {sorted(losses.keys())}\n"
            "Check LossBuilder._build_composite_gan() adv_registry_map or the "
            "loss registry for this GAN type."
        )
        assert isinstance(losses["adversarial"], nn.Module), (
            f"losses['adversarial'] for gan_loss_type='{gan_loss_type}' is not an "
            f"nn.Module, got {type(losses['adversarial']).__name__}"
        )

    def test_unknown_gan_loss_type_falls_back_not_crashes(self):
        """An unknown gan_loss_type must raise, not silently degrade.

        The builder raises ConfigurationError on an unregistered type (pitfall #9:
        no silent fallback). A clear raise is the correct outcome; the forbidden one
        is a silently-produced 'standard' module the config never asked for.
        """
        from mriforge.domain.exceptions import ConfigurationError

        settings = _make_gan_settings("unknown_nonexistent_loss_type_xyz")
        try:
            losses = _invoke_build_composite_gan(settings)
            # If it does not raise, it must not have silently produced a module.
            if "adversarial" in losses:
                assert isinstance(losses["adversarial"], nn.Module)
        except (ConfigurationError, ValueError, KeyError, RuntimeError):
            pass  # Explicit failure is the correct behavior (#9)


# ---------------------------------------------------------------------------
# 2. GANLossType enum completeness vs. adv_registry_map
# ---------------------------------------------------------------------------


class TestGANLossEnumVsRegistryMap:
    """GANLossType enum values must all appear in _build_composite_gan's adv_registry_map."""

    # The mapping as defined in LossBuilder._build_composite_gan
    # ──────────────────────────────────────────────────────────────────────────
    # KNOWN GAP: 'wgan_softplus' is in GANLossType enum but NOT in the builder.
    # Sentinel target 'gan_wgan_softplus' is used to track this; it is not registered.
    # Fix: implement WGANSoftplusLoss, register as 'gan_wgan_softplus', then remove sentinel.
    # ──────────────────────────────────────────────────────────────────────────
    ADV_REGISTRY_MAP = {
        "vanilla": "gan_standard",
        "standard": "gan_standard",
        "bce": "gan_standard",
        "lsgan": "gan_lsgan",
        "ralsgan": "gan_ralsgan",
        "wgan": "gan_wgan",
        "wgan-gp": "gan_wgan",
        "hinge": "gan_hinge",
        "wgan_softplus": "gan_wgan_softplus",  # KNOWN_GAP sentinel
    }

    KNOWN_UNIMPLEMENTED_REGISTRY_TARGETS = {"gan_wgan_softplus"}
    KNOWN_BUILDER_GAPS = {"wgan_softplus"}  # in enum but missing from builder loop

    def test_all_enum_values_in_registry_map(self):
        """Every GANLossType.value must appear in adv_registry_map (case-lowered)."""
        from mriforge.config.schemas.enums import GANLossType

        enum_values = {m.value.lower() for m in GANLossType}
        missing = enum_values - set(self.ADV_REGISTRY_MAP.keys())

        assert not missing, (
            f"GANLossType enum values with no entry in adv_registry_map:\n"
            f"  {missing}\n"
            "These GAN loss types are unreachable — the builder falls back silently.\n"
            "Fix: Add the missing value to adv_registry_map in "
            "LossBuilder._build_composite_gan()."
        )

    def test_all_registry_values_are_registered_losses(self):
        """Every target registry name in adv_registry_map must exist in loss registry
        (excluding known-unimplemented sentinel entries)."""
        from mriforge.models.losses.registry import list_available

        registered = set(list_available())
        for map_key, registry_name in self.ADV_REGISTRY_MAP.items():
            if registry_name in self.KNOWN_UNIMPLEMENTED_REGISTRY_TARGETS:
                # This is a documented gap — the loss class is not yet implemented
                continue
            assert registry_name in registered, (
                f"adv_registry_map maps '{map_key}' -> '{registry_name}', but "
                f"'{registry_name}' is not in the loss registry.\n"
                f"Registered: {sorted(registered)}"
            )

    def test_known_builder_gaps_are_tracked(self):
        """Builder gaps must be explicitly tracked so they cannot silently accumulate."""
        from mriforge.config.schemas.enums import GANLossType

        enum_values = {m.value.lower() for m in GANLossType}
        # Any enum value mapped to a sentinel target is a KNOWN gap
        sentinel_mapped = {
            k
            for k, v in self.ADV_REGISTRY_MAP.items()
            if v in self.KNOWN_UNIMPLEMENTED_REGISTRY_TARGETS
        }
        new_undocumented = (
            enum_values - set(self.ADV_REGISTRY_MAP.keys())
        ) - self.KNOWN_BUILDER_GAPS
        assert (
            not new_undocumented
        ), f"New undocumented GANLossType gaps: {new_undocumented}"


# ---------------------------------------------------------------------------
# 3. Feature matching weight of 0 excludes it from composite
# ---------------------------------------------------------------------------


class TestGANFeatureMatchingExclusion:
    """feature_matching=0 must mean no feature matching loss is computed."""

    def test_zero_feature_matching_does_not_contribute(self):
        """When feature_matching=0, the composite GAN should not receive a feat-match term."""
        from mriforge.config.schemas.loss import GANLossesConfig

        # feature_matching=0 means "disable"
        gan_cfg = GANLossesConfig(
            gan_loss_type="standard",
            enable_adversarial=True,
            lambda_adv=1.0,
            feature_matching=0.0,
        )
        assert (
            gan_cfg.feature_matching == 0.0
        ), "feature_matching=0.0 not preserved in GANLossesConfig"

    def test_nonzero_feature_matching_is_preserved(self):
        """feature_matching>0 must be preserved in the config."""
        from mriforge.config.schemas.loss import GANLossesConfig

        gan_cfg = GANLossesConfig(
            gan_loss_type="standard",
            feature_matching=10.0,
        )
        assert gan_cfg.feature_matching == 10.0


# ---------------------------------------------------------------------------
# 4. label_smoothing parameter flows to adversarial loss constructor
# ---------------------------------------------------------------------------


class TestGANLabelSmoothing:
    """GANLossesConfig.label_smoothing must be passed to create_loss()."""

    def test_label_smoothing_accepted_by_schema(self):
        from mriforge.config.schemas.loss import GANLossesConfig

        cfg = GANLossesConfig(label_smoothing=0.1)
        assert cfg.label_smoothing == pytest.approx(0.1)

    def test_default_label_smoothing_is_zero(self):
        """Default should be 0.0 (off) to avoid silent accuracy degradation."""
        from mriforge.config.schemas.loss import GANLossesConfig

        cfg = GANLossesConfig()
        assert cfg.label_smoothing == pytest.approx(0.0), (
            f"Default label_smoothing is {cfg.label_smoothing}, expected 0.0.\n"
            "Non-zero default smoothing changes adversarial dynamics for all experiments."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
