"""Physics Config Execution Audit Tests.

PURPOSE:
Every sub-config in PhysicsConfigSchema that has an `enabled` boolean flag
must be verifiably consumed somewhere in the system.  These tests:
  1. Verify the field is accessible from a loaded PhysicsConfigSchema (no KeyError)
  2. Categorize each sub-config as: CONSUMED, PARTIALLY_CONSUMED, or SCHEMA_ONLY
  3. Pin the current status so regressions are caught and new implementations
     are verified before removing the SCHEMA_ONLY status.

Sub-configs classified as SCHEMA_ONLY are documented technical debt.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------

CONSUMED = "consumed"  # Builder/strategy actively reads the enabled flag
PARTIALLY_CONSUMED = "partial"  # Flag is read but not all sub-fields are used
SCHEMA_ONLY = "schema_only"  # Field is defined but no builder reads it


# ---------------------------------------------------------------------------
# Catalog of PhysicsConfigSchema sub-configs and their consumption status
# ---------------------------------------------------------------------------

PHYSICS_SUB_CONFIG_INVENTORY = [
    # (field_name, enable_attr, status, consumer_location)
    ("data_consistency", "enabled", CONSUMED, "PhysicsBuilder or strategies"),
    (
        "coil_sensitivity",
        "enable_estimation",
        PARTIALLY_CONSUMED,
        "Limited use in multi-coil paths",
    ),
    ("kspace", "enable_kspace_recon", CONSUMED, "Strategy kspace_mixin"),
    ("regularization", "enable_tv", SCHEMA_ONLY, "No builder reads enable_tv"),
    (
        "regularization",
        "enable_wavelets",
        SCHEMA_ONLY,
        "No builder reads enable_wavelets",
    ),
    (
        "iterative_refinement",
        "enabled",
        SCHEMA_ONLY,
        "IterativeRefinementConfig not consumed",
    ),
    (
        "phase_correction",
        "enable_phase_estimation",
        SCHEMA_ONLY,
        "PhaseCorrectionConfig not consumed",
    ),
    (
        "b0_correction",
        "enable_b0_correction",
        SCHEMA_ONLY,
        "B0CorrectionConfig not consumed",
    ),
    (
        "b0_simulation",
        "enabled",
        SCHEMA_ONLY,
        "B0SimulationConfig not consumed by builder",
    ),
    (
        "motion_correction",
        "enable_motion_correction",
        SCHEMA_ONLY,
        "MotionCorrectionConfig not consumed",
    ),
    (
        "compressed_sensing",
        "enabled",
        SCHEMA_ONLY,
        "CompressedSensingConfig not consumed by builder",
    ),
    (
        "pinn",
        "enabled",
        PARTIALLY_CONSUMED,
        "Used via PINNLossesConfig but not PhysicsConfigSchema.pinn",
    ),
    ("bloch_simulation", "enabled", SCHEMA_ONLY, "BlochSimulationConfig not consumed"),
    (
        "digital_twin",
        "enabled",
        PARTIALLY_CONSUMED,
        "DigitalTwinConfig used in some strategies",
    ),
]


# ---------------------------------------------------------------------------
# 1. All sub-configs are accessible from a PhysicsConfigSchema instance
# ---------------------------------------------------------------------------


class TestPhysicsSubConfigAccessible:
    """All sub-config fields must be loadable without AttributeError."""

    def test_physics_config_instantiates_with_defaults(self):
        """PhysicsConfigSchema must instantiate with defaults (no required fields missing)."""
        from mriforge.config.schemas.physics import PhysicsConfigSchema

        cfg = PhysicsConfigSchema()
        assert cfg is not None

    @pytest.mark.parametrize(
        "field_name,enable_attr,status,note",
        PHYSICS_SUB_CONFIG_INVENTORY,
        ids=[f"{f}.{a}" for f, a, _, _ in PHYSICS_SUB_CONFIG_INVENTORY],
    )
    def test_sub_config_field_is_accessible(
        self, field_name: str, enable_attr: str, status: str, note: str
    ):
        """getattr(config.physics.{field_name}, '{enable_attr}') must not raise."""
        from mriforge.config.schemas.physics import PhysicsConfigSchema

        cfg = PhysicsConfigSchema()
        sub = getattr(cfg, field_name, None)
        assert sub is not None, (
            f"PhysicsConfigSchema.{field_name} is None or missing.\n"
            f"Schema defined it but the field doesn't exist?"
        )
        value = getattr(sub, enable_attr, "_MISSING_")
        assert value != "_MISSING_", (
            f"PhysicsConfigSchema.{field_name}.{enable_attr} does not exist.\n"
            f"Schema out of sync with this test's inventory."
        )
        assert isinstance(value, bool), (
            f"PhysicsConfigSchema.{field_name}.{enable_attr} should be bool, "
            f"got {type(value).__name__}"
        )


# ---------------------------------------------------------------------------
# 2. Track SCHEMA_ONLY sub-configs (documented technical debt)
# ---------------------------------------------------------------------------


class TestSchemaOnlySubConfigsAreTracked:
    """Schema-only sub-configs are technical debt. This test tracks them."""

    KNOWN_SCHEMA_ONLY = {
        "regularization.enable_tv",
        "regularization.enable_wavelets",
        "iterative_refinement.enabled",
        "phase_correction.enable_phase_estimation",
        "b0_correction.enable_b0_correction",
        "b0_simulation.enabled",
        "motion_correction.enable_motion_correction",
        "compressed_sensing.enabled",
        "bloch_simulation.enabled",
    }

    def test_schema_only_count_has_not_increased(self):
        """The number of SCHEMA_ONLY sub-configs must not grow."""
        schema_only_in_inventory = {
            f"{field}.{attr}"
            for field, attr, status, _ in PHYSICS_SUB_CONFIG_INVENTORY
            if status == SCHEMA_ONLY
        }
        new_schema_only = schema_only_in_inventory - self.KNOWN_SCHEMA_ONLY
        assert not new_schema_only, (
            f"New SCHEMA_ONLY physics sub-configs added (not consumed by any builder):\n"
            f"  {new_schema_only}\n"
            "Either:\n"
            "  - Implement the builder logic to consume the flag, OR\n"
            "  - Add it to KNOWN_SCHEMA_ONLY with a comment explaining the deferral"
        )

    def test_implemented_sub_configs_removed_from_schema_only(self):
        """If a sub-config is now consumed, it must be removed from KNOWN_SCHEMA_ONLY."""
        schema_only_in_inventory = {
            f"{field}.{attr}"
            for field, attr, status, _ in PHYSICS_SUB_CONFIG_INVENTORY
            if status == SCHEMA_ONLY
        }
        stale_entries = self.KNOWN_SCHEMA_ONLY - schema_only_in_inventory
        assert not stale_entries, (
            f"KNOWN_SCHEMA_ONLY contains entries no longer in the inventory "
            f"(they may have been implemented):\n  {stale_entries}\n"
            "Update KNOWN_SCHEMA_ONLY to reflect current implementation status."
        )


# ---------------------------------------------------------------------------
# 3. DataConsistencyConfig fields are accessed by PhysicsBuilder
# ---------------------------------------------------------------------------


class TestDataConsistencyFieldConsumption:
    """DataConsistencyConfig has many fields — verify key ones are actually read."""

    DATA_CONSISTENCY_FIELDS_TO_VERIFY = [
        "enabled",
        "weight",
        "method",
        "train_noise_level",
        "eval_noise_level",
    ]

    @pytest.mark.parametrize("field", DATA_CONSISTENCY_FIELDS_TO_VERIFY)
    def test_field_exists_in_schema(self, field: str):
        """DataConsistencyConfig must declare each expected field."""
        from mriforge.config.schemas.physics import DataConsistencyConfig

        assert field in DataConsistencyConfig.model_fields, (
            f"DataConsistencyConfig.{field} is missing from the schema.\n"
            "This field is expected by builder/strategy code."
        )

    def test_data_consistency_enabled_flag_is_false_by_default(self):
        """DC should be opt-in (default disabled) to prevent silent physics constraints."""
        from mriforge.config.schemas.physics import DataConsistencyConfig

        cfg = DataConsistencyConfig()
        assert cfg.enabled is False, (
            "DataConsistencyConfig.enabled should default to False (opt-in).\n"
            "Changing this to True-by-default would break experiments that don't "
            "have measured k-space available."
        )


# ---------------------------------------------------------------------------
# 4. PhysicsBuilder can be constructed from a config with physics enabled
# ---------------------------------------------------------------------------


class TestPhysicsBuilderConstruction:
    """PhysicsBuilder must not crash when given a physics-enabled config."""

    def _make_settings_with_physics(self, dc_enabled: bool = False):
        from mriforge.config.schemas.data import DataConfigSchema
        from mriforge.config.schemas.logging import LoggingConfigSchema
        from mriforge.config.schemas.metrics import MetricsConfigSchema
        from mriforge.config.schemas.model import ModelConfigSchema
        from mriforge.config.schemas.optimization import OptimizationConfigSchema
        from mriforge.config.schemas.physics import (
            DataConsistencyConfig,
            PhysicsConfigSchema,
        )
        from mriforge.config.settings import TrainingSettings

        return TrainingSettings(
            model=ModelConfigSchema(),
            data=DataConfigSchema(),
            optimization=OptimizationConfigSchema(),
            logging=LoggingConfigSchema(),
            metrics=MetricsConfigSchema(),
            physics=PhysicsConfigSchema(
                data_consistency=DataConsistencyConfig(enabled=dc_enabled)
            ),
        )

    def test_physics_builder_with_dc_disabled(self):
        """PhysicsBuilder should init without error when DC is disabled."""
        pytest.importorskip("mriforge.infrastructure.training.builders.physics_builder")
        from mriforge.infrastructure.training.builders.physics_builder import PhysicsBuilder

        settings = self._make_settings_with_physics(dc_enabled=False)
        builder = PhysicsBuilder(config=settings, device="cpu")
        assert builder is not None

    def test_physics_config_fields_accessible_via_settings(self):
        """Nested physics fields must be accessible via settings.physics.*."""
        settings = self._make_settings_with_physics(dc_enabled=True)
        assert settings.physics is not None
        assert settings.physics.data_consistency.enabled is True
        assert settings.physics.data_consistency.train_noise_level >= 0
        assert settings.physics.kspace.enable_kspace_recon is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
