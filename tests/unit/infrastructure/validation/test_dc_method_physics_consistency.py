"""Unit tests for ``check_dc_method_physics_consistency``.

Anchor: experiment_11_kspace_cold_diffusion mosaic triage 2026-05-28.
"""

from __future__ import annotations

from spectramr.infrastructure.validation.config_health_checker import (
    ConfigHealthChecker,
)


class _Model:
    def __init__(self, dc_method=None, set_dc_method=False):
        if set_dc_method:
            self.model_kwargs = {"dc_method": dc_method}
        else:
            self.model_kwargs = {}


class _DC:
    def __init__(self, enabled=False, method="adaptive", weight=0.25):
        self.enabled = enabled
        self.method = method
        self.weight = weight


class _Physics:
    def __init__(self, data_consistency=None):
        self.data_consistency = data_consistency


class _Config:
    def __init__(self, model, physics):
        self.model = model
        self.physics = physics


def _check(model_kwargs, phys_enabled):
    dc = _DC(enabled=phys_enabled)
    cfg = _Config(
        model=_Model(
            dc_method=model_kwargs.get("dc_method"),
            set_dc_method="dc_method" in model_kwargs,
        ),
        physics=_Physics(data_consistency=dc),
    )
    return ConfigHealthChecker().check_dc_method_physics_consistency(cfg)


class TestConsistentConfigs:
    def test_both_disabled_passes(self):
        result = _check({"dc_method": None}, phys_enabled=False)
        assert result.passed

    def test_both_enabled_with_explicit_method_passes(self):
        result = _check({"dc_method": "adaptive"}, phys_enabled=True)
        assert result.passed

    def test_model_unset_physics_enabled_passes(self):
        # SSOT reconciliation will inject method from physics.
        result = _check({}, phys_enabled=True)
        assert result.passed

    def test_model_unset_physics_disabled_passes(self):
        result = _check({}, phys_enabled=False)
        assert result.passed


class TestInconsistentConfigs:
    def test_physics_on_model_disabled_warns(self):
        """User's previous experiment_11 state: physics block claims DC
        is enabled but model_kwargs.dc_method=null disables it. After
        the 2026-05-28 dispatcher fix the disable wins — the warning
        makes that resolution visible at audit time."""
        result = _check({"dc_method": None}, phys_enabled=True)
        assert not result.passed
        assert "disable sentinel" in result.message
        assert result.severity == "warning"

    def test_physics_off_model_explicit_warns(self):
        result = _check({"dc_method": "hard"}, phys_enabled=False)
        assert not result.passed
        assert "physics.data_consistency.enabled=false" in result.message
        assert result.severity == "warning"

    def test_disable_sentinels_all_flagged(self):
        for sentinel in (None, "", "none", "off", "disabled"):
            result = _check({"dc_method": sentinel}, phys_enabled=True)
            assert not result.passed, f"sentinel {sentinel!r} did not flag"
