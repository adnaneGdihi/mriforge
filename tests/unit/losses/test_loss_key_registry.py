"""Unit tests for loss_key_registry.py.

Covers: LossKeyRegistry, LossKeyValidator.
"""

from __future__ import annotations

import pytest

from spectramr.models.losses.loss_key_registry import LossKeyRegistry, LossKeyValidator


class TestLossKeyRegistry:

    # ── Canary ──────────────────────────────────────────────────────────────

    def test_canary_get_all_loss_keys_returns_dict(self):
        keys = LossKeyRegistry.get_all_loss_keys()
        assert isinstance(keys, dict) and len(keys) > 0

    # ── Parametric ──────────────────────────────────────────────────────────

    @pytest.mark.parametrize("mode", ["gan", "diffusion", "vae", "vqvae", "reconstruction"])
    def test_parametric_mode_keys(self, mode: str):
        keys = LossKeyRegistry.get_training_mode_keys(mode)
        assert isinstance(keys, dict) and len(keys) > 0

    @pytest.mark.parametrize("mode, expected_prefix", [
        ("gan", "g_total_loss"),
        ("diffusion", "diffusion_total_loss"),
        ("vae", "vae_total_loss"),
        ("vqvae", "vqvae_total_loss"),
        ("reconstruction", "reconstruction_total_loss"),
    ])
    def test_parametric_total_loss_key(self, mode: str, expected_prefix: str):
        key = LossKeyRegistry.get_total_loss_key(mode)
        assert key == expected_prefix

    # ── Property ────────────────────────────────────────────────────────────

    def test_property_all_keys_are_strings(self):
        keys = LossKeyRegistry.get_all_loss_keys()
        for k, v in keys.items():
            assert isinstance(k, str) and isinstance(v, str)

    def test_property_class_constants_in_all_keys(self):
        all_keys = LossKeyRegistry.get_all_loss_keys()
        assert LossKeyRegistry.GAN_TOTAL_LOSS in all_keys
        assert LossKeyRegistry.DIFFUSION_TOTAL_LOSS in all_keys
        assert LossKeyRegistry.VAE_TOTAL_LOSS in all_keys

    def test_property_validate_accepts_standard_keys(self):
        losses = {LossKeyRegistry.GAN_TOTAL_LOSS: None}
        result = LossKeyRegistry.validate_loss_dict(losses, "gan")
        assert result is True

    def test_property_validate_accepts_metric_prefix(self):
        losses = {"metric_custom_foo": None}
        result = LossKeyRegistry.validate_loss_dict(losses, "gan")
        assert result is True

    # ── Edge ────────────────────────────────────────────────────────────────

    def test_edge_unknown_mode_returns_empty(self):
        keys = LossKeyRegistry.get_training_mode_keys("unknown_mode_xyz")
        assert isinstance(keys, dict) and len(keys) == 0

    def test_edge_unknown_mode_total_key_fallback(self):
        key = LossKeyRegistry.get_total_loss_key("unknown_mode_xyz")
        assert key == "total_loss"

    # ── Raises ──────────────────────────────────────────────────────────────

    def test_raises_nonstandard_key(self):
        with pytest.raises(ValueError, match="non-standard|Non-standard"):
            LossKeyRegistry.validate_loss_dict({"completely_invalid_loss_key_xyz": None}, "gan")


class TestLossKeyValidator:

    # ── Canary ──────────────────────────────────────────────────────────────

    def test_canary_non_strict_valid(self):
        validator = LossKeyValidator("gan")
        result = validator.validate({"g_total_loss": 0.0}, strict=False)
        assert result is True

    # ── Parametric ──────────────────────────────────────────────────────────

    @pytest.mark.parametrize("mode", ["gan", "reconstruction"])
    def test_parametric_modes_validate_known_keys(self, mode: str):
        validator = LossKeyValidator(mode)
        all_keys = LossKeyRegistry.get_training_mode_keys(mode)
        if all_keys:
            sample_key = next(iter(all_keys))
            result = validator.validate({sample_key: 0.0})
            assert result is True

    # ── Property ────────────────────────────────────────────────────────────

    def test_property_strict_requires_total_key(self):
        validator = LossKeyValidator("gan")
        with pytest.raises(ValueError, match="total"):
            validator.validate({"g_loss_l1": 0.0}, strict=True)

    def test_property_strict_passes_with_total_key(self):
        validator = LossKeyValidator("gan")
        result = validator.validate({"g_total_loss": 0.0}, strict=True)
        assert result is True

    # ── Edge ────────────────────────────────────────────────────────────────

    def test_edge_empty_dict(self):
        validator = LossKeyValidator("reconstruction")
        result = validator.validate({}, strict=False)
        assert result is True

    # ── Raises ──────────────────────────────────────────────────────────────

    def test_raises_unknown_key(self):
        validator = LossKeyValidator("gan")
        with pytest.raises(ValueError, match="Unknown"):
            validator.validate({"totally_bogus_key_xyz": 0.0})
