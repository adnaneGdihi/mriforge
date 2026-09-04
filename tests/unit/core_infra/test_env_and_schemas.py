"""Unit tests for environment resolution and config schemas."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.config.env_resolver import (
    resolve_cache_root,
    resolve_data_root,
    resolve_device,
)


class TestEnvResolver:
    """Tests for environment variable resolution logic."""

    @pytest.fixture(autouse=True)
    def clean_env(self):
        """Ensure a clean environment for each test."""
        with patch.dict(os.environ, {}, clear=True):
            yield

    def test_resolve_data_root_default(self, tmp_path):
        """Test fallback data root resolution."""
        # Use tmp_path to mock Path.home() to avoid creating actual folders in ~/.cache during tests
        with patch.object(Path, "home", return_value=tmp_path):
            root = resolve_data_root()
            assert root == tmp_path / ".cache" / "spectramr" / "data"
            assert root.exists()

    def test_resolve_data_root_env_override(self, tmp_path):
        """Test explicit environment override for data root."""
        custom_dir = tmp_path / "custom_data"
        with patch.dict(os.environ, {"SPECTRAMR_DATA_ROOT": str(custom_dir)}):
            root = resolve_data_root()
            assert root == custom_dir
            assert root.exists()

    def test_resolve_cache_root_default(self, tmp_path):
        """Test default cache root resolution."""
        with patch.object(Path, "home", return_value=tmp_path):
            root = resolve_cache_root()
            assert root == tmp_path / ".cache" / "spectramr" / "torch"
            assert root.exists()

    def test_resolve_cache_root_tmpdir(self, tmp_path):
        """Test TMPDIR cluster fallback for cache root."""
        cluster_tmp = tmp_path / "cluster_tmp"
        with patch.dict(os.environ, {"TMPDIR": str(cluster_tmp)}):
            root = resolve_cache_root()
            assert root == cluster_tmp / "spectramr_cache"
            assert root.exists()

    def test_resolve_cache_root_explicit(self, tmp_path):
        """Test explicit cache root override (highest priority)."""
        explicit_cache = tmp_path / "explicit"
        cluster_tmp = tmp_path / "cluster_tmp"
        with patch.dict(
            os.environ,
            {"TMPDIR": str(cluster_tmp), "SPECTRAMR_CACHE_ROOT": str(explicit_cache)},
        ):
            root = resolve_cache_root()
            assert root == explicit_cache
            assert root.exists()

    def test_resolve_device_default(self):
        """The unset default is `auto`, NOT `cpu`.

        Asserting "cpu" here pinned the silent CPU fallback that non-negotiable
        9b exists to forbid: a heavy pipeline must reach an accelerator or RAISE.
        `auto` is what lets `resolve_compute_device` make that decision; a
        hardcoded "cpu" default would pre-empt it and is the behaviour that once
        let a GPU-less node run ~100x slower and still report success.
        """
        assert resolve_device() == "auto"

    @pytest.mark.parametrize("device_str", ["cuda", "cpu", "auto", "CUDA", "AuTo"])
    def test_resolve_device_valid(self, device_str):
        """Test valid device overrides."""
        with patch.dict(os.environ, {"SPECTRAMR_DEVICE": device_str}):
            assert resolve_device() == device_str.lower()

    def test_resolve_device_invalid(self):
        """Test invalid device raises ValueError."""
        with patch.dict(os.environ, {"SPECTRAMR_DEVICE": "tpu"}):
            with pytest.raises(ValueError, match="Invalid device 'tpu'"):
                resolve_device()


class TestTrainingSettings:
    """Tests for TrainingSettings schema parsing and validation."""

    @pytest.fixture
    def valid_minimal_config_yaml(self, tmp_path) -> Path:
        """Create a minimal valid v6.0 config YAML for testing."""
        config_path = tmp_path / "test_config.yaml"
        yaml_content = (
            "config_version: '1.0'\n"
            "model:\n"
            "  model_type: standard_unet\n"
            "  in_channels: 2\n"
            "  out_channels: 2\n"
            "  spatial_dims: 2\n"
            "data:\n"
            "  dataset_type: m4raw\n"
            "  index_path: data/train.pkl\n"
            "  validation_index_path: data/val.pkl\n"
            "  known_dataset: m4raw_multicoil\n"
            "  patch_size: [256, 256, 1]\n"
            "  img_size: [256, 256]\n"
            "  batch_size: 4\n"
            "optimization:\n"
            "  learning_rate: 0.001\n"
            "  optimizer_type: adamw\n"
            "  scheduler:\n"
            "    warmup_steps: 10\n"
            "logging:\n"
            "  log_dir: test_logs\n"
        )
        config_path.write_text(yaml_content)
        return config_path

    def test_load_valid_config(self, valid_minimal_config_yaml):
        """Test loading a valid config seamlessly applies defaults and enforces schema."""
        settings = TrainingSettings.from_yaml(valid_minimal_config_yaml)

        # Verify specific parsed fields
        # The top-level `model_domain` alias was RETIRED on 2026-07-31 -- its
        # RENAMES posture is `raise`, not `fold` (renames.py:263, settings.py:340),
        # because it was a write-only second spelling. Asserted here on the
        # canonical home it always deferred to; the retired alias has no subject
        # left to assert against.
        assert settings.model.model_domain is None  # default
        # `device` folded to `run.device` with posture `raise` -- the top-level
        # spelling is gone, not merely relocated.
        assert settings.run.device == "cuda"  # default
        assert settings.model.model_type == "standard_unet"
        assert settings.data.loader.batch_size == 4
        assert settings.optimization.optimizer.learning_rate == 0.001

    def test_config_immutability(self, valid_minimal_config_yaml):
        """Test that settings are frozen and cannot be mutated (SSOT enforcement)."""
        settings = TrainingSettings.from_yaml(valid_minimal_config_yaml)

        with pytest.raises(ValidationError):
            settings.seed = 999  # Should be blocked

        with pytest.raises(ValidationError):
            settings.data.loader.batch_size = 10  # Nested mutation should be blocked

    def test_missing_config_version(self, tmp_path):
        """Test failure when config_version is omitted."""
        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text("model: {}\n")

        with pytest.raises(ValueError, match="config_version is required"):
            TrainingSettings.from_yaml(invalid_yaml)

    def test_unsupported_config_version(self, tmp_path):
        """Test failure for legacy config versions."""
        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text("config_version: '4.0'\n")

        with pytest.raises(ValueError, match=r"Config version 4\.0 not supported"):
            TrainingSettings.from_yaml(invalid_yaml)

    def test_file_not_found(self):
        """Test missing file raises standard error."""
        with pytest.raises(FileNotFoundError):
            TrainingSettings.from_yaml("nonexistent_path_to_config.yaml")

    def test_legacy_v5_config_rejected(self, tmp_path):
        """Legacy v5.0 configs are rejected outright (the auto-migration
        path was removed during the v6.0 cleanup — users must update the
        YAML rather than rely on a silent upgrade).
        """
        v5_yaml = tmp_path / "v5_config.yaml"
        v5_yaml.write_text(
            "config_version: '5.0'\n"
            "model: {model_type: standard_unet, in_channels: 2, out_channels: 2, spatial_dims: 2}\n"
        )

        with pytest.raises(ValueError, match=r"Config version 5\.0 not supported"):
            TrainingSettings.from_yaml(v5_yaml)
