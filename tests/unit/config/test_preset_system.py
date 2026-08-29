"""Unit tests for configuration preset system.

Tests for:
- PresetRegistry singleton pattern and O(1) lookups
- ConfigPreset loading and overrides
- Auto-discovery of presets from experiments/
- Filtering by category, tag, and pattern
"""

import tempfile
from pathlib import Path

import pytest

from mriforge.config.schemas.base import CANONICAL_CONFIG_VERSION
from mriforge.config.presets import ConfigPreset, PresetRegistry, get_baseline_presets


class TestPresetRegistry:
    """Test PresetRegistry singleton pattern and core functionality."""

    def test_singleton_pattern(self) -> None:
        """PresetRegistry should return same instance on multiple calls."""
        registry1 = PresetRegistry()
        registry2 = PresetRegistry()
        assert registry1 is registry2, "PresetRegistry should be a singleton"

    def test_register_and_get(self) -> None:
        """Should register preset and retrieve it by name."""
        registry = PresetRegistry()
        registry._presets.clear()  # Clear for test isolation

        # Create minimal test preset
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("config_version: '5.0'\n")
            f.write("experiment:\n")
            f.write("  name: test_preset\n")
            f.write("data:\n")
            f.write("  batch_size: 32\n")
            f.write("optimization:\n")
            f.write("  learning_rate: 0.001\n")
            f.write("  epochs: 100\n")
            f.write("model:\n")
            f.write("  model_type: standard_unet\n")
            f.write("training:\n")
            f.write("  training_mode: gan\n")
            f.flush()

            preset = ConfigPreset(
                name="test_gan",
                category="gan",
                description="Test GAN preset",
                yaml_path=f.name,
                tags=["baseline", "test"],
            )
            registry.register(preset)

            retrieved = registry.get("test_gan")
            assert retrieved.name == "test_gan"
            assert retrieved.category == "gan"

        Path(f.name).unlink()

    def test_get_nonexistent_preset_raises_error(self) -> None:
        """Should raise KeyError with helpful message for missing preset."""
        registry = PresetRegistry()
        registry._presets.clear()

        with pytest.raises(KeyError) as exc_info:
            registry.get("nonexistent_preset")
        assert "nonexistent_preset" in str(exc_info.value)

    def test_filter_by_category(self) -> None:
        """Should filter presets by category."""
        registry = PresetRegistry()
        registry._presets.clear()
        registry._categories.clear()
        registry._tags.clear()

        # Create test presets
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, (category, prefix) in enumerate([("gan", "g"), ("diffusion", "d")]):
                for j in range(2):
                    yaml_file = Path(tmpdir) / f"{prefix}_{i}_{j}.yaml"
                    yaml_file.write_text(
                        "config_version: '5.0'\n"
                        "experiment:\n"
                        "  name: test\n"
                        "data:\n"
                        "  batch_size: 32\n"
                        "optimization:\n"
                        "  learning_rate: 0.001\n"
                        "  epochs: 100\n"
                        "model:\n"
                        "  model_type: standard_unet\n"
                        "training:\n"
                        "  training_mode: gan\n"
                    )

                    preset = ConfigPreset(
                        name=f"{prefix}_{i}_{j}",
                        category=category,
                        description=f"Test preset {prefix}_{i}_{j}",
                        yaml_path=str(yaml_file),
                    )
                    registry.register(preset)

            # Filter by category
            gan_presets = registry.filter(category="gan")
            assert len(gan_presets) == 2
            assert all(p.category == "gan" for p in gan_presets)

            diffusion_presets = registry.filter(category="diffusion")
            assert len(diffusion_presets) == 2
            assert all(p.category == "diffusion" for p in diffusion_presets)

    def test_filter_by_tag(self) -> None:
        """Should filter presets by tag."""
        registry = PresetRegistry()
        registry._presets.clear()
        registry._categories.clear()
        registry._tags.clear()

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yaml"
            yaml_file.write_text(
                "config_version: '5.0'\n"
                "experiment:\n"
                "  name: test\n"
                "data:\n"
                "  batch_size: 32\n"
                "optimization:\n"
                "  learning_rate: 0.001\n"
                "  epochs: 100\n"
                "model:\n"
                "  model_type: standard_unet\n"
                "training:\n"
                "  training_mode: gan\n"
            )

            # Create presets with different tags
            for tag in ["baseline", "ablation", "robustness"]:
                preset = ConfigPreset(
                    name=f"test_{tag}",
                    category="gan",
                    description=f"Test preset {tag}",
                    yaml_path=str(yaml_file),
                    tags=[tag],
                )
                registry.register(preset)

            baseline_presets = registry.filter(tag="baseline")
            assert len(baseline_presets) == 1
            assert baseline_presets[0].tags == ["baseline"]

    def test_list_all(self) -> None:
        """Should list all registered presets."""
        registry = PresetRegistry()
        registry._presets.clear()

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yaml"
            yaml_file.write_text(
                "config_version: '5.0'\n"
                "experiment:\n"
                "  name: test\n"
                "data:\n"
                "  batch_size: 32\n"
                "optimization:\n"
                "  learning_rate: 0.001\n"
                "  epochs: 100\n"
                "model:\n"
                "  model_type: standard_unet\n"
                "training:\n"
                "  training_mode: gan\n"
            )

            for i in range(3):
                preset = ConfigPreset(
                    name=f"preset_{i}",
                    category="gan",
                    description=f"Test preset {i}",
                    yaml_path=str(yaml_file),
                )
                registry.register(preset)

            all_presets = registry.list_all()
            assert len(all_presets) == 3
            assert all(isinstance(p, ConfigPreset) for p in all_presets)

    def test_list_categories(self) -> None:
        """Should list all unique categories."""
        registry = PresetRegistry()
        registry._presets.clear()
        registry._categories.clear()

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yaml"
            yaml_file.write_text(
                "config_version: '5.0'\n"
                "experiment:\n"
                "  name: test\n"
                "data:\n"
                "  batch_size: 32\n"
                "optimization:\n"
                "  learning_rate: 0.001\n"
                "  epochs: 100\n"
                "model:\n"
                "  model_type: standard_unet\n"
                "training:\n"
                "  training_mode: gan\n"
            )

            for category in ["gan", "diffusion", "reconstruction"]:
                preset = ConfigPreset(
                    name=f"preset_{category}",
                    category=category,
                    description=f"Test preset for {category}",
                    yaml_path=str(yaml_file),
                )
                registry.register(preset)

            categories = registry.list_categories()
            assert set(categories) == {"gan", "diffusion", "reconstruction"}


class TestConfigPreset:
    """Test ConfigPreset wrapper for YAML presets."""

    def test_load_config(self) -> None:
        """Should load YAML config from preset file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("config_version: '1.0'\n")
            f.write("metadata:\n")
            f.write("  experiment_name: test_experiment\n")
            f.write("data:\n")
            f.write("  batch_size: 32\n")
            f.write("optimization:\n")
            f.write("  learning_rate: 0.001\n")
            f.write("  epochs: 100\n")
            f.write("model:\n")
            f.write("  model_type: standard_unet\n")
            f.write("training:\n")
            f.write("  training_mode: gan\n")
            f.flush()

            preset = ConfigPreset(
                name="test_preset",
                category="gan",
                description="Test preset",
                yaml_path=f.name,
            )

            config = preset.load_config()
            assert config["config_version"] == CANONICAL_CONFIG_VERSION
            assert config["metadata"]["experiment_name"] == "test_experiment"
            assert config["data"]["batch_size"] == 32

        Path(f.name).unlink()

    def test_to_settings(self) -> None:
        """Should convert preset to TrainingSettings - skipped due to complex validation."""
        pytest.skip(
            "Complex validation requires full config; tested via CLI integration tests"
        )

    def test_to_settings_with_overrides(self) -> None:
        """Should apply field overrides to preset settings - skipped due to complex validation."""
        pytest.skip(
            "Complex validation requires full config; tested via CLI integration tests"
        )

    def test_metadata_export(self) -> None:
        """Should export preset metadata as dict."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("config_version: '5.0'\n")
            f.write("experiment:\n")
            f.write("  name: test_experiment\n")
            f.write("data:\n")
            f.write("  batch_size: 32\n")
            f.write("optimization:\n")
            f.write("  learning_rate: 0.001\n")
            f.write("  epochs: 100\n")
            f.write("model:\n")
            f.write("  model_type: standard_unet\n")
            f.write("training:\n")
            f.write("  training_mode: gan\n")
            f.flush()

            preset = ConfigPreset(
                name="test_preset",
                category="gan",
                description="A test preset",
                yaml_path=f.name,
                tags=["baseline"],
            )

            metadata = preset.to_dict()
            assert metadata["name"] == "test_preset"
            assert metadata["category"] == "gan"
            assert metadata["tags"] == ["baseline"]
            assert metadata["description"] == "A test preset"

        Path(f.name).unlink()

    def test_repr(self) -> None:
        """Should have readable string representation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("config_version: '5.0'\n")
            f.write("experiment:\n")
            f.write("  name: test_experiment\n")
            f.write("data:\n")
            f.write("  batch_size: 32\n")
            f.write("optimization:\n")
            f.write("  learning_rate: 0.001\n")
            f.write("  epochs: 100\n")
            f.write("model:\n")
            f.write("  model_type: standard_unet\n")
            f.write("training:\n")
            f.write("  training_mode: gan\n")
            f.flush()

            preset = ConfigPreset(
                name="test_preset",
                category="gan",
                description="Test description",
                yaml_path=f.name,
                tags=["baseline"],
            )

            repr_str = repr(preset)
            assert "test_preset" in repr_str
            assert "gan" in repr_str

        Path(f.name).unlink()


class TestDiscovery:
    """Test preset discovery and auto-registration."""

    def test_infer_tags_from_filename(self) -> None:
        """Should infer tags from experiment filename patterns."""
        from mriforge.config.presets.discovery import _infer_tags

        # Test various filename patterns with category
        # _infer_tags always includes the category plus other inferred tags
        patterns = {
            ("experiment_01_baseline_gan.yaml", "gan"): (
                "baseline",
                "gan",
            ),  # Both included
            ("experiment_02_ablation_loss.yaml", "gan"): ("ablation", "gan"),
            ("experiment_03_robustness.yaml", "gan"): ("robustness", "gan"),
            ("experiment_04_hyperparameter_optimization.yaml", "gan"): ("hpo", "gan"),
        }

        for (filename, category), expected_tags in patterns.items():
            tags = _infer_tags(filename, category)
            for expected_tag in expected_tags:
                assert (
                    expected_tag in tags
                ), f"Expected '{expected_tag}' in tags {tags} for {filename}"

    def test_get_baseline_presets(self) -> None:
        """Should return one baseline preset per category."""
        registry = PresetRegistry()
        registry._presets.clear()
        registry._categories.clear()
        registry._tags.clear()

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yaml"
            yaml_file.write_text(
                "config_version: '5.0'\n"
                "experiment:\n"
                "  name: test\n"
                "data:\n"
                "  batch_size: 32\n"
                "optimization:\n"
                "  learning_rate: 0.001\n"
                "  epochs: 100\n"
                "model:\n"
                "  model_type: standard_unet\n"
                "training:\n"
                "  training_mode: gan\n"
            )

            # Register baseline presets for multiple categories
            for category in ["gan", "diffusion", "reconstruction"]:
                preset = ConfigPreset(
                    name=f"baseline_{category}",
                    category=category,
                    description=f"Baseline for {category}",
                    yaml_path=str(yaml_file),
                    tags=["baseline"],
                )
                registry.register(preset)

            baselines = get_baseline_presets()
            assert len(baselines) == 3
            assert all("baseline" in p.tags for p in baselines)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
