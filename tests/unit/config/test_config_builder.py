"""Unit tests for spectramr.config.config_builder.ConfigBuilder.

PART A — Task III.3.

Covers:
  1. canary   — fluent API produces a valid dict.
  2. parametrized — preset templates each produce well-formed configs.
  3. edge     — single-field sets, boundary learning_rate, batch_size=1.
  4. expected-failure — missing required fields → ValueError; unknown
     task type; to_yaml round-trip on missing required field.

All tests @pytest.mark.unit.
"""

from __future__ import annotations

import pytest

from spectramr.config.config_builder import (
    ConfigBuilder,
    ConfigTemplates,
    build_config,
    get_template,
)


# ---------------------------------------------------------------------------
# 1. CANARY
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConfigBuilderCanary:
    def test_fluent_chain_builds_valid_dict(self) -> None:
        cfg = (
            ConfigBuilder()
            .with_architecture("standard_unet")
            .with_batch_size(32)
            .with_learning_rate(1e-3)
            .build()
        )
        assert cfg["model"]["model_type"] == "standard_unet"
        assert cfg["data"]["batch_size"] == 32
        assert cfg["optimization"]["learning_rate"] == pytest.approx(1e-3)

    def test_build_returns_dict_with_required_top_level_keys(self) -> None:
        cfg = (
            ConfigBuilder()
            .with_architecture("unet")
            .with_batch_size(4)
            .with_learning_rate(2e-4)
            .build()
        )
        for key in ("model", "data", "optimization", "losses"):
            assert key in cfg, f"Key '{key}' missing from built config"

    def test_validate_returns_true_on_valid_config(self) -> None:
        builder = (
            ConfigBuilder()
            .with_architecture("unet")
            .with_batch_size(4)
            .with_learning_rate(1e-3)
        )
        assert builder.validate() is True


# ---------------------------------------------------------------------------
# 2. PARAMETRIZED — preset templates
# ---------------------------------------------------------------------------


TEMPLATE_NAMES = [
    "reconstruction_baseline",
    "reconstruction_advanced",
    "synthesis_gan",
    "super_resolution",
    "denoising",
]


@pytest.mark.unit
@pytest.mark.parametrize("template_name", TEMPLATE_NAMES)
def test_preset_template_builds(template_name: str) -> None:
    builder = get_template(template_name)
    cfg = builder.build()
    assert "model" in cfg
    assert "data" in cfg
    assert "optimization" in cfg
    assert cfg["model"].get("model_type") is not None
    assert cfg["data"].get("batch_size", 0) > 0
    assert cfg["optimization"].get("learning_rate", 0) > 0


@pytest.mark.unit
class TestConfigTemplateClasses:
    def test_reconstruction_baseline(self) -> None:
        cfg = ConfigTemplates.reconstruction_baseline().build()
        assert cfg["model"]["model_type"] == "standard_unet"
        assert cfg["data"]["batch_size"] == 32

    def test_reconstruction_advanced(self) -> None:
        cfg = ConfigTemplates.reconstruction_advanced().build()
        assert cfg["model"]["model_type"] == "swin_unet"

    def test_synthesis_gan_has_gan_losses(self) -> None:
        cfg = ConfigTemplates.synthesis_gan().build()
        assert cfg["losses"].get("use_gan") is True

    def test_super_resolution_has_perceptual(self) -> None:
        cfg = ConfigTemplates.super_resolution().build()
        assert cfg["losses"].get("use_perceptual") is True

    def test_denoising_baseline(self) -> None:
        cfg = ConfigTemplates.denoising().build()
        assert cfg["model"]["model_type"] == "standard_unet"


# ---------------------------------------------------------------------------
# 3. EDGE
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConfigBuilderEdgeCases:
    def test_batch_size_one_builds(self) -> None:
        cfg = (
            ConfigBuilder()
            .with_architecture("unet")
            .with_batch_size(1)
            .with_learning_rate(1e-3)
            .build()
        )
        assert cfg["data"]["batch_size"] == 1

    def test_very_small_learning_rate(self) -> None:
        cfg = (
            ConfigBuilder()
            .with_architecture("unet")
            .with_batch_size(4)
            .with_learning_rate(1e-7)
            .build()
        )
        assert cfg["optimization"]["learning_rate"] == pytest.approx(1e-7)

    def test_epochs_set_under_optimization(self) -> None:
        cfg = (
            ConfigBuilder()
            .with_architecture("unet")
            .with_batch_size(4)
            .with_learning_rate(1e-3)
            .with_epochs(200)
            .build()
        )
        assert cfg["optimization"]["num_epochs"] == 200

    def test_optimizer_set(self) -> None:
        cfg = (
            ConfigBuilder()
            .with_architecture("unet")
            .with_batch_size(4)
            .with_learning_rate(1e-3)
            .with_optimizer("adamw")
            .build()
        )
        assert cfg["optimization"]["optimizer"] == "adamw"

    def test_enable_metrics_sets_flag(self) -> None:
        builder = (
            ConfigBuilder()
            .with_architecture("unet")
            .with_batch_size(4)
            .with_learning_rate(1e-3)
            .enable_metrics()
        )
        cfg = builder.build()
        assert cfg["training"]["track_metrics"] is True

    def test_enable_mixed_precision_sets_flag(self) -> None:
        builder = (
            ConfigBuilder()
            .with_architecture("unet")
            .with_batch_size(4)
            .with_learning_rate(1e-3)
            .enable_mixed_precision()
        )
        cfg = builder.build()
        assert cfg["optimization"]["use_amp"] is True

    def test_use_gan_losses_sets_flag(self) -> None:
        builder = (
            ConfigBuilder()
            .with_architecture("unet")
            .with_batch_size(4)
            .with_learning_rate(1e-3)
            .use_gan_losses()
        )
        cfg = builder.build()
        assert cfg["losses"]["use_gan"] is True

    def test_to_yaml_returns_string(self) -> None:
        yaml_str = (
            ConfigBuilder()
            .with_architecture("unet")
            .with_batch_size(4)
            .with_learning_rate(1e-3)
            .to_yaml()
        )
        assert isinstance(yaml_str, str)
        assert "model_type" in yaml_str

    def test_to_yaml_saves_file(self, tmp_path) -> None:
        yaml_path = tmp_path / "config.yaml"
        (
            ConfigBuilder()
            .with_architecture("unet")
            .with_batch_size(4)
            .with_learning_rate(1e-3)
            .to_yaml(str(yaml_path))
        )
        assert yaml_path.exists()

    def test_save_yaml_extension(self, tmp_path) -> None:
        out = tmp_path / "config.yaml"
        result = (
            ConfigBuilder()
            .with_architecture("unet")
            .with_batch_size(4)
            .with_learning_rate(1e-3)
            .save(str(out))
        )
        assert result == out

    def test_summary_returns_string(self) -> None:
        summary = (
            ConfigBuilder()
            .with_architecture("unet")
            .with_batch_size(4)
            .with_learning_rate(1e-3)
            .summary()
        )
        assert isinstance(summary, str)
        assert "unet" in summary


# ---------------------------------------------------------------------------
# 4. EXPECTED-FAILURE
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConfigBuilderExpectedFailures:
    def test_build_without_model_type_raises(self) -> None:
        """CLAUDE.md pitfall #1 analog: missing required field → ValueError."""
        with pytest.raises(ValueError, match="Model type"):
            ConfigBuilder().with_batch_size(4).with_learning_rate(1e-3).build()

    def test_build_without_batch_size_raises(self) -> None:
        with pytest.raises(ValueError, match="Batch size"):
            (
                ConfigBuilder()
                .with_architecture("unet")
                .with_learning_rate(1e-3)
                .build()
            )

    def test_build_without_learning_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="Learning rate"):
            (
                ConfigBuilder()
                .with_architecture("unet")
                .with_batch_size(4)
                .build()
            )

    def test_unknown_task_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown task"):
            ConfigBuilder().for_task("time_travel")

    def test_unknown_template_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown template"):
            get_template("nonexistent_template_xyz")

    def test_build_config_convenience_unknown_task_raises(self) -> None:
        with pytest.raises(ValueError):
            build_config("not_a_task").build()

    def test_save_unsupported_extension_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            (
                ConfigBuilder()
                .with_architecture("unet")
                .with_batch_size(4)
                .with_learning_rate(1e-3)
                .save(str(tmp_path / "config.toml"))
            )


class TestPrepareForYamlPydanticV2:
    """Regression WS1-core-03: ``_prepare_for_yaml`` must serialize a nested
    Pydantic model via v2 ``model_dump()``, not the deprecated v1 ``.dict()``.

    The v1 ``.dict()`` shim emits a ``DeprecationWarning`` from
    ``spectramr.config.config_builder``, which the repo's
    ``error::DeprecationWarning:spectramr.*`` filter promotes to an error — so if the
    bug regressed, this test would *error* rather than merely assert-fail.
    """

    def test_nested_pydantic_model_serialized_via_model_dump(self) -> None:
        from pydantic import BaseModel

        class _Tiny(BaseModel):
            a: int = 1
            b: str = "x"

        builder = ConfigBuilder()
        out = builder._prepare_for_yaml({"block": {"m": _Tiny()}})
        assert out["block"]["m"] == {"a": 1, "b": "x"}
