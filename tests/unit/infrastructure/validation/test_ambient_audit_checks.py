"""Unit tests for the Ambient Diffusion audit check (Phase B step 5)."""

from __future__ import annotations

from mriforge.infrastructure.validation.config_health_checker import ConfigHealthChecker


class _Model:
    def __init__(self, model_type="score_based_diffusion"):
        self.model_type = model_type


class _Data:
    def __init__(self, dataset_type="kspace"):
        self.dataset_type = dataset_type


class _Training:
    def __init__(self, ambient_block=True, input_domain="kspace", training_mode="diffusion"):
        self.training_mode = training_mode
        self.input_domain = input_domain
        self.ambient = object() if ambient_block else None
        self.strategy_class = (
            "mriforge.infrastructure.training.strategies."
            "ambient_diffusion_strategy.AmbientDiffusionStrategy"
        )


class _Config:
    def __init__(self, ambient_block=True, model_type="score_based_diffusion",
                 input_domain="kspace", dataset_type="kspace"):
        self.training = _Training(ambient_block=ambient_block, input_domain=input_domain)
        self.model = _Model(model_type=model_type)
        self.data = _Data(dataset_type=dataset_type)


def _c():
    return ConfigHealthChecker()


def test_not_applicable_when_not_ambient():
    cfg = _Config(ambient_block=False)
    cfg.training.strategy_class = "mriforge.x.Y"
    cfg.training.training_mode = "reconstruction"
    r = _c().check_ambient_requires_score_model(cfg)
    assert r.passed and r.severity == "info"


def test_passes_score_model_on_kspace():
    r = _c().check_ambient_requires_score_model(_Config())
    assert r.passed


def test_errors_on_non_score_model():
    r = _c().check_ambient_requires_score_model(_Config(model_type="enhanced_unet"))
    assert not r.passed and r.severity == "error"


def test_errors_on_image_only_input():
    r = _c().check_ambient_requires_score_model(
        _Config(input_domain="image", dataset_type="image")
    )
    assert not r.passed and r.severity == "error"


def test_ambient_check_wired_into_run_all_checks():
    import inspect

    src = inspect.getsource(ConfigHealthChecker.run_all_checks)
    assert "check_ambient_requires_score_model" in src
