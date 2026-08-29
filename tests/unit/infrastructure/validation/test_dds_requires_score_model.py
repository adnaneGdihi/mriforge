"""Unit tests for ``check_dds_requires_score_model`` (SOTA plan step 2).

DDS (Decomposed Diffusion Sampling) is a posterior sampler for a *score /
eps-prediction* diffusion model: it needs an eps predictor and a variance
schedule (``alphas_cumprod``). Selecting ``inference_sampler: dds`` on a
non-score paradigm (cold diffusion has no eps+schedule; reconstruction/GAN are
not diffusion at all) would fail loud at runtime in ``DDSReconSampler`` — this
check rejects it at config time so the dead-knob never silently degrades
(pitfalls #15/#16).
"""

from __future__ import annotations

from mriforge.infrastructure.validation.config_health_checker import ConfigHealthChecker


class _Diffusion:
    def __init__(self, inference_sampler="ddim"):
        self.inference_sampler = inference_sampler


class _Training:
    def __init__(self, training_mode="diffusion", inference_sampler="ddim"):
        self.training_mode = training_mode
        self.diffusion = _Diffusion(inference_sampler=inference_sampler)


class _Model:
    def __init__(self, model_type="score_based_diffusion"):
        self.model_type = model_type


class _Config:
    def __init__(self, training_mode="diffusion", inference_sampler="ddim",
                 model_type="score_based_diffusion"):
        self.training = _Training(training_mode=training_mode, inference_sampler=inference_sampler)
        self.model = _Model(model_type=model_type)


def _check(**kwargs):
    return ConfigHealthChecker().check_dds_requires_score_model(_Config(**kwargs))


def test_not_applicable_when_sampler_is_not_dds():
    r = _check(inference_sampler="ddim")
    assert r.passed and r.severity == "info"


def test_dds_on_score_diffusion_passes():
    r = _check(inference_sampler="dds", training_mode="diffusion",
               model_type="score_based_diffusion")
    assert r.passed


def test_dds_on_cold_diffusion_errors():
    """Cold diffusion has no eps+schedule contract → DDS would fail at runtime."""
    r = _check(inference_sampler="dds", training_mode="kspace_cold_diffusion",
               model_type="kspace_cold_diffusion")
    assert not r.passed and r.severity == "error"


def test_dds_on_non_diffusion_errors():
    r = _check(inference_sampler="dds", training_mode="reconstruction",
               model_type="enhanced_unet")
    assert not r.passed and r.severity == "error"


def test_dynamic_dps_on_score_diffusion_passes():
    """dynamic_dps shares the score-model guard with dds (Phase C)."""
    r = _check(inference_sampler="dynamic_dps", training_mode="diffusion",
               model_type="score_based_diffusion")
    assert r.passed


def test_dynamic_dps_on_cold_diffusion_errors():
    r = _check(inference_sampler="dynamic_dps", training_mode="kspace_cold_diffusion",
               model_type="kspace_cold_diffusion")
    assert not r.passed and r.severity == "error"


def test_dds_check_is_wired_into_run_all_checks():
    """The check must be callable from the orchestrator, not an orphan method."""
    import inspect

    src = inspect.getsource(ConfigHealthChecker.run_all_checks)
    assert "check_dds_requires_score_model" in src
