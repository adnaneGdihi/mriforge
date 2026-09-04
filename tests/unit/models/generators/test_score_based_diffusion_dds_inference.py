r"""Tests for score-generator reconstruction inference via the SamplerRegistry (XC.1).

``inference_sampler: dds`` was parsed, validated, and audit-guarded, but
``score_based_diffusion`` only had an UNCONDITIONAL ``generate(z)`` path
(p_sample_loop) — so DDS (a CONDITIONAL posterior reconstruction sampler) was
never consumed end-to-end. This adds a reconstruction ``sample(measurement,
mask)`` that routes through ``get_sampler(self._sampler_name, model=self, ...)``
— the same World-A convention the cold-diffusion generator uses, which the
``DiffusionTrainingStrategy`` multistep-validation path already invokes.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.models.generators.score_based_diffusion_generator import (
    ScoreBasedDiffusionGenerator,
)


def _tiny_score_gen(inference_sampler: str = "dds") -> ScoreBasedDiffusionGenerator:
    torch.manual_seed(0)
    return ScoreBasedDiffusionGenerator(
        denoising_model=nn.Conv2d(1, 1, 3, padding=1),
        timesteps=10,
        in_channels=1,
        inference_sampler=inference_sampler,
        sampling_steps=4,
    )


def _measurement(h: int = 16, w: int = 16):
    img = torch.randn(1, 1, h, w)
    mask = torch.zeros(1, 1, h, w)
    mask[..., :, ::2] = 1.0
    k = mask.to(torch.complex64) * fft2c(torch.complex(img, torch.zeros_like(img)))
    return k, mask


def test_sampler_name_read_from_inference_sampler():
    assert _tiny_score_gen("dds")._sampler_name == "dds"


def test_sampler_name_defaults_to_ddim():
    gen = ScoreBasedDiffusionGenerator(
        denoising_model=nn.Conv2d(1, 1, 3, padding=1), timesteps=10, in_channels=1
    )
    assert gen._sampler_name == "ddim"


def test_sample_reconstruction_routes_and_returns_finite():
    gen = _tiny_score_gen("dds")
    measurement, mask = _measurement()
    out = gen.sample(measurement=measurement, mask=mask)
    assert out.shape[-2:] == (16, 16)
    assert torch.isfinite(out).all()


def test_get_sampler_dds_with_score_gen_is_call_compatible():
    from spectramr.models.diffusion.samplers import get_sampler
    from spectramr.models.diffusion.samplers.posterior_samplers import DDSReconSampler

    gen = _tiny_score_gen("dds")
    sampler = get_sampler("dds", model=gen, num_timesteps=gen.timesteps, sampling_steps=4)
    assert isinstance(sampler, DDSReconSampler)


def test_dynamic_dps_guidance_knobs_thread_to_sampler():
    """model_kwargs guidance_mode/sigma must reach the resolved DynamicDPSSampler."""
    from spectramr.models.diffusion.samplers.posterior_samplers import DynamicDPSSampler

    gen = ScoreBasedDiffusionGenerator(
        denoising_model=nn.Conv2d(1, 1, 3, padding=1), timesteps=10, in_channels=1,
        inference_sampler="dynamic_dps", guidance_mode="ncchi", sigma=0.02, n_coils=4,
    )
    assert gen._sampler_kwargs["guidance_mode"] == "ncchi"
    measurement, mask = _measurement()
    out = gen.sample(measurement=measurement, mask=mask)
    assert torch.isfinite(out).all()
    from spectramr.models.diffusion.samplers import get_sampler

    sampler = get_sampler("dynamic_dps", model=gen, num_timesteps=gen.timesteps, **gen._sampler_kwargs)
    assert isinstance(sampler, DynamicDPSSampler) and sampler.guidance_mode == "ncchi"


def test_score_prior_opts_out_of_input_invariant_probe():
    """The unconditional score prior declares the input_invariant probe skip
    (measurement-dependence is in the sampler/strategy, not the bare forward)."""
    assert "input_invariant" in ScoreBasedDiffusionGenerator.synthetic_forward_probe_skip


def test_diffusion_strategy_dispatches_dds_to_sample():
    """The score-validation branch routes posterior samplers (dds) to
    gen.sample(measurement, mask) — not the unconditional generate()."""
    import inspect

    from spectramr.infrastructure.training.strategies import diffusion as diff

    assert "dds" in diff._POSTERIOR_RECON_SAMPLERS
    src = inspect.getsource(diff.DiffusionTrainingStrategy)
    assert "_POSTERIOR_RECON_SAMPLERS" in src
    assert "gen.sample(" in src and "kspace_measured" in src
