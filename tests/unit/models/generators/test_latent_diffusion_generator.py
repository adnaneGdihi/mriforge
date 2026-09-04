"""Paired tests for LatentDiffusionGenerator's ``vae_backbone`` selection.

Covers the transfer-learning seam added alongside pretrained_vae_adapter.py: the
generator can build either the from-scratch custom ``AutoencoderKL`` (default) or
a frozen pretrained Stable-Diffusion VAE. ``diffusers`` is NOT required — the SD
branch is exercised only up to its clear dependency error (via monkeypatch) and
its 2D-only guard, both of which fire before any network download.
"""

from __future__ import annotations

import sys

import pytest
import torch

from spectramr.models.generators.latent_diffusion_generator import (
    AutoencoderKL,
    LatentDiffusionGenerator,
    LatentDiffusionGeneratorConfig,
)


def _config(spatial_dims: int = 2) -> LatentDiffusionGeneratorConfig:
    return LatentDiffusionGeneratorConfig(
        in_channels=1,
        out_channels=1,
        latent_channels=4,
        base_channels=16,
        timesteps=5,
        spatial_dims=spatial_dims,
        device="cpu",
    )


def test_default_backbone_builds_custom_autoencoder():
    gen = LatentDiffusionGenerator(_config(), num_layers=3)
    assert gen._vae_backbone == "custom"
    assert isinstance(gen.autoencoder, AutoencoderKL)


def test_unknown_vae_backbone_raises_no_silent_fallback():
    with pytest.raises(ValueError, match="vae_backbone"):
        LatentDiffusionGenerator(_config(), num_layers=3, vae_backbone="monai")


def test_sd_backbone_requires_diffusers(monkeypatch):
    # Force `import diffusers` to fail regardless of install state.
    monkeypatch.setitem(sys.modules, "diffusers", None)
    with pytest.raises(ImportError, match="diffusers"):
        LatentDiffusionGenerator(_config(), num_layers=3, vae_backbone="sd")


def test_sd_backbone_is_2d_only():
    with pytest.raises(ValueError, match=r"2D-only|spatial_dims"):
        LatentDiffusionGenerator(_config(spatial_dims=3), vae_backbone="sd")


# ── SR3 conditioning de-facade (2026-07 ldm_two_stage triage) ──────────────────
# conditional_translation was swallowed by **kwargs → the SR3 concat was never
# built and the model ran unconditionally. These tests are the mechanism-fires
# guard that would have caught the facade.


def _conditional_kwargs(**over):
    base = {
        "in_channels": 1,
        "out_channels": 1,
        "latent_channels": 4,
        "base_channels": 16,
        "timesteps": 5,
        "spatial_dims": 2,
        "conditional_translation": True,
        "conditioning_key": "concat",
        "num_layers": 3,
    }
    base.update(over)
    return base


def _first_conv_in_channels(gen) -> int:
    import torch.nn as nn

    for m in gen.denoise_net.modules():
        if isinstance(m, nn.Conv2d):
            return m.in_channels
    raise AssertionError("no Conv2d in denoise_net")


def test_conditional_translation_threaded_and_unet_doubled():
    """The knob reaches config AND the UNet first conv doubles for the concat."""
    gen = LatentDiffusionGenerator(**_conditional_kwargs())
    assert gen.config.conditional_translation is True
    assert _first_conv_in_channels(gen) == 2 * gen.latent_channels


def test_unconditional_lodm_unet_not_doubled():
    """An unconditional LDM must keep single-width UNet input (no regression)."""
    gen = LatentDiffusionGenerator(_config(), num_layers=3)
    assert gen.config.conditional_translation is False
    assert _first_conv_in_channels(gen) == gen.latent_channels


def test_forward_conditional_requires_condition_image():
    gen = LatentDiffusionGenerator(**_conditional_kwargs())
    x = torch.randn(2, 4, 8, 8)
    t = torch.randint(0, 5, (2,))
    with pytest.raises(ValueError, match="condition_image is None"):
        gen.forward(x, t, condition_image=None)


def test_forward_conditional_runs_with_condition_image():
    gen = LatentDiffusionGenerator(**_conditional_kwargs())
    x = torch.randn(2, 4, 8, 8)
    t = torch.randint(0, 5, (2,))
    out = gen.forward(x, t, condition_image=torch.randn(2, 1, 64, 64))
    assert tuple(out.shape) == (2, 4, 8, 8)


def test_unknown_conditioning_key_raises():
    with pytest.raises(ValueError, match="conditioning_key"):
        LatentDiffusionGenerator(**_conditional_kwargs(conditioning_key="crossattn"))


def test_contrast_embed_dim_must_match_time_emb_dim():
    with pytest.raises(ValueError, match="contrast_embed_dim"):
        LatentDiffusionGenerator(
            **_conditional_kwargs(use_contrast_guidance=True, contrast_embed_dim=64)
        )


# ── contrast guidance actually reaches the output ────────────────────────────
# The tests above prove the SR3 *concat* seam is built. Nothing proved the OTHER
# conditioning channel — per-contrast guidance — changes anything, and it is the
# harder one to check because of the trap the next test pins.


def _run(gen, contrast_idx, cond_image, *, seed=1234):
    """Forward with everything held fixed except the conditioning under test."""
    torch.manual_seed(seed)  # re-seed per call: forward() may draw internally
    with torch.no_grad():
        return gen.forward(
            torch.zeros(2, 4, 8, 8) + 0.5,
            torch.zeros(2, dtype=torch.long) + 3,
            context={"contrast_idx": contrast_idx} if contrast_idx is not None else None,
            condition_image=cond_image,
        )


def _contrast_gen():
    gen = LatentDiffusionGenerator(
        **_conditional_kwargs(
            use_contrast_guidance=True, contrast_embed_dim=128, num_contrasts=4
        )
    )
    # eval(): forward() applies 10% classifier-free-guidance dropout to the
    # condition latent while training, which would make any "outputs differ"
    # assertion below a coin flip.
    return gen.eval()


def test_contrast_guidance_is_inert_at_init_because_adagn_is_zero_init():
    """DOCUMENTS A TRAP: at init, contrast_idx provably cannot move the output.

    ``AdaptiveGroupNorm`` zero-initializes its projection
    (``normalization.py`` — ``proj.weight.data.zero_()`` / ``bias.data.zero_()``),
    so ``norm(x) * (1 + scale) + shift`` collapses to ``norm(x)`` for ANY
    embedding until those weights train. The contrast embedding travels the AdaGN
    path (``emb_decoder = emb + contrast_context``), so on a freshly built model
    switching contrasts changes the output by exactly 0.

    That is correct-by-design, NOT a facade — but it means the obvious probe
    ("build the model, flip contrast_idx, see if anything moves") reports a dead
    mechanism for a live one. This test pins the trap so the next person reads it
    before filing the bug; ``test_contrast_guidance_changes_output`` below is the
    check that actually has power.
    """
    from spectramr.models.blocks.normalization import AdaptiveGroupNorm

    gen = _contrast_gen()
    agn = [m for m in gen.modules() if isinstance(m, AdaptiveGroupNorm)]
    assert agn, "expected AdaptiveGroupNorm layers in the latent UNet"
    assert all(m.proj.weight.abs().max() == 0 for m in agn)

    cond = torch.randn(2, 1, 64, 64)
    a = _run(gen, torch.zeros(2, dtype=torch.long), cond)
    b = _run(gen, torch.full((2,), 3, dtype=torch.long), cond)
    assert torch.equal(a, b)


def test_contrast_guidance_changes_output():
    """Mechanism-fires guard: contrast_idx must move the output once AdaGN trains.

    Perturbing the AdaGN projections off their zero init is what a few optimizer
    steps do; it is the minimum state in which the contrast path can express
    itself at all. If this ever fails, the embedding is computed and discarded —
    a real pitfall #16 facade — and every contrast-conditioned arm is silently
    contrast-agnostic while still paying for the embedding.
    """
    import torch.nn as nn

    from spectramr.models.blocks.normalization import AdaptiveGroupNorm

    gen = _contrast_gen()
    torch.manual_seed(7)
    for m in gen.modules():
        if isinstance(m, AdaptiveGroupNorm):
            nn.init.normal_(m.proj.weight, std=0.02)

    cond = torch.randn(2, 1, 64, 64)
    a = _run(gen, torch.zeros(2, dtype=torch.long), cond)
    b = _run(gen, torch.full((2,), 3, dtype=torch.long), cond)
    assert not torch.allclose(a, b), "contrast_idx did not reach the output"


def test_contrast_guidance_without_contrast_idx_raises():
    """Declared guidance + absent contrast_idx must RAISE, not run unconditioned.

    ``_process_slices`` gates the contrast path on ``contrast_idx is not None``,
    so before this guard a model declaring ``use_contrast_guidance`` and never
    receiving the index trained contrast-AGNOSTICALLY while advertising
    per-contrast conditioning — carrying an ``nn.Embedding`` whose gradient is
    always zero (pitfall #16). Neither contrast audit check covered this
    direction: ``multi_contrast_model_support`` guards data->model, and
    ``contrast_conditioning_strategy_threaded`` keys off a different kwarg
    (``use_contrast_conditioning``).
    """
    gen = _contrast_gen()
    with pytest.raises(ValueError, match="contrast_idx is None"):
        gen.forward(
            torch.randn(2, 4, 8, 8),
            torch.randint(0, 5, (2,)),
            context=None,
            condition_image=torch.randn(2, 1, 64, 64),
        )


def test_contrast_guidance_guard_is_forward_only_sample_stays_permissive():
    """sample() must NOT raise on contrast_idx=None — CFG needs that path.

    Unconditional sampling is a legitimate classifier-free-guidance use, so the
    guard belongs on the training forward only. Pinning this keeps a later
    "tighten the guard everywhere" change from silently removing CFG.
    """
    gen = _contrast_gen()
    out = gen.sample(
        (1, 1, 64, 64),
        device="cpu",
        condition_image=torch.randn(1, 1, 64, 64),
        contrast_idx=None,
        num_inference_steps=1,
    )
    assert tuple(out.shape[-2:]) == (64, 64)


def test_custom_backbone_is_2d_only():
    """spatial_dims=3 must RAISE at construction, not die at the first encode.

    AutoencoderKL branches Conv2d/Conv3d for quant_conv but its encoder/decoder
    submodules are hardcoded nn.Conv2d, so a 3-D instance used to CONSTRUCT and
    then fail mid-forward with a bare "Expected 3D (unbatched) or 4D (batched)
    input to conv2d" — a shape error with nothing pointing at the config key
    that caused it. The `sd` backbone always had this guard; the custom one did
    not.
    """
    with pytest.raises(ValueError, match=r"2D-only|spatial_dims"):
        LatentDiffusionGenerator(_config(spatial_dims=3), num_layers=3)


def test_registry_declares_2d_only_for_both_names():
    """The advertised capability must match what the model can do (#8).

    ``spatial_dims=(2, 3)`` is how a 3-D arm got written against this model: the
    declaration is what the audit spec card and check_workflow_spatial_rank read.
    ``stubs.py`` aliases ``latent_diffusion`` onto the same registry entry, so
    both names are asserted here — a future split of the two registrations must
    keep them in agreement.
    """
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY

    populate_model_registry()
    for name in ("latent_gan_generator", "latent_diffusion"):
        caps = MODEL_REGISTRY[name]["capabilities"]
        assert caps.spatial_dims == (2,), f"{name} advertises {caps.spatial_dims}"


def test_sr3_condition_image_changes_output_at_init():
    """The concat path does NOT go through AdaGN, so it is live from step 0.

    This is the control for the two tests above: it shows the zero-init
    explanation is specific to the embedding path, not a property of the whole
    forward, so "no output change" on the contrast path cannot be waved away as
    the model being untrained in general.
    """
    gen = _contrast_gen()
    ci = torch.zeros(2, dtype=torch.long)
    a = _run(gen, ci, torch.randn(2, 1, 64, 64))
    b = _run(gen, ci, torch.zeros(2, 1, 64, 64))
    assert not torch.allclose(a, b), "condition_image did not reach the output"


def test_post_quant_conv_applied_once_on_round_trip():
    """encode_to_latent must NOT apply post_quant_conv (decode already does)."""
    gen = LatentDiffusionGenerator(_config(), num_layers=3)
    calls = {"n": 0}
    gen.autoencoder.post_quant_conv.register_forward_hook(
        lambda m, i, o: calls.__setitem__("n", calls["n"] + 1)
    )
    z = gen.encode_to_latent(torch.randn(1, 1, 64, 64))
    _ = gen.decode_from_latent(z)
    assert calls["n"] == 1


def test_sample_crops_to_requested_odd_size():
    """The latent floor (//ds) can undershoot; sample() aligns to the request."""
    gen = LatentDiffusionGenerator(**_conditional_kwargs())
    gen.eval()
    out = gen.sample(
        (1, 1, 72, 72),
        device="cpu",
        condition_image=torch.randn(1, 1, 72, 72),
        contrast_idx=torch.tensor([0]),
    )
    assert tuple(out.shape[-2:]) == (72, 72)


# ── latent scaling knob (2026-07): set_latent_statistics had no caller ─────────


def test_latent_scaling_factor_sets_std():
    gen = LatentDiffusionGenerator(
        in_channels=1,
        out_channels=1,
        latent_channels=4,
        base_channels=16,
        timesteps=5,
        spatial_dims=2,
        latent_scaling_factor=2.45,
        num_layers=3,
    )
    assert float(gen.latent_std.flatten()[0]) == pytest.approx(1.0 / 2.45, abs=1e-5)


def test_latent_scaling_absent_is_identity_no_regression():
    gen = LatentDiffusionGenerator(_config(), num_layers=3)
    assert float(gen.latent_std.flatten()[0]) == pytest.approx(1.0)


def test_latent_scaling_factor_must_be_positive():
    with pytest.raises(ValueError, match="latent_scaling_factor"):
        LatentDiffusionGenerator(_config(), num_layers=3, latent_scaling_factor=-1.0)


def test_latent_scaling_factor_rejected_on_sd_backbone():
    pytest.importorskip("diffusers")
    with pytest.raises(ValueError, match="latent_scaling_factor"):
        LatentDiffusionGenerator(
            _config(), num_layers=3, vae_backbone="sd", latent_scaling_factor=2.0
        )


# ---------------------------------------------------------------------------
# Reverse-schedule SSOT (2026-07-11 stage-2 LDM triage).
#
# The strategy q_samples with ``training.diffusion.noise_schedule`` while the
# generator's internal reverse ``Diffusion`` was built from the dataclass
# default ``beta_schedule="linear"``, which no YAML ever wired. Training on a
# cosine forward trajectory and sampling with linear posterior coefficients
# desyncs the reverse process → the decoded image collapses to ~black
# (val_psnr≈6 dB on both stage-2 arms while train_psnr≈32).
# ---------------------------------------------------------------------------


def test_set_diffusion_schedule_rebinds_reverse_process():
    gen = LatentDiffusionGenerator(_config(), num_layers=3)
    assert gen.beta_schedule == "linear"

    gen.set_diffusion_schedule(timesteps=7, beta_schedule="cosine", device="cpu")

    assert gen.beta_schedule == "cosine"
    assert gen.timesteps == 7
    assert gen.diffusion.timesteps == 7
    # The betas must come from the cosine schedule, not the linear one.
    linear = LatentDiffusionGenerator(_config(), num_layers=3).diffusion.betas
    assert not torch.allclose(gen.diffusion.betas[: min(5, 7)], linear[:5])


def test_set_diffusion_schedule_rejects_unknown_schedule():
    gen = LatentDiffusionGenerator(_config(), num_layers=3)
    with pytest.raises(ValueError, match="beta schedule"):
        gen.set_diffusion_schedule(timesteps=5, beta_schedule="not_a_schedule")


def test_beta_schedule_kwarg_is_honoured():
    """model_kwargs.beta_schedule must reach the reverse process (#15).

    Exercises the production path: the model builder passes model_kwargs, NOT a
    config object (an explicit ``config=`` makes the scalar args a documented
    no-op).
    """
    gen = LatentDiffusionGenerator(
        in_channels=1,
        out_channels=1,
        latent_channels=4,
        base_channels=16,
        timesteps=5,
        beta_schedule="cosine",
        device="cpu",
        num_layers=3,
    )
    assert gen.beta_schedule == "cosine"
    assert gen.diffusion.betas.shape[0] == 5


# ---------------------------------------------------------------------------
# Stage-1 VAE hand-off (2026-07-11).
#
# CheckpointDirector.save() writes the weights under the key "generator" (and
# also writes "epoch"). The loader only looked for "model_state_dict" /
# "state_dict", and its raw-dict fallback is disabled by the presence of
# "epoch" — so EVERY stage-1 checkpoint this repo produces failed to load and
# the LDM silently trained against a randomly-initialised VAE.
# ---------------------------------------------------------------------------


def _stage1_checkpoint(tmp_path, key: str = "generator"):
    """Write a checkpoint shaped exactly like CheckpointDirector.save()."""
    from spectramr.models.generators.autoencoder_pretrain_generator import (
        AutoencoderPretrainGenerator,
    )

    # Same autoencoder geometry the LDM builds from ``_config()`` so the
    # state-dict keys AND shapes line up (base_channels is the only knob the
    # LDM forwards; num_layers/downsample_factor take AutoencoderKL defaults).
    stage1 = AutoencoderPretrainGenerator(
        in_channels=1,
        out_channels=1,
        latent_channels=4,
        base_channels=16,
        spatial_dims=2,
    )
    path = tmp_path / "checkpoint_best.pt"
    torch.save(
        {
            "epoch": 3,
            "global_step": 3000,
            key: stage1.state_dict(),
            "optimizer_g": {},
            "metrics": {"val_psnr": 34.9},
        },
        path,
    )
    return path, stage1


def test_loads_checkpoint_director_generator_key(tmp_path):
    """The 'generator' key written by CheckpointDirector must be recognised."""
    path, stage1 = _stage1_checkpoint(tmp_path)

    gen = LatentDiffusionGenerator(
        _config(), num_layers=3, vae_checkpoint_path=str(path), freeze_vae=True
    )

    # Weights actually transferred (not left at random init).
    ref = stage1.autoencoder.state_dict()
    for name, param in gen.autoencoder.state_dict().items():
        if name in ref and param.is_floating_point():
            assert torch.allclose(param, ref[name]), f"{name} did not load"
            break
    else:  # pragma: no cover - defensive
        pytest.fail("no comparable autoencoder parameter found")

    # freeze_vae=True must actually freeze.
    assert all(not p.requires_grad for p in gen.autoencoder.parameters())


def test_unloadable_vae_checkpoint_raises_not_warns(tmp_path):
    """A checkpoint with no recognisable state-dict must RAISE (#9), never warn
    and silently leave the autoencoder random-initialised."""
    path = tmp_path / "checkpoint_best.pt"
    torch.save({"epoch": 1, "metrics": {}, "not_a_state_dict": 123}, path)

    with pytest.raises(RuntimeError, match="state-dict"):
        LatentDiffusionGenerator(_config(), num_layers=3, vae_checkpoint_path=str(path))


def test_vae_checkpoint_with_no_matching_keys_raises(tmp_path):
    """Architecture divergence between stage-1 and stage-2 must RAISE (#9)."""
    path = tmp_path / "checkpoint_best.pt"
    torch.save({"epoch": 1, "generator": {"totally.unrelated.weight": torch.zeros(2)}}, path)

    with pytest.raises(RuntimeError, match="no keys matching"):
        LatentDiffusionGenerator(_config(), num_layers=3, vae_checkpoint_path=str(path))


def test_partially_matching_vae_checkpoint_raises(tmp_path):
    """A checkpoint that restores only SOME autoencoder tensors must RAISE (#9).

    ``load_state_dict(strict=False)`` tolerates missing keys by leaving them at
    their random init, and the emptiness guard only catches a *total* miss. A
    stage-1 VAE whose key set merely diverges (different depth) therefore loaded
    a PARTIALLY random autoencoder and still reported PASS — the same defect the
    'generator'-key fix closes, reached by a different door.
    """
    path, stage1 = _stage1_checkpoint(tmp_path)

    # Drop one tensor: key names still line up, shapes still line up, but the
    # restore is incomplete. This is the depth-drift case (shape drift already
    # raises inside load_state_dict).
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    dropped = next(k for k in ckpt["generator"] if k.startswith("autoencoder."))
    del ckpt["generator"][dropped]
    torch.save(ckpt, path)

    with pytest.raises(RuntimeError, match="randomly initialised"):
        LatentDiffusionGenerator(_config(), num_layers=3, vae_checkpoint_path=str(path))


class TestValidationSamplerSteps:
    """``sample(num_inference_steps=...)`` — wiring for ``validation.sampler_steps``.

    ``DiffusionTrainingStrategy._generate_validation_prediction`` forwards the
    resolved step count only under a parameter name the sampler's signature
    actually exposes. Before this seam existed, ``sampler_steps: 25`` on the
    ldm_ulf_to_hf stage-2 arms was silently dropped and every validation ran the
    full 1000-step reverse loop (40x the declared cost).
    """

    def _gen(self, timesteps: int = 20) -> LatentDiffusionGenerator:
        cfg = LatentDiffusionGeneratorConfig(
            in_channels=1,
            out_channels=1,
            latent_channels=4,
            base_channels=16,
            timesteps=timesteps,
            spatial_dims=2,
            device="cpu",
        )
        return LatentDiffusionGenerator(cfg, num_layers=3)

    def _count_denoiser_calls(self, gen, monkeypatch) -> list[int]:
        seen: list[int] = []

        def _spy(x, t, mode, **kwargs):
            seen.append(int(t.flatten()[0].item()))
            return torch.zeros_like(x[:, : gen.latent_channels])

        monkeypatch.setattr(gen, "_process_slices", _spy)
        return seen

    def test_strategy_probe_finds_a_step_parameter(self):
        # Bind to the strategy's REAL probe list, not a copy: at least one of
        # those names must be on sample() or the configured sampler_steps is
        # dropped and validation silently runs the full chain.
        import inspect

        from spectramr.infrastructure.training.strategies.diffusion import (
            _SAMPLER_STEP_PARAM_NAMES,
        )

        params = inspect.signature(LatentDiffusionGenerator.sample).parameters
        assert any(name in params for name in _SAMPLER_STEP_PARAM_NAMES)

    def test_num_inference_steps_limits_denoiser_evaluations(self, monkeypatch):
        gen = self._gen(timesteps=20)
        seen = self._count_denoiser_calls(gen, monkeypatch)

        gen.sample((1, 1, 16, 16), device="cpu", num_inference_steps=5)

        assert len(seen) == 5
        # Strided schedule must run strictly high -> low and end at t=0.
        assert seen == sorted(seen, reverse=True)
        assert seen[-1] == 0

    def test_default_runs_the_full_reverse_loop(self, monkeypatch):
        gen = self._gen(timesteps=20)
        seen = self._count_denoiser_calls(gen, monkeypatch)

        gen.sample((1, 1, 16, 16), device="cpu")

        assert len(seen) == 20

    def test_step_count_above_schedule_is_clamped(self, monkeypatch):
        gen = self._gen(timesteps=20)
        seen = self._count_denoiser_calls(gen, monkeypatch)

        gen.sample((1, 1, 16, 16), device="cpu", num_inference_steps=999)

        assert len(seen) == 20

    def test_single_step_starts_at_max_noise_not_zero(self, monkeypatch):
        # torch.linspace(a, b, 1) returns [a]. Spanning low->high would put the
        # lone evaluation at t=0 — telling the denoiser that pure noise is
        # already clean. sampler_steps: 1 is legal config (schema ge=1).
        gen = self._gen(timesteps=20)
        seen = self._count_denoiser_calls(gen, monkeypatch)

        gen.sample((1, 1, 16, 16), device="cpu", num_inference_steps=1)

        assert seen == [19]

    def test_strided_schedule_is_distinct_and_spans_the_chain(self, monkeypatch):
        # Dedup must never silently shorten the chain: asking for k steps has to
        # cost exactly k denoiser evaluations, from max noise down to t=0.
        for k in (2, 3, 7, 19):
            gen = self._gen(timesteps=20)
            seen = self._count_denoiser_calls(gen, monkeypatch)
            gen.sample((1, 1, 16, 16), device="cpu", num_inference_steps=k)
            assert len(seen) == k, f"k={k} produced {len(seen)} evaluations"
            assert seen[0] == 19 and seen[-1] == 0, f"k={k} spans {seen[0]}..{seen[-1]}"


# --------------------------------------------------------------------------
# quant_conv is UNTRAINED on the from-scratch stage-1 path (2026-07 ldm review)
#
# Stage 1 trains through AutoencoderPretrainGenerator.forward -> AutoencoderKL
# .encode()/.decode(); encode() never touches quant_conv, so it keeps its random
# init in the stage-1 checkpoint. encode_to_latent used to apply it anyway, and
# decode_from_latent never inverted it -- a random 1x1 channel mix wedged
# between the trained encoder and the trained decoder.
# --------------------------------------------------------------------------


def test_encode_to_latent_matches_the_encode_stage1_trained() -> None:
    """The stage-2 latent must be the stage-1 encoder's mean, up to standardisation."""
    import torch

    from spectramr.models.generators.latent_diffusion_generator import (
        LatentDiffusionGenerator,
    )

    gen = LatentDiffusionGenerator(
        in_channels=1, out_channels=1, latent_channels=4, base_channels=8, spatial_dims=2
    )
    gen.eval()
    # Make quant_conv conspicuously non-identity, as a random init would be.
    with torch.no_grad():
        gen.autoencoder.quant_conv.weight.normal_(0.0, 1.0)
        gen.autoencoder.quant_conv.bias.normal_(0.0, 1.0)

    x = torch.randn(2, 1, 32, 32)
    z = gen.encode_to_latent(x)

    with torch.no_grad():
        mean, _ = gen.autoencoder.encode(x)
        expected = (mean - gen.latent_mean) / (gen.latent_std + 1e-8)

    assert z.shape == expected.shape
    torch.testing.assert_close(z, expected)


def test_quant_conv_receives_no_gradient_from_the_stage1_objective() -> None:
    """Pins WHY the fix is required rather than just the fix."""
    import torch

    from spectramr.models.generators.autoencoder_pretrain_generator import (
        AutoencoderPretrainGenerator,
    )

    gen = AutoencoderPretrainGenerator(
        in_channels=1, out_channels=1, latent_channels=4, base_channels=8
    )
    out = gen(torch.randn(2, 1, 32, 32))
    out["reconstruction"].square().mean().backward()

    assert gen.autoencoder.quant_conv.weight.grad is None, (
        "quant_conv now takes gradient in stage 1 — if the stage-1 objective was "
        "changed to train it, encode_to_latent must apply it again."
    )
    # The decoder-side twin IS trained, because decode() applies it.
    assert gen.autoencoder.post_quant_conv.weight.grad is not None
