"""Unit tests for the pretrained Stable-Diffusion VAE backbone adapter.

The adapter wraps a frozen ``diffusers`` ``AutoencoderKL`` so it can serve as the
latent encoder for :class:`LatentDiffusionGenerator` (transfer learning instead
of training a VAE from scratch). ``diffusers`` is an optional dependency and is
NOT required to run these tests — behaviour is exercised via a small injected
fake VAE that mimics the ``diffusers`` API (``encode(x).latent_dist`` /
``decode(z).sample`` / ``.config``). Only the "missing dependency" test touches
the real import path (via monkeypatch).
"""

from __future__ import annotations

import math
import sys

import pytest
import torch
import torch.nn as nn

from mriforge.models.generators.pretrained_vae_adapter import (
    PretrainedSDVAEAdapter,
    reduce_from_rgb,
    replicate_to_rgb,
)

SF = 0.18215  # Stable-Diffusion VAE latent scaling factor


class _FakeLatentDist:
    def __init__(self, mean: torch.Tensor, logvar: torch.Tensor) -> None:
        self.mean = mean
        self.logvar = logvar

    def sample(self) -> torch.Tensor:
        return self.mean

    def mode(self) -> torch.Tensor:
        return self.mean


class _FakeEncodeOut:
    def __init__(self, latent_dist: _FakeLatentDist) -> None:
        self.latent_dist = latent_dist


class _FakeDecodeOut:
    def __init__(self, sample: torch.Tensor) -> None:
        self.sample = sample


class _FakeConfig:
    scaling_factor = SF
    latent_channels = 4
    block_out_channels = (128, 256, 512, 512)  # 4 stages -> 2**3 = 8x downsample


class FakeDiffusersVAE(nn.Module):
    """Minimal stand-in for ``diffusers.AutoencoderKL`` (3-channel, 8x)."""

    def __init__(self) -> None:
        super().__init__()
        self.config = _FakeConfig()
        # 3ch -> 8ch (=2*latent) at 1/8 resolution; deterministic so scaling is testable.
        self.enc = nn.Conv2d(3, 8, kernel_size=3, stride=8, padding=1)
        self.dec = nn.ConvTranspose2d(4, 3, kernel_size=8, stride=8)

    def encode(self, x: torch.Tensor) -> _FakeEncodeOut:
        h = self.enc(x)
        mean, logvar = torch.chunk(h, 2, dim=1)
        return _FakeEncodeOut(_FakeLatentDist(mean, logvar))

    def decode(self, z: torch.Tensor) -> _FakeDecodeOut:
        return _FakeDecodeOut(self.dec(z))


def _adapter(**kw) -> PretrainedSDVAEAdapter:
    return PretrainedSDVAEAdapter(vae=FakeDiffusersVAE(), **kw)


# --- channel adaptation helpers -------------------------------------------------


def test_replicate_to_rgb_broadcasts_grayscale():
    x = torch.randn(2, 1, 16, 16)
    out = replicate_to_rgb(x, in_channels=1)
    assert out.shape == (2, 3, 16, 16)
    # each RGB channel is a copy of the single grayscale channel
    assert torch.equal(out[:, 0], x[:, 0])
    assert torch.equal(out[:, 2], x[:, 0])


def test_replicate_to_rgb_passthrough_when_already_rgb():
    x = torch.randn(2, 3, 16, 16)
    assert torch.equal(replicate_to_rgb(x, in_channels=3), x)


def test_replicate_to_rgb_rejects_unsupported_channel_count():
    with pytest.raises(ValueError):
        replicate_to_rgb(torch.randn(2, 2, 16, 16), in_channels=2)


def test_reduce_from_rgb_averages_to_grayscale():
    x = torch.randn(2, 3, 16, 16)
    out = reduce_from_rgb(x, out_channels=1)
    assert out.shape == (2, 1, 16, 16)
    assert torch.allclose(out[:, 0], x.mean(dim=1))


# --- adapter behaviour ----------------------------------------------------------


def test_encode_returns_mean_logvar_at_downsampled_latent():
    z_mean, z_logvar = _adapter().encode(torch.randn(2, 1, 64, 64))
    assert z_mean.shape == (2, 4, 8, 8)
    assert z_logvar.shape == (2, 4, 8, 8)


def test_encode_applies_latent_scaling_factor():
    vae = FakeDiffusersVAE()
    adapter = PretrainedSDVAEAdapter(vae=vae)
    x = torch.randn(2, 1, 64, 64)
    raw_mean = vae.encode(replicate_to_rgb(x, 1)).latent_dist.mean
    z_mean, z_logvar = adapter.encode(x)
    # mean scaled by sf; logvar shifted by 2*log(sf) (variance scales by sf**2)
    assert torch.allclose(z_mean, raw_mean * SF, atol=1e-6)
    raw_logvar = vae.encode(replicate_to_rgb(x, 1)).latent_dist.logvar
    assert torch.allclose(z_logvar, raw_logvar + 2 * math.log(SF), atol=1e-5)


def test_decode_reduces_to_out_channels_at_full_resolution():
    out = _adapter().decode(torch.randn(2, 4, 8, 8))
    assert out.shape == (2, 1, 64, 64)


def test_downsample_factor_matches_block_stages():
    assert _adapter().downsample_factor == 8


def test_freeze_disables_grad_on_vae_and_sets_eval():
    adapter = _adapter(freeze=True)
    assert all(not p.requires_grad for p in adapter.vae.parameters())
    assert not adapter.vae.training


def test_no_freeze_keeps_grad_enabled():
    adapter = _adapter(freeze=False)
    assert any(p.requires_grad for p in adapter.vae.parameters())


def test_frozen_vae_stays_eval_when_parent_is_set_to_train():
    # A frozen pretrained backbone must NOT re-enter train mode when the parent
    # LDM calls .train() (would flip GroupNorm/dropout stats on frozen weights).
    adapter = _adapter(freeze=True)
    adapter.train()
    assert not adapter.vae.training


def test_latent_channel_mismatch_raises():
    # adapter told to expect 8 latent channels, fake VAE provides 4
    with pytest.raises(ValueError, match="latent_channels"):
        PretrainedSDVAEAdapter(vae=FakeDiffusersVAE(), latent_channels=8)


def test_exposes_identity_quant_convs_for_encode_to_latent_path():
    # LatentDiffusionGenerator.encode_to_latent touches .quant_conv/.post_quant_conv;
    # they must be no-ops here so the diffusers latent is not transformed twice.
    adapter = _adapter()
    z = torch.randn(2, 4, 8, 8)
    assert torch.equal(adapter.post_quant_conv(z), z)
    doubled = torch.cat([z, torch.zeros_like(z)], dim=1)
    assert torch.equal(adapter.quant_conv(doubled), doubled)


def test_missing_diffusers_raises_clear_error(monkeypatch):
    # Force `import diffusers` to fail regardless of whether it is installed.
    monkeypatch.setitem(sys.modules, "diffusers", None)
    with pytest.raises(ImportError, match="diffusers") as exc_info:
        PretrainedSDVAEAdapter(vae=None)
    # The install hint must advertise the REAL optional-dependency extra
    # (pyproject: ``diffusion = ["diffusers>=0.30"]``). It previously pointed
    # at a non-existent ``generative`` extra, so `pip install -e '.[generative]'`
    # failed and the cluster sdvae headline arm could never be installed (F4).
    msg = str(exc_info.value)
    assert ".[diffusion]" in msg
    assert "generative" not in msg


# --------------------------------------------------------------------------- #
# Offline-node load failure is actionable (2026-07-25)
# --------------------------------------------------------------------------- #
def test_vae_load_failure_names_the_cause_and_the_knob(monkeypatch):
    """A Hub-unreachable node must produce an actionable error, not diffusers'.

    ``stage2_ldm_ulf_to_hf_sdvae`` died on 2026-07-25 (SLURM 7796517 task 4) with
    diffusers' stock message, whose advice ("make sure you don't have a local
    directory with the same name") points away from the real cause: the compute
    node has no outbound network and the weights were never in the local cache.
    The wrapper must name the offline cause and both remedies — pre-fetching, and
    the ``vae_pretrained_id`` knob that accepts a local directory.
    """
    import types

    from mriforge.models.generators import pretrained_vae_adapter as mod

    class _Boom:
        @staticmethod
        def from_pretrained(_id):
            raise OSError("Can't load the model for 'stabilityai/sd-vae-ft-mse'.")

    monkeypatch.setattr(
        mod.importlib,
        "import_module",
        lambda name: types.SimpleNamespace(AutoencoderKL=_Boom),
    )

    with pytest.raises(RuntimeError) as excinfo:
        mod._load_diffusers_vae("stabilityai/sd-vae-ft-mse")

    msg = str(excinfo.value)
    assert "offline" in msg.lower(), "must name the offline cause"
    assert "vae_pretrained_id" in msg, "must name the knob that takes a local path"
    assert "huggingface-cli download" in msg, "must give the pre-fetch remedy"
    assert "stabilityai/sd-vae-ft-mse" in msg, "must name the id that failed"
    # The original error stays attached for debugging.
    assert isinstance(excinfo.value.__cause__, OSError)
