"""Pretrained Stable-Diffusion VAE backbone for the latent diffusion generator.

The two-stage LDM recipe assumes a *pretrained* autoencoder whose latent is a
broad image prior. Training :class:`AutoencoderKL` from scratch on a tiny paired
cohort (see ``experiments/inprogress/ldm_two_stage_ulf_to_hf/stage1_vae_hf.yaml``)
yields only an overfit per-subject latent — not a generative prior. This adapter
lets stage 2 instead load a **frozen** ``diffusers`` ``AutoencoderKL`` (e.g.
``stabilityai/sd-vae-ft-mse``, whose ``latent_channels=4`` + 8x downsampling
already match the stage-2 kwargs) as a drop-in latent encoder — transfer learning
with no from-scratch stage 1.

It exposes exactly the surface :class:`LatentDiffusionGenerator` expects from its
``self.autoencoder``:

* ``encode(x) -> (mean, logvar)`` — Gaussian latent params (NOT a diffusers
  ``AutoencoderKLOutput``), scaled by the VAE's ``scaling_factor`` so the latent
  the diffusion U-Net sees is ~unit-variance (SD convention).
* ``decode(z) -> image`` — un-scales and maps back to ``out_channels``.
* ``downsample_factor`` attribute.
* no-op ``quant_conv`` / ``post_quant_conv`` (diffusers applies them *inside*
  ``encode``, so the generator's ``encode_to_latent`` path must not re-apply them).

``diffusers`` is an **optional** dependency (declared under the ``generative``
extra in ``pyproject.toml``). Construction with ``vae=None`` and ``diffusers``
absent raises a clear :class:`ImportError` — CLAUDE.md #9 forbids silent
fallbacks (a missing backbone must not degrade to a random latent).

Grayscale MRI is 1-channel and the SD VAE is 3-channel RGB; the adapter
replicates the single channel to RGB on encode and averages the 3 channels back
on decode (fixed, parameter-free, so the encoder stays fully frozen). MRI must be
normalised to ``[-1, 1]`` (the SD VAE's expected input range) — matched by the
configs' ``normalization_type: minmax`` / ``rescale_range: [-1.0, 1.0]``.
"""

from __future__ import annotations

import importlib
import math

import torch
import torch.nn as nn

DEFAULT_SD_VAE_ID = "stabilityai/sd-vae-ft-mse"

__all__ = [
    "DEFAULT_SD_VAE_ID",
    "PretrainedSDVAEAdapter",
    "reduce_from_rgb",
    "replicate_to_rgb",
]


def replicate_to_rgb(x: torch.Tensor, in_channels: int) -> torch.Tensor:
    """Broadcast a 1-channel MRI tensor to the 3-channel RGB the SD VAE expects.

    A 3-channel input is passed through unchanged. Any other channel count raises
    (no silent reshape — CLAUDE.md #9).
    """
    if in_channels == 3:
        return x
    if in_channels == 1:
        return x.repeat(1, 3, 1, 1)
    raise ValueError(
        "PretrainedSDVAEAdapter supports 1-channel (grayscale) or 3-channel "
        f"(RGB) input, got in_channels={in_channels}."
    )


def reduce_from_rgb(x: torch.Tensor, out_channels: int) -> torch.Tensor:
    """Map a 3-channel SD-VAE reconstruction back to ``out_channels``.

    For ``out_channels == 1`` the three channels are averaged (the inverse of
    :func:`replicate_to_rgb`). ``out_channels == 3`` passes through. Otherwise
    raise.
    """
    if out_channels == 3:
        return x
    if out_channels == 1:
        return x.mean(dim=1, keepdim=True)
    raise ValueError(
        "PretrainedSDVAEAdapter can reduce the SD-VAE's 3-channel output to 1 "
        f"(grayscale) or 3 (RGB) channels, got out_channels={out_channels}."
    )


def _load_diffusers_vae(pretrained_id: str) -> nn.Module:
    """Load a pretrained ``diffusers`` ``AutoencoderKL`` or raise a clear error."""
    try:
        diffusers = importlib.import_module("diffusers")
    except Exception as exc:  # ImportError incl. sys.modules['diffusers'] = None
        raise ImportError(
            "vae_backbone='sd' requires the optional 'diffusers' package, which "
            "is not installed. Install it (e.g. `pip install diffusers` or the "
            "project's 'diffusion' extra: `pip install -e '.[diffusion]'`) to "
            "use a pretrained Stable-Diffusion VAE as the latent encoder."
        ) from exc
    try:
        return diffusers.AutoencoderKL.from_pretrained(pretrained_id)
    except Exception as exc:
        # A compute node with no outbound network cannot reach the Hub, and
        # diffusers' own message ("make sure you don't have a local directory
        # with the same name") sends the reader down the wrong path. Name the
        # actual cause and the two ways out — this is how
        # stage2_ldm_ulf_to_hf_sdvae failed on 2026-07-25 (SLURM 7796517 task 4).
        raise RuntimeError(
            f"Could not load the pretrained VAE '{pretrained_id}'. On an "
            "offline compute node the weights must already be in the local "
            "HuggingFace cache. Either (a) pre-fetch them on a login node "
            f"(`huggingface-cli download {pretrained_id}`, sharing HF_HOME with "
            "the job), or (b) point the arm at a local directory: "
            "`model.model_kwargs.vae_pretrained_id: /path/to/sd-vae-ft-mse`. "
            f"Underlying error: {exc}"
        ) from exc


class PretrainedSDVAEAdapter(nn.Module):
    """Frozen pretrained SD ``AutoencoderKL`` presented as the LDM's autoencoder.

    Args:
        pretrained_id: HuggingFace repo id of the ``diffusers`` VAE to load when
            ``vae`` is not injected.
        in_channels: Channel count of the MRI input (1 for grayscale).
        out_channels: Channel count of the decoded output (1 for grayscale).
        latent_channels: Expected latent channels; validated against the VAE
            config (SD VAE = 4). A mismatch raises rather than silently truncates.
        freeze: Freeze all VAE parameters and keep the VAE in eval mode.
        vae: Optional pre-constructed VAE (dependency-injection seam for tests).
    """

    def __init__(
        self,
        pretrained_id: str = DEFAULT_SD_VAE_ID,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_channels: int = 4,
        freeze: bool = True,
        vae: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.pretrained_id = pretrained_id
        self.frozen = freeze

        if vae is None:
            vae = _load_diffusers_vae(pretrained_id)
        self.vae: nn.Module = vae

        config = getattr(vae, "config", None)
        vae_latent = int(getattr(config, "latent_channels", latent_channels))
        if vae_latent != latent_channels:
            raise ValueError(
                f"latent_channels mismatch: the LDM is configured for "
                f"{latent_channels} latent channels but VAE '{pretrained_id}' "
                f"provides {vae_latent}. Set model_kwargs.latent_channels to "
                f"{vae_latent} (and keep it in sync with the diffusion U-Net)."
            )

        self.scaling_factor = float(getattr(config, "scaling_factor", 1.0) or 1.0)
        block = getattr(config, "block_out_channels", None)
        self.downsample_factor = 2 ** (len(block) - 1) if block else 8

        # diffusers' AutoencoderKL applies its quant_conv/post_quant_conv INSIDE
        # encode()/decode(); LatentDiffusionGenerator.encode_to_latent calls them
        # again on self.autoencoder. Expose no-ops so the latent is not
        # transformed twice (the cat([z, zeros]) -> [:latent_channels] slice in
        # that path then round-trips z unchanged).
        self.quant_conv = nn.Identity()
        self.post_quant_conv = nn.Identity()

        if freeze:
            for param in self.vae.parameters():
                param.requires_grad_(False)
            self.vae.eval()

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an image to scaled Gaussian latent params ``(mean, logvar)``."""
        x_rgb = replicate_to_rgb(x, self.in_channels)
        latent_dist = self.vae.encode(x_rgb).latent_dist
        mean = latent_dist.mean * self.scaling_factor
        if self.scaling_factor != 1.0:
            # var scales by sf**2 -> logvar shifts by 2*log(sf)
            logvar = latent_dist.logvar + 2.0 * math.log(self.scaling_factor)
        else:
            logvar = latent_dist.logvar
        return mean, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a (scaled) latent back to an ``out_channels`` image."""
        image = self.vae.decode(z / self.scaling_factor).sample
        return reduce_from_rgb(image, self.out_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """VAE-style forward: ``(reconstruction, mean, logvar)``."""
        mean, logvar = self.encode(x)
        reconstruction = self.decode(mean)
        return reconstruction, mean, logvar

    def train(self, mode: bool = True) -> PretrainedSDVAEAdapter:
        """Keep a frozen VAE in eval mode even when the parent LDM trains."""
        super().train(mode)
        if self.frozen:
            self.vae.eval()
        return self
