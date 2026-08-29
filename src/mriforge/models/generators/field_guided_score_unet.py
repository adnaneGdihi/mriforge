"""Field-guided score network for continuous-field guided diffusion (B-3.5).

``FieldGuidedScoreUNet`` is the ε-prediction score net for source-conditioned,
field-conditioned diffusion translation. It predicts the noise ``ε_θ(x_t, t, b | src)``
where ``x_t`` is the noisy target-field image, ``t`` the diffusion timestep, ``b`` the
**continuous** target field strength (Tesla, ``b = 10**(log10 B0)``), and ``src`` the
source-field image (always-on translation condition, concatenated to ``x_t``).

The distinctive piece (vs vanilla conditional diffusion) is **classifier-free
guidance on the continuous field**: the field enters via :class:`FieldFiLMBlock`
(field + timestep embedding -> per-channel gamma/beta), and a **learned NULL-field
FiLM** is substituted per-sample when the field is dropped. Training drops the field
with probability ``guidance_prob``; sampling combines the field-conditional and
null-field scores with ``guidance_scale``. The field is therefore a genuine, non-inert
conditioning channel (anti-facade, pitfall #16): with the null path learned
separately, the guided score ``eps_null + w*(eps_field - eps_null)`` actually depends
on ``b``.

Magnitude-only (1→1 for ``x``; ``src`` is a 1-channel condition); raises on complex.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from mriforge.models.blocks.contrast_conditioning import (
    build_contrast_sequence,
    contrast_sequence_dim,
)
from mriforge.models.blocks.embeddings import SinusoidalPositionEmbedding
from mriforge.models.blocks.field_film_modulation import FieldFiLMBlock
from mriforge.models.registry import register_model


def _conv_block(cin: int, cout: int) -> nn.Sequential:
    g = min(8, cout)
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(g, cout),
        nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(g, cout),
        nn.SiLU(),
    )


@register_model(name="field_guided_score_unet", training_mode="field_guided_diffusion")
class FieldGuidedScoreUNet(nn.Module):
    """ε-net conditioned on (timestep, continuous field) + a learned null for CFG."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        cond_channels: int = 1,
        width: int = 64,
        time_dim: int = 64,
        use_contrast_conditioning: bool = False,
        num_contrasts: int = 3,
    ) -> None:
        super().__init__()
        if in_channels != 1 or out_channels != 1:
            raise ValueError(
                "FieldGuidedScoreUNet is magnitude-only (x is 1->1); "
                f"got in={in_channels}, out={out_channels}."
            )
        self.cond_channels = int(cond_channels)
        self.use_contrast_conditioning = bool(use_contrast_conditioning)
        self.num_contrasts = int(num_contrasts)
        # timestep -> sinusoidal -> small MLP, used as the FiLM sequence vector.
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        self.body_in = _conv_block(in_channels + self.cond_channels, width)
        # FiLM conditions on (continuous field, timestep embedding), plus a contrast
        # one-hot appended to the timestep vector when contrast conditioning is on.
        seq_dim = contrast_sequence_dim(
            time_dim, self.num_contrasts, self.use_contrast_conditioning
        )
        self.film = FieldFiLMBlock(num_channels=width, sequence_dim=seq_dim)
        # Learned NULL-field FiLM: substituted when the field is dropped (CFG).
        # gamma~1 / beta~0 at init so the null path starts as the identity (matches
        # FieldFiLMBlock's identity init).
        self.null_gamma = nn.Parameter(torch.ones(width))
        self.null_beta = nn.Parameter(torch.zeros(width))
        self.body_out = nn.Sequential(
            _conv_block(width, width),
            nn.Conv2d(width, out_channels, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        timesteps: torch.Tensor,
        field_strength: torch.Tensor,
        cond_image: torch.Tensor | None = None,
        field_cond: torch.Tensor | None = None,
        contrast_id: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        """Predict ε.

        Args:
            x: noisy target image ``[B, 1, H, W]``.
            timesteps: diffusion step indices ``[B]``.
            field_strength: continuous target field (Tesla) ``[B]`` or ``[B, 1]``.
            cond_image: source-field image ``[B, cond_channels, H, W]`` (required when
                ``cond_channels > 0``); the always-on translation condition.
            field_cond: per-sample mask ``[B]`` — 1 keeps the field FiLM, 0 uses the
                learned null (CFG dropout / unconditional sampling). ``None`` = all 1.
        """
        if torch.is_complex(x):
            raise ValueError("FieldGuidedScoreUNet expects magnitude (real) input.")
        if self.cond_channels > 0:
            if cond_image is None:
                raise ValueError(
                    "FieldGuidedScoreUNet requires cond_image (the source image) "
                    f"when cond_channels={self.cond_channels}."
                )
            inp = torch.cat([x, cond_image], dim=1)
        else:
            inp = x

        t_emb = self.time_embed(timesteps.float().reshape(-1))  # [B, time_dim]
        b = field_strength.reshape(-1, 1).float()
        # Append the contrast one-hot to the timestep vector when conditioning is
        # on (raises on missing/out-of-range ids, #15/#9); passthrough when off.
        seq = build_contrast_sequence(
            t_emb,
            contrast_id,
            self.num_contrasts,
            self.use_contrast_conditioning,
            batch_size=t_emb.shape[0],
            device=t_emb.device,
            dtype=t_emb.dtype,
        )
        gamma, beta = self.film(b, seq)  # [B, width]
        if field_cond is not None:
            m = field_cond.reshape(-1, 1).to(gamma.dtype)  # 1=field, 0=null
            gamma = m * gamma + (1.0 - m) * self.null_gamma.unsqueeze(0)
            beta = m * beta + (1.0 - m) * self.null_beta.unsqueeze(0)

        h = self.body_in(inp)
        view = (h.shape[0], h.shape[1], 1, 1)
        h = h * gamma.view(*view) + beta.view(*view)
        return self.body_out(h)
