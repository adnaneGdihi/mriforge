"""Physics-embedded field-conditioned score PRIOR (MICCAI MRIxFields2026, A-5.2, FieldScore).

A STANDALONE, reusable score (eps-prediction) prior ``eps_theta(x_t, t, b)`` over magnitude images
at a continuous field ``b`` (Tesla). Two things make it a genuine prior, distinct from B-3.5
:class:`~mriforge.models.generators.field_guided_score_unet.FieldGuidedScoreUNet` (a field-guided
TRANSLATION score that always conditions on a source image):

* **Unconditional** (no source image, ``cond_channels=0``): it models ``p(x | b)`` and is therefore
  usable in ANY measurement likelihood — e.g.
  :class:`~mriforge.models.diffusion.posterior_langevin.PosteriorLangevinEnsemble` / DPS — as a
  reusable cross-field prior, not tied to its training strategy. Its ``eps(x_t, timesteps=, \
  field_strength=)`` signature is bridged to the sampler's ``score_model(x_t, y, t)`` contract by
  the shipped :class:`~mriforge.models.diffusion.field_score_prior_adapter.FieldScorePriorAdapter`.
* **Physics-embedded conditioning**: the field enters through the Bottomley ``T1(B0)`` relaxation
  dispersion (:class:`~mriforge.models.blocks.physics_field_embedding.PhysicsFieldEmbedding`) rather
  than a raw ``b0`` scalar. The dispersion interpolates the field continuum BY CONSTRUCTION (the
  physically-correct nonlinear law — verified in the embedding unit test); whether the trained model
  LEVERAGES this for better unseen-field reconstruction is the empirical question the
  ``enable_physics`` ablation (and a held-out-field evaluation) measures, not an asserted guarantee.

``enable_physics`` (``model_kwargs``) toggles the dispersion features vs raw-field-only conditioning
(the parameter-matched one-knob ablation). A learned NULL-field FiLM gives classifier-free guidance
on the field. Magnitude-only (1->1); raises on complex.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from mriforge.models.blocks.embeddings import SinusoidalPositionEmbedding
from mriforge.models.blocks.field_film_modulation import FieldFiLMBlock
from mriforge.models.blocks.physics_field_embedding import PhysicsFieldEmbedding
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


@register_model(
    name="field_physics_score_unet",
    training_mode="field_guided_diffusion",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
)
class FieldPhysicsEmbeddedScoreUNet(nn.Module):
    """Standalone physics-embedded field-conditioned score prior (A-5.2)."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 64,
        time_dim: int = 64,
        enable_physics: bool = True,
    ) -> None:
        super().__init__()
        if in_channels != 1 or out_channels != 1:
            raise ValueError(
                f"FieldPhysicsEmbeddedScoreUNet is magnitude-only (1->1); got in={in_channels}, "
                f"out={out_channels}."
            )
        self.enable_physics = bool(enable_physics)
        self.physics_embed = PhysicsFieldEmbedding(enable_physics=self.enable_physics)
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        self.body_in = _conv_block(in_channels, width)
        # FiLM conditions on (log10 b0 scalar, [timestep emb ; physics T1-dispersion features]).
        self._seq_dim = time_dim + (PhysicsFieldEmbedding.embed_dim - 1)
        self.film = FieldFiLMBlock(num_channels=width, sequence_dim=self._seq_dim)
        # Learned NULL-field FiLM for classifier-free guidance (identity init).
        self.null_gamma = nn.Parameter(torch.ones(width))
        self.null_beta = nn.Parameter(torch.zeros(width))
        self.body_out = nn.Sequential(_conv_block(width, width), nn.Conv2d(width, out_channels, 1))

    def forward(
        self,
        x: torch.Tensor,
        *,
        timesteps: torch.Tensor,
        field_strength: torch.Tensor,
        field_cond: torch.Tensor | None = None,
        cond_image: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        """Predict eps for the noisy image ``x`` [B,1,H,W] at step ``timesteps`` and field ``b``.

        ``cond_image`` is declared and INTENTIONALLY IGNORED (not silently swallowed): this is an
        UNCONDITIONAL prior ``p(x|b)``, but it shares the ``field_guided_diffusion`` strategy with
        the source-conditioned B-3.5 translator, which passes the source image — declaring the
        parameter makes the "no source" contract explicit (pitfall #9) rather than a ``**_`` drop.
        """
        if torch.is_complex(x):
            raise ValueError("FieldPhysicsEmbeddedScoreUNet expects magnitude (real) input.")
        t_emb = self.time_embed(timesteps.float().reshape(-1))  # [B, time_dim]
        phys = self.physics_embed(field_strength)  # [B, embed_dim]
        log_field = phys[:, 0:1]  # scalar field arg for the FiLM
        seq = torch.cat([t_emb, phys[:, 1:]], dim=-1)  # [B, time_dim + dispersion]
        gamma, beta = self.film(log_field, seq)
        if field_cond is not None:
            m = field_cond.reshape(-1, 1).to(gamma.dtype)
            gamma = m * gamma + (1.0 - m) * self.null_gamma.unsqueeze(0)
            beta = m * beta + (1.0 - m) * self.null_beta.unsqueeze(0)
        h = self.body_in(x)
        view = (h.shape[0], h.shape[1], 1, 1)
        h = h * gamma.view(*view) + beta.view(*view)
        return self.body_out(h)
