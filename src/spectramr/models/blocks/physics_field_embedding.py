"""Physics-grounded field embedding for the FieldScore prior (MICCAI MRIxFields2026, A-5.2).

Maps a static-field strength ``b0`` (Tesla) to a PHYSICS-grounded conditioning vector built from the
Bottomley ``T1(B0) = a·B0^alpha`` relaxation dispersion (white/gray matter, CSF) rather than the raw
field value. The dispersion is the physically-correct nonlinear law, so the embedding interpolates
the field continuum BY CONSTRUCTION (a smooth, monotone basis a raw-``b0`` scalar does not provide);
whether a model that consumes it actually GENERALISES better to unseen fields is the empirical
question the ``enable_physics`` ablation tests — not an asserted guarantee. This is the distinctive
seam of A-5.2 vs the raw-field FiLM of B-3.5
:class:`~spectramr.models.generators.field_guided_score_unet.FieldGuidedScoreUNet`.

``enable_physics=False`` zeroes the relaxation features (keeping only ``log10 b0``) — the
parameter-matched one-knob ablation isolating the physics dispersion from raw-field conditioning.
NOTE: the CSF feature (``alpha=0.04``) is near field-invariant by physics — a low-variance but
correct prior fact (CSF T1 barely disperses), not a dead input; WM/GM carry the field discrimination.
"""

from __future__ import annotations

import torch
from torch import nn

from spectramr.infrastructure.physics.relaxation_priors import TissueClass, bottomley_t1_tensor

# Normalisation so each feature is O(1): T1 in ms (~hundreds to ~thousands), log-field in ~[-1, 1].
_T1_REF_MS = 1500.0
_TISSUES = (TissueClass.WHITE_MATTER, TissueClass.GRAY_MATTER, TissueClass.CSF)


class PhysicsFieldEmbedding(nn.Module):
    """Field strength -> physics (Bottomley T1-dispersion) conditioning vector (A-5.2).

    Output dim is ``1 + len(tissues)`` = 4: ``[log10 b0, T1_wm/ref, T1_gm/ref, T1_csf/ref]``.
    """

    embed_dim: int = 1 + len(_TISSUES)

    def __init__(self, enable_physics: bool = True) -> None:
        super().__init__()
        if not isinstance(enable_physics, bool):
            raise ValueError(f"enable_physics must be a bool; got {type(enable_physics).__name__}.")
        self.enable_physics = bool(enable_physics)

    def forward(self, field_strength: torch.Tensor) -> torch.Tensor:
        """Return ``[B, embed_dim]`` physics conditioning for per-sample ``b0`` (Tesla)."""
        b = field_strength.reshape(-1).float()
        if bool((b <= 0.0).any()):  # raise, don't silently clamp (pitfall #9); fields are > 0 T
            raise ValueError(f"field_strength must be > 0 T; got min={float(b.min()):.4g}.")
        log_field = torch.log10(b).unsqueeze(-1)  # [B, 1]
        if self.enable_physics:
            t1 = torch.stack(
                [bottomley_t1_tensor(b, t) / _T1_REF_MS for t in _TISSUES], dim=-1
            )  # [B, 3] physics dispersion
        else:
            t1 = torch.zeros(b.shape[0], len(_TISSUES), device=b.device, dtype=b.dtype)
        return torch.cat([log_field, t1], dim=-1)  # [B, embed_dim]
