"""McCann single-potential field-path generator (MICCAI MRIxFields2026, B-3.9).

A SINGLE global input-convex potential ``psi(x)`` (NOT field-conditioned) parameterises the
whole cross-field family as one monotone displacement path (the McCann/Brenier displacement
form). The displacement is ``d(x) = map_scale * grad_x psi(x)`` (= ``brenier_map(x) - x``); a
learnable monotone path ``t(b) = sigmoid(a*log10(B0) + c)`` (``a`` floored > 0 so the path
cannot collapse to a constant) maps field strength to a parameter in ``[0,1]``, and translating
``b_s -> b_t`` moves along the path::

    x_out = x + (t(b_t) - t(b_s)) * d(x)

so ONE convex potential serves every field pair (any-to-any, Task 3); same source/target field
gives the identity. Distinct from B-1.5 (one potential PER source field, -> 7T only), B-3.1
(velocity ODE) and B-3.3 (stochastic bridge). ``enforce_convexity=False`` is the one-knob
ablation. HONEST SCOPE: trained by paired L1 (not a semi-dual OT objective), so this is a
convex-constrained monotone path, NOT a guaranteed W2-optimal geodesic; ``d(x)`` is curl-free,
limiting spatially-varying changes. Magnitude-only (1->1).
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import nn

from spectramr.models.blocks.input_convex_neural_network import InputConvexNeuralNetwork
from spectramr.models.registry import register_model


@register_model(name="mccann_geodesic_icnn", training_mode="mccann_field_path")
class McCannGeodesicICNN(nn.Module):
    """Single global Brenier potential + learnable McCann field path (B-3.9)."""

    # Near-identity at init by design (small displacement); ALSO the probe synthesises
    # field_strength == field_strength_target, for which McCann is correctly the identity
    # (zero increment). Both make the identity-collapse heuristic a false positive. dead_loss
    # is suppressed for the same reason the identity start causes it: the real objective is
    # computed by McCannFieldPathStrategy (compute_mccann_loss), NOT the placeholder
    # image_losses, and the zero-increment identity drives that placeholder l1 to ~0 on the
    # probe's phantom — a false dead-loss verdict (the arm trains on the cluster, g_total=0.0758).
    # The input-invariance (#20) check stays active and passes (the map is input-dependent).
    synthetic_forward_probe_skip: ClassVar[set[str]] = {"identity_collapse", "dead_loss"}

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: int = 32,
        n_layers: int = 3,
        map_scale_init: float = 0.1,
        enforce_convexity: bool = True,
    ) -> None:
        super().__init__()
        if in_channels != 1 or out_channels != 1:
            raise ValueError(
                f"McCannGeodesicICNN is magnitude-only (1->1); got in={in_channels}, out={out_channels}."
            )
        self.icnn = InputConvexNeuralNetwork(
            in_channels=1,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            field_dim=0,  # GLOBAL potential — one geodesic for all field pairs
            map_scale_init=map_scale_init,
            enforce_convexity=enforce_convexity,
        )
        # Monotone path t(b) = sigmoid(a*log10(B0) + c); a,c learnable (a>0 keeps it monotone
        # in field, enforced via softplus at use). Init a=1,c=0.
        self.path_a_raw = nn.Parameter(torch.tensor(0.5413, dtype=torch.float32))  # softplus->~1
        self.path_c = nn.Parameter(torch.zeros((), dtype=torch.float32))
        # Floor on the geodesic slope so the path can't collapse to a constant (a->0 -> every
        # field maps identically -> the whole McCann mechanism becomes a no-op, pitfall #20).
        self._path_a_min = 1.0e-2

    def path_slope(self) -> torch.Tensor:
        """Resolved monotone-path slope ``a`` (softplus + floor); always > 0."""
        return torch.nn.functional.softplus(self.path_a_raw) + self._path_a_min

    def path_function(self, field_strength: torch.Tensor) -> torch.Tensor:
        """Monotone field->geodesic-parameter map t(b) in (0,1)."""
        tau = torch.log10(field_strength.reshape(-1).float().clamp_min(1e-3))
        return torch.sigmoid(self.path_slope() * tau + self.path_c)

    def resolved_path(self) -> dict[str, float]:
        """Provenance: the resolved path params (so a collapsed path is visible post-hoc)."""
        return {"path_a": float(self.path_slope().detach()), "path_c": float(self.path_c.detach())}

    def forward(
        self,
        x: torch.Tensor,
        *,
        field_strength: torch.Tensor,
        field_strength_target: torch.Tensor,
        **_: Any,
    ) -> torch.Tensor:
        # Both fields REQUIRED (no default) so the Tier-2 probe injects them.
        if torch.is_complex(x):
            raise ValueError("McCannGeodesicICNN expects magnitude (real) input.")
        displacement = self.icnn.brenier_map(x) - x  # d(x) = map_scale * grad psi(x)
        t_s = self.path_function(field_strength).reshape(-1, 1, 1, 1)
        t_t = self.path_function(field_strength_target).reshape(-1, 1, 1, 1)
        return x + (t_t - t_s) * displacement  # McCann displacement interpolation
