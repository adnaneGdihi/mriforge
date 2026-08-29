r"""FPSP -- Field-Pushforward Score Prior.

Trains a single score network :math:`s_\theta(\boldsymbol\xi,t)\approx\nabla_
{\boldsymbol\xi}\log p_t(\boldsymbol\xi)` on *field-independent* tissue-parameter
maps. For any field/protocol the image prior is the pushforward through the
deterministic signal map :math:`I=\mathcal A(\boldsymbol\xi;\boldsymbol\varphi,
B_0)`:

.. math::

   p_{\boldsymbol\varphi,B_0}(I)
     = \bigl(\mathcal A_{\boldsymbol\varphi,B_0}\bigr)_{\#}\,p(\boldsymbol\xi).

The field dependence of the image prior is carried entirely by the known physics,
not by the learned model, so one trained prior serves every regime. The score is
field-invariant *by construction*: its forward takes only the parameter map and
the diffusion time, never a field or acquisition argument. Reconstruction /
synthesis runs the reverse-time SDE in parameter space with a data-consistency
term and renders images through :func:`field_pushforward`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.models.registry import register_model

_CONTRAST_TO_SEQ = {"spin_echo": 0, "t2": 0, "inversion_recovery": 1, "flair": 1}


def field_pushforward(
    params: torch.Tensor,
    acquisition: dict[str, float],
    contrast_type: str = "spin_echo",
) -> torch.Tensor:
    r"""Deterministic signal-map pushforward :math:`I=\mathcal A(\boldsymbol\xi;
    \boldsymbol\varphi,B_0)`.

    Renders an image from a parameter map through the Bloch SSOT. The field /
    protocol dependence lives entirely here, not in the score network.

    Args:
        params: ``[B, 3, ...]`` ``(rho, T1_ms, T2_ms)`` parameter map.
        acquisition: ``{"TR","TE","TI"}`` (ms).
        contrast_type: Contrast label routed to the Bloch forward.

    Returns:
        Rendered image ``[B, 1, ...]``.
    """
    from mriforge.infrastructure.physics.multi_physics_bloch import (
        MultiPhysicsBlochLayer,
    )

    seq = _CONTRAST_TO_SEQ.get(str(contrast_type).lower())
    if seq is None:
        raise ValueError(
            f"Unknown contrast_type {contrast_type!r}; expected one of {sorted(_CONTRAST_TO_SEQ)}."
        )
    layer = MultiPhysicsBlochLayer(learnable_flip_angle=False).to(params.device)
    return layer(
        {"rho": params[:, 0:1], "t1": params[:, 1:2], "t2": params[:, 2:3]},
        {
            "TR": float(acquisition["TR"]),
            "TE": float(acquisition["TE"]),
            "TI": float(acquisition.get("TI", 0.0)),
            "contrast_type": int(seq),
        },
    )


@register_model(
    name="fpsp_score",
    training_mode="diffusion",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
)
class FieldPushforwardScorePrior(nn.Module):
    r"""Parameter-space score network :math:`s_\theta(\boldsymbol\xi,t)`.

    Operates purely in tissue-parameter space (no field/acquisition input), so it
    is field-invariant by construction. The image prior at any field is obtained
    by :func:`field_pushforward`.

    Args:
        param_channels: Tissue-parameter channels (3 = rho, T1, T2).
        width: Hidden width.
        depth: Number of conv blocks.
    """

    def __init__(self, param_channels: int = 3, width: int = 32, depth: int = 3) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(1, width), nn.GELU(), nn.Linear(width, width))
        layers: list[nn.Module] = [nn.Conv2d(param_channels, width, 3, padding=1)]
        for _ in range(max(depth - 1, 0)):
            layers += [nn.GELU(), nn.Conv2d(width, width, 3, padding=1)]
        self.body = nn.ModuleList(layers)
        self.head = nn.Conv2d(width, param_channels, 3, padding=1)

    def forward(self, params: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Score estimate in parameter space.

        Args:
            params: Noised parameter map ``[B, param_channels, H, W]``.
            t: Diffusion time ``[B]`` in ``[0, 1]``.

        Returns:
            Score ``[B, param_channels, H, W]``.
        """
        t_embed = self.time_mlp(t.reshape(-1, 1).to(params.dtype))[:, :, None, None]
        h = params
        for module in self.body:
            h = module(h)
            if isinstance(module, nn.Conv2d) and h.shape[1] == t_embed.shape[1]:
                h = h + t_embed
        return self.head(h)


__all__ = ["FieldPushforwardScorePrior", "field_pushforward"]
