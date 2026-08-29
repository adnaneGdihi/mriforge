r"""Counterfactual-healthy generator wrapper for anomaly saliency.

Implements §7 of the multi-contrast brainstorm — given a pathological
image :math:`\mathbf{x}_p`, encodes it through a *healthy-trained* base
generator and decodes back to its closest healthy counterpart
:math:`\mathbf{x}_h`. The voxel-wise difference

.. math::

   \mathbf{s}(\mathbf{r})
   \;=\; |\mathbf{x}_p(\mathbf{r}) - \mathbf{x}_h(\mathbf{r})|

is an *interpretable lesion saliency map* — voxels that the healthy
prior cannot explain. This dual-purpose pipeline:

1. Provides anomaly detection without lesion annotations
   (useful for unsupervised screening).
2. Generates a self-distillation signal for downstream lesion-aware
   training (the saliency map is the supervision target).

The wrapper is *generator-agnostic*: any generator with a clear
encode/decode split (DisentangledMRI, latent diffusion, autoencoders,
the Bloch-bottleneck) can be wrapped. The base generator must implement
either:

- ``encode(x) -> latent`` and ``decode(latent, contrast=None) -> x``, or
- A ``forward(x, contrast=None)`` that takes a contrast id and returns
  the reconstruction (in which case "encoding" is implicit).

The healthy prior is enforced by *which* contrast id is presented at
decode time: the wrapper sets ``contrast_idx="healthy"`` (configurable)
to push the latent through the healthy prior of the base generator.
For models without an explicit healthy/pathology axis, set
``healthy_index=0`` and train the base generator on the healthy subset
of the cohort.

Forward signature
-----------------
Returns a dict::

    {
        "counterfactual": Tensor (B, C, H, W),  # x_h
        "saliency":       Tensor (B, 1, H, W),  # |x_p - x_h|, channel-mean
        "reconstruction": Tensor (B, C, H, W),  # passthrough x_p for losses
    }

References
----------
[3]  B. Billot *et al.*, "SynthSeg: Segmentation of brain MRI scans of
     any contrast and resolution without retraining," *Med. Image
     Anal.*, vol. 86, p. 102789, 2023.
[11] H. Chung, J. C. Ye, "Score-based diffusion models for accelerated
     MRI," *Med. Image Anal.*, vol. 80, p. 102479, 2022.

§7 of ``docs/multi_contrast.md``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from mriforge.models.registry import (
    get_model_class,
    list_models_with_capability,
    register_model,
)

__all__ = ["CounterfactualHealthyGenerator"]


@register_model(
    name="counterfactual_healthy",
    training_mode="reconstruction",
    supports_contrast_conditioning=True,
)
class CounterfactualHealthyGenerator(nn.Module):
    r"""Wrap a healthy-trained generator to produce counterfactual + saliency.

    Args:
        base_model_name: Name of a generator registered with
            ``supports_contrast_conditioning=True``. Defaults to a
            latent-diffusion-style backbone but works with any
            contrast-capable model.
        base_model_kwargs: Keyword args forwarded to the base generator.
        healthy_index: Integer contrast id treated as the "healthy" axis
            of the base generator. The wrapper uses this id at decode
            time to project pathological inputs onto the healthy prior.
        in_channels / out_channels: Forwarded to the base model and
            reported to the audit ladder.
    """

    def __init__(
        self,
        base_model_name: str = "geo_mamba_unet",
        base_model_kwargs: dict[str, Any] | None = None,
        healthy_index: int = 0,
        in_channels: int = 1,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        capable = list_models_with_capability("supports_contrast_conditioning")
        if base_model_name == "counterfactual_healthy":
            raise ValueError(
                "CounterfactualHealthyGenerator cannot wrap itself — pick a "
                f"contrast-capable backbone from {sorted(capable)}."
            )
        if base_model_name not in capable:
            raise ValueError(
                f"base_model_name {base_model_name!r} is not "
                f"contrast-capable. Pick one of {sorted(capable)}."
            )
        if healthy_index < 0:
            raise ValueError(f"healthy_index must be >= 0, got {healthy_index}")
        kwargs = dict(base_model_kwargs or {})
        kwargs.setdefault("in_channels", in_channels)
        kwargs.setdefault("out_channels", out_channels)
        cls = get_model_class(base_model_name)
        self.base = cls(**kwargs)
        self.base_name = base_model_name
        self.healthy_index = healthy_index
        self.in_channels = in_channels
        self.out_channels = out_channels

    def _decode_healthy(
        self,
        x: torch.Tensor,
        acquisition_params: dict[str, Any] | None,
    ) -> torch.Tensor:
        """Run the base model with the healthy contrast id stamped onto every sample."""
        B = x.shape[0]
        healthy_idx = torch.full((B,), self.healthy_index, dtype=torch.long, device=x.device)
        try:
            out = self.base(x, acquisition_params=acquisition_params, contrast_idx=healthy_idx)
        except TypeError:
            # Some base models accept (x, contrast_idx=...) without
            # acquisition_params — fall back gracefully.
            out = self.base(x, contrast_idx=healthy_idx)
        # Some base models (e.g. cross_contrast_cross_domain) return a dict
        # of multiple heads. Pick the canonical image head.
        if isinstance(out, dict):
            for k in ("image_target", "image_source", "image", "output"):
                if k in out:
                    return out[k]
            # Fall back to the first tensor entry.
            for v in out.values():
                if isinstance(v, torch.Tensor):
                    return v
            raise RuntimeError(f"base model {self.base_name!r} returned a dict with no image key.")
        return out

    def forward(
        self,
        x: torch.Tensor,
        acquisition_params: dict[str, Any] | None = None,
        contrast_idx: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x_h = self._decode_healthy(x, acquisition_params)
        # Channel-mean absolute difference → (B, 1, H, W) saliency.
        diff = (x - x_h).abs()
        if diff.shape[1] > 1:
            saliency = diff.mean(dim=1, keepdim=True)
        else:
            saliency = diff
        return {
            "counterfactual": x_h,
            "saliency": saliency,
            "reconstruction": x,
        }
