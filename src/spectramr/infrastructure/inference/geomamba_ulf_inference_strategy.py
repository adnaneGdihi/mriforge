"""GeoMamba-ULF Inference Strategy.

Companion to
:class:`spectramr.infrastructure.training.strategies.geomamba_ulf_strategy.GeoMambaULFStrategy`
for inference-time deployment. Reproduces the training-time
forward path (optional unwarp → mask → contrast-conditioned
GeoMambaUNet → background re-embed) without the loss / autograd
machinery.

Used by ``python -m spectramr.cli predict --config <yaml> --model
<ckpt> --input <ULF_volume>`` once the v0 strategy has produced
a checkpoint.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

from spectramr.config.schemas.enums import TrainingModeTypes

from .base_inference_strategy import BaseInferenceStrategy


class GeoMambaULFInferenceStrategy(BaseInferenceStrategy):
    """Inference for the GeoMamba-ULF paradigm.

    Args:
        model: a trained
            :class:`spectramr.models.generators.geo_mamba_unet.GeoMambaUNet`
            (or any module exposing the same forward signature
            ``model(volume, mask=, contrast_id=, ...) -> Tensor``).
        device: target device.
        config: inference-time config; recognised keys:

            - ``contrast_id``: int or list[int] — defaults to 0.
            - ``clamp_output``: bool — clamp output to ``[0, 1]``.
            - ``apply_unwarp``: bool — when True and a ``b0_hz``
              kwarg is supplied to ``run_inference``, applies
              :class:`spectramr.infrastructure.physics.unwarp.PhysicalUnwarp`
              before the model forward pass.
            - ``echo_spacing_s``: float — required when ``apply_unwarp`` is True.
    """

    @property
    def training_mode(self) -> TrainingModeTypes:
        return TrainingModeTypes.GEOMAMBA_ULF

    def preprocess_input(self, input_tensor: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
        if input_tensor.device != self.device:
            input_tensor = input_tensor.to(self.device)
        return input_tensor.detach()

    def run_inference(
        self,
        input_tensor: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        if input_tensor.ndim != 5:
            raise ValueError(
                f"GeoMamba-ULF inference expects (B, C, D, H, W); got {tuple(input_tensor.shape)}"
            )

        volume = input_tensor

        # Optional physical unwarp (B0 / Maxwell). Lazily import to
        # avoid the heavy physics module load when unused.
        if self.config.get("apply_unwarp", False):
            volume = self._apply_unwarp(volume, kwargs)

        mask_obj = kwargs.get("mask")
        mask: torch.Tensor | None = None
        if isinstance(mask_obj, torch.Tensor):
            mt = cast(torch.Tensor, mask_obj)
            mask = mt.to(self.device) if mt.device != self.device else mt

        contrast_id = kwargs.get("contrast_id", self.config.get("contrast_id", 0))
        if not isinstance(contrast_id, torch.Tensor):
            contrast_id = torch.tensor(
                [contrast_id] * volume.shape[0]
                if isinstance(contrast_id, int)
                else list(contrast_id),
                dtype=torch.long,
                device=self.device,
            )

        with torch.no_grad():
            output = self.model(
                volume,
                mask=mask,
                contrast_id=contrast_id,
            )
            # Some models return dicts (cf. ReconstructionInferenceStrategy).
            if isinstance(output, dict):
                output = (
                    output.get("reconstruction")
                    or output.get("output")
                    or next(iter(output.values()))
                )
        return output

    def postprocess_output(self, output_tensor: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
        out = output_tensor.detach().cpu()
        if self.config.get("clamp_output", False):
            out = out.clamp(0.0, 1.0)
        return out

    @torch.no_grad()
    def infer_single(self, input_data: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.infer(input_data, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_unwarp(self, volume: torch.Tensor, kwargs: dict[str, Any]) -> torch.Tensor:
        from spectramr.infrastructure.physics.unwarp import PhysicalUnwarp

        echo_spacing_s = self.config.get("echo_spacing_s")
        if echo_spacing_s is None:
            raise ValueError("config['apply_unwarp']=True requires config['echo_spacing_s']")
        unwarp = PhysicalUnwarp(echo_spacing_s=float(echo_spacing_s))
        b0_obj = kwargs.get("b0_hz")
        maxwell_obj = kwargs.get("maxwell_hz")
        gnl_obj = kwargs.get("gnl_displacement")
        b0_hz = (
            cast(torch.Tensor, b0_obj).to(self.device) if isinstance(b0_obj, torch.Tensor) else None
        )
        maxwell_hz = (
            cast(torch.Tensor, maxwell_obj).to(self.device)
            if isinstance(maxwell_obj, torch.Tensor)
            else None
        )
        gnl_displacement = (
            cast(torch.Tensor, gnl_obj).to(self.device)
            if isinstance(gnl_obj, torch.Tensor)
            else None
        )
        return unwarp(
            volume,
            b0_hz=b0_hz,
            maxwell_hz=maxwell_hz,
            gnl_displacement=gnl_displacement,
        )


def _resolve_geomamba_ulf_strategy(
    model: nn.Module,
    device: torch.device,
    config: dict[str, Any] | None = None,
) -> GeoMambaULFInferenceStrategy:
    """Convenience constructor used by the CLI predict path."""
    return GeoMambaULFInferenceStrategy(model=model, device=device, config=config)


__all__ = [
    "GeoMambaULFInferenceStrategy",
    "_resolve_geomamba_ulf_strategy",
]
