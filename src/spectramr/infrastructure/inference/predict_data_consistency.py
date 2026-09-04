"""Test-time hard data-consistency projection for the predict verb.

Keyed on ``physics.data_consistency.apply_at_predict``. The training-time DC
layer is model-integrated (``StrategyInitializationHelper.initialize_data_consistency``
reuses ``generator.dc_layer``), so an arm that declares ``enabled: true`` and
trains a plain reconstruction network gets no projection at predict: the
reconstruction and GAN inference strategies run one forward pass and stop. The
samplers that do project (diffusion under ``method: hard``, cold diffusion with
a mask, physics-driven) do it inside their own loops.

This module is the one owner of the projection the predict verb adds. The hook
that calls it lives in :meth:`BaseInferenceStrategy.infer`, after
``run_inference`` and before ``postprocess_output``, so the projection sees the
model's output scale rather than a clamped copy. It needs three tensors
(prediction, measurement, mask), which is why it is not an adapter.

The measurement is used as acquired: ``eval_noise_level`` simulates acquisition
noise on a synthesized reference at validation, and at predict there is no
reference to corrupt, so both noise levels are zero here.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch

from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.physics.data_consistency import HardDataConsistency
from spectramr.models.registry import MODEL_REGISTRY, get_model_capabilities

logger = logging.getLogger(__name__)

KNOB = "physics.data_consistency.apply_at_predict"

#: Registered ``output_domain`` -> whether the projection runs in k-space.
#: ``complex_image`` is refused on purpose: ``HardDataConsistency`` splits a real
#: even-channel tensor as interleaved (real0, imag0, ...) while the repo packs
#: stacked halves, and the layer reads any real even-channel tensor as k-space
#: regardless of ``is_kspace_domain``. Neither can be projected honestly here.
_DOMAIN_IS_KSPACE: dict[str, bool] = {"kspace": True, "image": False}


class PredictDataConsistency:
    """The resolved knob, the layer it builds, and the ledger of what fired.

    Construct through :meth:`from_config`, which returns ``None`` when the knob
    is off so a strategy carries no projection state at all in that case.
    """

    def __init__(self, *, output_domain: str, noise_type: str, model_type: str) -> None:
        self.output_domain = output_domain
        self.is_kspace_domain = _DOMAIN_IS_KSPACE[output_domain]
        self.model_type = model_type
        # Both noise levels zero: the measurement is the acquisition (see module
        # docstring). ``noise_type`` still goes through the layer's validation so
        # an unsupported value raises here as it does at training.
        self._layer = HardDataConsistency(
            train_noise_level=0.0, eval_noise_level=0.0, noise_type=noise_type
        ).eval()
        # Per-call ledger (reset by ``begin``) and run-level counters.
        self._applied_by_this_call: list[str] = []
        self.calls = 0
        self.projections = 0
        self.skipped_already_applied = 0
        self.applied_by: dict[str, int] = {}

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> PredictDataConsistency | None:
        """Resolve the knob from the strategy config (``settings.model_dump()``).

        Returns:
            ``None`` when the knob is off or absent.

        Raises:
            ConfigurationError: On a non-boolean value, an unsupported
                ``noise_type``, or a model whose ``output_domain`` is not
                registered as ``kspace`` or ``image``.
        """
        if not isinstance(config, Mapping):
            return None
        physics = config.get("physics") or {}
        dc = physics.get("data_consistency") or {} if isinstance(physics, Mapping) else {}
        flag = dc.get("apply_at_predict", False) if isinstance(dc, Mapping) else False
        if not isinstance(flag, bool):
            raise ConfigurationError(f"{KNOB} must be a boolean, got {flag!r}.")
        if not flag:
            return None

        model_type = str((config.get("model") or {}).get("model_type") or "")
        output_domain = _resolve_output_domain(model_type)
        try:
            resolved = cls(
                output_domain=output_domain,
                noise_type=str(dc.get("noise_type", "gaussian")),
                model_type=model_type,
            )
        except ValueError as exc:  # the layer's noise_type policy, retyped
            raise ConfigurationError(f"{KNOB}: {exc}") from exc
        logger.info(
            "[PREDICT-DC] %s=true: predictions of %r (output_domain=%r) will be projected "
            "onto the measured k-space in the %s domain, measurement used as acquired.",
            KNOB,
            model_type,
            output_domain,
            "k-space" if resolved.is_kspace_domain else "image",
        )
        return resolved

    def begin(self) -> None:
        self._applied_by_this_call = []
        self.calls += 1

    def note_applied(self, by: str) -> None:
        """A strategy's own loop pinned the measurement for this call."""
        self._applied_by_this_call.append(by)
        self.applied_by[by] = self.applied_by.get(by, 0) + 1

    @property
    def applied_this_call(self) -> bool:
        return bool(self._applied_by_this_call)

    def provenance(self) -> dict[str, Any]:
        """What a run can honestly claim about DC at predict."""
        return {
            "apply_at_predict": True,
            "domain": self.output_domain,
            "model_type": self.model_type,
            "calls": self.calls,
            "projections_by_predict_step": self.projections,
            "skipped_already_applied": self.skipped_already_applied,
            "applied_by": dict(self.applied_by),
            "measurement_noise_added": False,
        }

    def finalize(
        self,
        prediction: torch.Tensor,
        *,
        mask: torch.Tensor | None,
        measured_kspace: torch.Tensor | None,
        strategy: str,
    ) -> torch.Tensor:
        """Project unless this call's sampler already pinned the measurement."""
        if self.applied_this_call:
            self.skipped_already_applied += 1
            logger.info(
                "[PREDICT-DC] %s: measurement already pinned by %s; not projecting twice.",
                strategy,
                ", ".join(self._applied_by_this_call),
            )
            return prediction
        out = self.project(prediction, mask=mask, measured_kspace=measured_kspace)
        self.note_applied("predict_step")
        return out

    def project(
        self,
        prediction: torch.Tensor,
        *,
        mask: torch.Tensor | None,
        measured_kspace: torch.Tensor | None,
    ) -> torch.Tensor:
        """Hard projection: sampled bins take the measurement, the rest keep the prediction.

        Raises:
            ConfigurationError: When the batch carries no mask or no measurement,
                or their layouts cannot be projected without guessing.
        """
        pairs = (("mask", mask), ("measured_kspace", measured_kspace))
        if missing := [n for n, t in pairs if t is None]:
            raise ConfigurationError(
                f"{KNOB} is true but this batch carries no {' and no '.join(missing)}. The "
                "predict verb reads the mask from the 'mask' dataset of an HDF5 input and "
                "takes the k-space input as the measurement (data.dataset_type must serve "
                "k-space). Supply them or turn the knob off."
            )
        assert mask is not None and measured_kspace is not None
        measured = measured_kspace.to(prediction.device)
        plane = _as_plane_mask(mask, prediction)
        self._check_measurement(prediction, measured, plane)
        out = self._layer(
            prediction,
            kspace_obs=measured,
            mask=plane,
            is_kspace_domain=self.is_kspace_domain,
        )
        self.projections += 1
        return out

    def _check_measurement(
        self, prediction: torch.Tensor, measured: torch.Tensor, plane: torch.Tensor
    ) -> None:
        """Refuse the layouts the layer would silently truncate or recast."""
        if self.is_kspace_domain:
            same = prediction.shape == measured.shape
            if not same or prediction.is_complex() != measured.is_complex():
                raise ConfigurationError(
                    f"{KNOB}: a k-space prediction and its measurement must share shape and "
                    f"dtype family; got {tuple(prediction.shape)} ({prediction.dtype}) vs "
                    f"{tuple(measured.shape)} ({measured.dtype})."
                )
            k_channels = prediction.shape[1]
        else:
            if not prediction.is_complex() and prediction.shape[1] % 2 == 0:
                raise ConfigurationError(
                    f"{KNOB}: {self.model_type!r} is an image model but its prediction has "
                    f"{prediction.shape[1]} real channels, which HardDataConsistency reads as "
                    "k-space regardless of the declared domain; only a complex image or a "
                    "real image with an odd channel count (a magnitude) can be projected."
                )
            k_channels = prediction.shape[1]
            if measured.is_complex():
                m_channels = measured.shape[1]
            elif measured.shape[1] == 2:
                m_channels = 1
            else:
                raise ConfigurationError(
                    f"{KNOB}: a real measurement must be one coil as (B, 2, H, W); got "
                    f"{tuple(measured.shape)}. The layer's real/imag split is interleaved and "
                    "this repo packs stacked halves, so supply multi-coil data as complex."
                )
            if m_channels != k_channels or measured.shape[-2:] != prediction.shape[-2:]:
                raise ConfigurationError(
                    f"{KNOB}: prediction {tuple(prediction.shape)} implies {k_channels} "
                    f"k-space channel(s) but the measurement {tuple(measured.shape)} carries "
                    f"{m_channels}; the layer would truncate silently."
                )
        if plane.shape[1] not in (1, k_channels):
            raise ConfigurationError(
                f"{KNOB}: mask has {plane.shape[1]} channel(s); expected 1 or {k_channels}."
            )


def _resolve_output_domain(model_type: str) -> str:
    """The projection domain, from the model's registered capabilities."""
    caps = get_model_capabilities(model_type) if model_type else None
    domain = getattr(caps, "output_domain", None)
    if isinstance(domain, str) and domain in _DOMAIN_IS_KSPACE:
        return domain
    if model_type and model_type not in MODEL_REGISTRY:
        state = "is not in MODEL_REGISTRY (populate_model_registry() not run, or unknown name)"
    elif domain is None:
        state = "declares no output_domain on its registration"
    else:
        state = f"declares output_domain={domain!r}"
    raise ConfigurationError(
        f"{KNOB} needs the model's registered output_domain to pick the projection domain, "
        f"and {model_type!r} {state}. Supported: {sorted(_DOMAIN_IS_KSPACE)} ('complex_image' "
        "is refused: the DC layer's real/imag split disagrees with this repo's stacked-halves "
        "packing). Annotate output_domain=... on @register_model, or turn the knob off."
    )


def _as_plane_mask(mask: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """Normalize a sampling mask to ``(N, C, H, W)`` float on the prediction's device.

    Accepted layouts: ``(H, W)``, ``(N, H, W)``, ``(N, C, H, W)`` with ``N`` in
    ``{1, B}``. A 1-D mask is refused rather than expanded: which axis is
    phase-encode is not guessed here.
    """
    if mask.is_complex():
        raise ConfigurationError(f"{KNOB}: the mask must be real-valued, got {mask.dtype}.")
    if mask.ndim == 2:
        plane = mask[None, None]
    elif mask.ndim == 3:
        plane = mask[:, None]
    elif mask.ndim == 4:
        plane = mask
    else:
        raise ConfigurationError(
            f"{KNOB}: mask of shape {tuple(mask.shape)} is not (H, W), (N, H, W) or "
            "(N, C, H, W). A 1-D line mask must be expanded to the plane by the producer."
        )
    if plane.shape[-2:] != prediction.shape[-2:]:
        raise ConfigurationError(
            f"{KNOB}: mask plane {tuple(plane.shape[-2:])} does not match the prediction's "
            f"{tuple(prediction.shape[-2:])}."
        )
    if plane.shape[0] not in (1, prediction.shape[0]):
        raise ConfigurationError(
            f"{KNOB}: mask has {plane.shape[0]} leading entries; expected 1 or the batch "
            f"size {prediction.shape[0]}."
        )
    plane = plane.to(device=prediction.device, dtype=torch.float32)
    if not bool(torch.all((plane == 0) | (plane == 1))):
        raise ConfigurationError(
            f"{KNOB}: the mask must be binary; a soft mask would turn hard DC into a blend."
        )
    return plane
