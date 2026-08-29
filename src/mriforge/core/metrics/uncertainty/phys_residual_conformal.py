r"""PR-CC physics-residual consistency metric (eval-tier handle).

The registered metric ``phys_residual_consistency`` reports the *mean physics
residual* of a qMRI-output model: re-render the observed contrast from the
estimated ``(rho, T1, T2)`` parameter map through the Bloch forward and average
``|render - observed|``. It is a real forward-model consistency scalar (lower is
better) -- not a calibrated certificate.

The full PR-CC certificate (split-conformal coverage + the flagged hallucination
map, with Mondrian field-stratification) is
:class:`mriforge.infrastructure.calibration.phys_residual_certificate.PhysResidualConformalCertificate`,
which lives in the calibration tier because it needs a held-out calibration
split that a per-batch metric cannot hold. This keeps the metric honest about
what it computes (pitfall #18 -- the metric matches its claim).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from mriforge.core.metrics.registry import register_metric


@register_metric("phys_residual_consistency", aliases=["prcc_consistency"])
class PhysResidualConsistencyMetric:
    """Mean forward-model (physics) residual of a qMRI parameter-map prediction.

    Args:
        acquisition: Default sequence parameters ``{"TR","TE","TI"}`` (ms). May
            be overridden per call via ``**kwargs``.
        contrast_type: Default contrast label routed to the Bloch forward.
    """

    # Param-map (3ch) vs observed-image (1ch): a non-standard pair, so the eval
    # loop drives it via explicit (prediction=param_map, target=image) calls.
    INPUT_SIGNATURE = "image_pair"

    def __init__(
        self,
        acquisition: Mapping[str, float] | None = None,
        contrast_type: str = "spin_echo",
    ) -> None:
        self._acquisition = acquisition
        self._contrast_type = contrast_type

    @property
    def name(self) -> str:
        return "phys_residual_consistency"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> float:
        # Lazy import: core/ must not import infrastructure/ at module scope
        # (layer-direction rule; calibration.scores is not physics-SSOT-exempt).
        # A module-level import here poisons the metrics walk-discovery.
        from mriforge.infrastructure.calibration.scores import physics_residual

        acquisition = kwargs.get("acquisition", self._acquisition)
        contrast_type = kwargs.get("contrast_type", self._contrast_type)
        residual = physics_residual(
            prediction,
            target,
            acquisition=acquisition,
            contrast_type=contrast_type,
        )
        return float(residual.mean().item())


__all__ = ["PhysResidualConsistencyMetric"]
