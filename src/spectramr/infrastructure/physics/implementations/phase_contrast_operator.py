"""Phase-contrast forward operator (velocity <-> flow-encoded complex signal).

Registered under ``phase_contrast`` in the physics :class:`OperatorRegistry`,
this is the forward model that backs the ``mri_flow`` regime. ``forward`` maps a
velocity map to the flow-encoded complex signal; ``adjoint`` recovers velocity
from the (wrapped) signal phase. See
:mod:`spectramr.infrastructure.physics.signal_models.flow_encoding`.
"""

from __future__ import annotations

import torch

from spectramr.infrastructure.physics.registry import (
    BaseForwardOperator,
    register_operator,
)
from spectramr.infrastructure.physics.signal_models.flow_encoding import (
    pc_adjoint,
    pc_forward,
)


class PhaseContrastOperator(BaseForwardOperator):
    """PC forward/adjoint at a fixed ``venc``.

    Args:
        venc: Velocity-encoding value (same units as the velocity map).
        img_size: ``(H, W)`` of the velocity map.
    """

    def __init__(self, venc: float = 1.0, img_size: tuple[int, int] = (256, 256)):
        if venc <= 0:
            raise ValueError(f"venc must be positive, got {venc}")
        self.venc = float(venc)
        self._img_size = img_size

    def forward(self, img: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Velocity map -> flow-encoded complex signal."""
        venc = float(kwargs.get("venc", self.venc))  # type: ignore[arg-type]
        magnitude = kwargs.get("magnitude", 1.0)
        return pc_forward(img, venc, magnitude)  # type: ignore[arg-type]

    def adjoint(self, data: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Flow-encoded complex signal -> velocity map."""
        venc = float(kwargs.get("venc", self.venc))  # type: ignore[arg-type]
        return pc_adjoint(data, venc)

    @property
    def img_size(self) -> tuple[int, int]:
        return self._img_size

    def get_operator_type(self) -> str:
        return "phase_contrast"


register_operator(
    name="phase_contrast",
    operator_class=PhaseContrastOperator,
    config={"venc": 1.0},
)


__all__ = ["PhaseContrastOperator"]
