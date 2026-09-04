r"""Galois-cohomological defensibility certificate for phase unwrapping.

Exposes the :math:`\\|[w]\\|_{H^1}` certificate from
:mod:`spectramr.infrastructure.physics.galois_unwrap` as a registered metric. The
metric is *lower-is-better* and equals zero iff the wrapped phase admits a
globally consistent lift on the cubic-grid Čech cover.

See ``docs/galois_unwrap.rst`` for the mathematical statement.
"""

from __future__ import annotations

import torch

from spectramr.core.metrics.registry import register_metric
from spectramr.infrastructure.physics.galois_unwrap import cohomology_norm_certificate


@register_metric(
    name="galois_h1_certificate",
    aliases=["h1_cert", "cohomology_certificate"],
    requires_reference=False,
)
class GaloisH1Certificate:
    r"""Scalar :math:`\|[w]\|_{H^1(\Omega,\underline{\mathbb{Z}})}` defensibility certificate.

    Zero iff the unwrapping is globally consistent. The metric ignores the
    target argument — it is a property of the *prediction* (the wrapped phase)
    alone.
    """

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        **_: object,
    ) -> float:
        cert = cohomology_norm_certificate(prediction)
        return float(cert.detach().cpu().item())

    @property
    def name(self) -> str:
        return "galois_h1_certificate"

    @property
    def higher_is_better(self) -> bool:
        return False
