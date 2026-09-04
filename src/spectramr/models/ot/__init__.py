"""Optimal Transport Module.

Provides OT algorithms for unpaired image translation in MRI.

The OT-based **loss classes** (``SinkhornDistance``, ``DynamicOTFlow``,
``KIDOTLoss``) live under :mod:`spectramr.models.losses.optimal_transport_losses`
per CLAUDE.md pitfall #12. The re-export from this package was dropped
2026-05-14 because it created a circular import — losses depend on
``VelocityFieldNetwork`` here, and this ``__init__`` re-exporting the
losses formed a cycle. Callers must import losses from the canonical
home directly::

    from spectramr.models.losses.optimal_transport_losses import (
        SinkhornDistance, DynamicOTFlow, KIDOTLoss,
    )

The OT **primitives** (``sinkhorn_knopp`` math, ``VelocityFieldNetwork``)
stay here — they're algorithms / nn.Modules, not losses.
"""

from spectramr.models.ot.optimal_transport import (
    VelocityFieldNetwork,
    entropic_gromov_wasserstein,
    gromov_wasserstein_feature_loss,
    sinkhorn_knopp,
)

__all__ = [
    "VelocityFieldNetwork",
    "entropic_gromov_wasserstein",
    "gromov_wasserstein_feature_loss",
    "sinkhorn_knopp",
]
