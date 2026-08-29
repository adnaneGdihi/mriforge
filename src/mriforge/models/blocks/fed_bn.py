r"""FedBN — federated BatchNorm with site-local statistics (idea 8 §8.2).

Implements the FedBN block from Li et al. *ICLR* 2021. In FL with
non-IID sites, a vanilla ``BatchNorm`` aggregated across sites
contaminates the per-site running statistics, hurting convergence.
**FedBN** keeps the BatchNorm parameters *strictly local* — they are
never shared in the aggregation round.

Implementation
--------------
We register the affine + running-stat tensors as
``persistent=False`` buffers on the module. A federated aggregator
that filters by ``persistent=True`` (the standard ``state_dict()``
behaviour with ``keep_vars=False``) will skip these tensors, which
is the exact semantic of FedBN.

The block is otherwise a drop-in replacement for ``nn.BatchNorm2d``
(or ``nn.BatchNorm1d`` / ``BatchNorm3d`` via the ``ndim`` argument).

References
----------
Li X., Jiang M., Zhang X., Kamp M., Dou Q. (2021). *FedBN: Federated
Learning on Non-IID Features via Local Batch Normalization*. ICLR.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FedBN(nn.Module):
    r"""BatchNorm with strictly site-local affine + running statistics.

    Args:
        num_features: Channel count ``C``.
        eps: Numerical floor on the batch variance.
        momentum: EMA factor for running statistics.
        ndim: Spatial rank — ``2`` for ``[B, C, H, W]`` (the most
            common case), ``1`` for ``[B, C, T]``, ``3`` for ``[B, C, D, H, W]``.

    All affine + running-statistic tensors are registered as
    ``persistent=False`` buffers. They are *not* included in the
    standard ``state_dict()`` — federated aggregators that average
    ``state_dict()`` therefore leave them untouched, which is the
    FedBN semantic.

    Trainability of γ, β: they are real :class:`nn.Parameter` objects
    (so backprop optimises them), but they are still excluded from
    federated aggregation by setting ``_local_only`` on the module —
    the aggregator code consults this attribute (or filters by name).
    """

    _local_only: bool = True

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        ndim: int = 2,
    ) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError(f"num_features must be > 0; got {num_features}")
        if not 0.0 < momentum <= 1.0:
            raise ValueError(f"momentum must lie in (0, 1]; got {momentum}")
        if ndim not in (1, 2, 3):
            raise ValueError(f"ndim must be 1, 2, or 3; got {ndim}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0; got {eps}")

        self.num_features = num_features
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.ndim = ndim

        # γ, β as parameters (trainable per-site, but flagged local-only).
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        # Running mean / var as *non-persistent* buffers — excluded from
        # state_dict() so federated aggregation never touches them.
        self.register_buffer("running_mean", torch.zeros(num_features), persistent=False)
        self.register_buffer("running_var", torch.ones(num_features), persistent=False)
        self.register_buffer(
            "num_batches_tracked",
            torch.zeros(1, dtype=torch.long),
            persistent=False,
        )

    @property
    def is_local_only(self) -> bool:
        """``True`` to mark this layer's params as never-aggregated."""
        return self._local_only

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != self.ndim + 2:
            raise ValueError(
                f"FedBN expected a {self.ndim + 2}-D input "
                f"(spatial rank {self.ndim}); got {x.dim()}-D"
            )
        if x.shape[1] != self.num_features:
            raise ValueError(f"channel mismatch: expected C={self.num_features}, got {x.shape[1]}")
        return F.batch_norm(
            x,
            running_mean=self.running_mean,
            running_var=self.running_var,
            weight=self.weight,
            bias=self.bias,
            training=self.training,
            momentum=self.momentum,
            eps=self.eps,
        )


__all__ = ["FedBN"]
