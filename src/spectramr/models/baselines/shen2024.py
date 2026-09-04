"""Shen 2024 adapter — Sci. Reports baseline.

Plan: ``TODO/backlog_baseline_replication_experiment_11.md`` Phase C.

Shen et al. 2024 (Sci. Reports) propose a conditional-diffusion MRI
reconstruction with a learned k-space prior. This adapter wraps the
upstream's published checkpoints under
``external/baselines/shen2024/`` so the network can be evaluated under
this repo's metric harness.

**Status (2026-05-14): blocked on upstream URL.** The Sci. Reports
Data Availability Statement does not yet expose a public repository
or Zenodo DOI. Options pending:

1. Contact corresponding author for the code drop URL.
2. Wait for the DAS to be amended (Sci. Reports' policy typically
   requires reproducibility artefacts within 12 months of publication).
3. Reimplement from the paper's pseudocode (Algorithm 1) — last resort,
   risks a "faithful but slightly off" replication that fails the
   reproducibility gate.

Once the URL is known, the vendoring command is::

    git submodule add <url> external/baselines/shen2024
    git -C external/baselines/shen2024 checkout <release-tag>

Then update :class:`Shen2024Baseline.forward` to wrap the upstream
sampler (mirror :class:`~spectramr.models.baselines.cdiffmr.CDiffMRBaseline`
or :class:`~spectramr.models.baselines.fdb.FDBBaseline` for the shim
pattern — single-step UNet forward + a ``sample()`` method calling
the full reverse-diffusion loop).

Until then, ``forward`` raises :class:`NotImplementedError` with a
pointer to this docstring (CLAUDE.md pitfall #9 — never silently
return wrong outputs).
"""

from __future__ import annotations

import torch

from spectramr.models.baselines._base import (
    BaselineAdapter,
    CoilHandling,
    FFTNorm,
)
from spectramr.models.registry import register_model


@register_model(name="shen2024_baseline", training_mode="cold_diffusion")
class Shen2024Baseline(BaselineAdapter):
    """Shen et al. 2024 — conditional diffusion baseline.

    Class attributes:
        REPO_NAME: Vendored directory name.
        PAPER_REF: ``Shen2024:CondDiffMRI``.
        PREFERRED_MASK_TYPE: ``gaussian_density``; same family as CDiffMR.
        PREFERRED_FFT_NORM: ``ortho``.
        COIL_HANDLING: ``rss`` — paper evaluates on fastMRI knee single-coil.
    """

    REPO_NAME = "shen2024"
    PAPER_REF = "Shen2024:CondDiffMRI"
    PREFERRED_MASK_TYPE = "gaussian_density"
    PREFERRED_FFT_NORM = FFTNorm.ORTHO
    COIL_HANDLING = CoilHandling.RSS

    def __init__(
        self,
        *,
        in_channels: int = 2,
        out_channels: int = 2,
        condition_dim: int = 64,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.condition_dim = condition_dim

    def forward(self, x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        raise NotImplementedError(
            "Shen 2024 adapter not yet wired — upstream URL TBD pending "
            "Sci. Reports DAS amendment. See module docstring for the "
            "three pending options and the vendoring command."
        )


__all__ = ["Shen2024Baseline"]
