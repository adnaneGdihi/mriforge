"""Base contract for prior-method baseline adapters.

See ``TODO/backlog_baseline_replication_experiment_11.md`` for the
replication design. The base class is **deliberately thin**: it
captures only what every adapter must declare for provenance,
canonical-layout interop, and metric comparability — it does NOT
inherit from any specific paradigm class. Each adapter picks its
own forward / sampling implementation while routing loss + mask
generation through this repo's existing Strategy and physics SSOT.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import ClassVar

import torch
import torch.nn as nn


class FFTNorm(StrEnum):
    """FFT normalisation conventions.

    This repo's canonical convention is ``"ortho"`` (orthonormal,
    centered). Adapters whose upstream uses a different convention
    must declare it here so a shim can be wrapped at the adapter
    boundary; never reimplement the normalisation inside the model.
    """

    ORTHO = "ortho"
    BACKWARD = "backward"  # numpy / scipy default
    FORWARD = "forward"


class CoilHandling(StrEnum):
    """Coil-combination strategies expected by the adapter."""

    RSS = "rss"
    MULTI_COIL_KSPACE = "multi_coil_kspace"
    SENSE_COMBINED = "sense_combined"


class BaselineAdapter(nn.Module, ABC):
    """Base contract for adapters that wrap an upstream baseline.

    Subclass requirements:

    1. Declare the class attributes ``REPO_NAME``, ``PAPER_REF``,
       ``PREFERRED_MASK_TYPE``, ``PREFERRED_FFT_NORM``, ``COIL_HANDLING``.
       The base class refuses to instantiate a subclass that leaves any
       of them at the sentinel value ``"<MUST_OVERRIDE>"`` — that is the
       CLAUDE.md pitfall #9 guard against silent fallbacks.
    2. Implement :meth:`forward` returning a tensor in this repo's
       canonical layout: ``[B, C, H, W]`` with complex dtype,
       ``fft2c``-centred when in k-space.
    3. Optionally override :meth:`provenance` to return a dict that
       extends the default with adapter-specific keys (e.g., upstream
       commit SHA — collected by ``baseline_provenance`` in
       ``src/infrastructure/reporting/metadata.py``).
    """

    REPO_NAME: ClassVar[str] = "<MUST_OVERRIDE>"
    PAPER_REF: ClassVar[str] = "<MUST_OVERRIDE>"
    PREFERRED_MASK_TYPE: ClassVar[str] = "<MUST_OVERRIDE>"
    PREFERRED_FFT_NORM: ClassVar[FFTNorm] = FFTNorm.ORTHO
    COIL_HANDLING: ClassVar[CoilHandling] = CoilHandling.RSS

    _REQUIRED_OVERRIDES: ClassVar[tuple[str, ...]] = (
        "REPO_NAME",
        "PAPER_REF",
        "PREFERRED_MASK_TYPE",
    )

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # Abstract subclasses can leave the sentinels; only concrete
        # adapters must override. ``ABC`` flips the abstract bit when
        # ``@abstractmethod`` decorators remain unimplemented.
        if getattr(cls, "__abstractmethods__", frozenset()):
            return
        missing = [
            name for name in cls._REQUIRED_OVERRIDES if getattr(cls, name) == "<MUST_OVERRIDE>"
        ]
        if missing:
            raise TypeError(
                f"BaselineAdapter subclass {cls.__name__} must override class "
                f"attribute(s) {missing}. Silent fallbacks are forbidden."
            )

    @abstractmethod
    def forward(self, x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor in canonical layout ``[B, C, H, W]``,
                complex dtype if k-space.

        Returns:
            Output tensor in canonical layout.
        """
        raise NotImplementedError

    def provenance(self) -> dict[str, str]:
        """Return adapter provenance metadata for the run summary.

        Override to add upstream-commit SHA or other adapter-specific
        keys. The base implementation returns the declared class-level
        attributes.
        """
        return {
            "repo_name": self.REPO_NAME,
            "paper_ref": self.PAPER_REF,
            "preferred_mask_type": self.PREFERRED_MASK_TYPE,
            "preferred_fft_norm": self.PREFERRED_FFT_NORM.value,
            "coil_handling": self.COIL_HANDLING.value,
        }
