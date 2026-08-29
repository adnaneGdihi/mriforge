"""Downstream-task-model adapter protocol.

Implements ``CC-3`` of
``TODO/backlog_paradigm_expansion_roadmap.md``. Provides a single
narrow interface that the H3 (downstream-task evaluation), L5
(adversarial-robustness), and Part-I G (BALD active acquisition)
pipelines all consume.

A *downstream* model is any pre-trained classifier / segmenter /
detector that the reconstruction pipeline must hand off its output to
in order to compute task-aware metrics (Dice on segmentation,
sensitivity / specificity on classification, etc.).

Implementations live in ``src/mriforge/infrastructure/services/`` and are
registered into the DI container by ``src/mriforge/bootstrap.py``. Pipeline /
loss code resolves them via ``resolve_service(IDownstreamModelService)``
— never instantiate adapters directly.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class IDownstreamModelService(Protocol):
    """Protocol for a downstream-task-model adapter.

    Implementations wrap a frozen pre-trained model and expose:

    - ``forward`` — run inference on a batch of reconstructions
    - ``compute_metric`` — produce a task-aware scalar metric
    - ``task_name`` — short identifier (e.g. ``"brain_seg"``)
    - ``num_classes`` — for classification / segmentation tasks; ``None``
      for regression / detection

    Adapters should:

    - Run the wrapped model in ``.eval()`` mode and ``torch.no_grad()``
      unless explicitly configured to back-prop (e.g. for BALD).
    - Tolerate complex / 2-channel real / multi-channel reconstructions
      by adapting them to the wrapped model's expected input shape.
    - Fail-loud (no silent fallback) when the input shape doesn't match
      the wrapped model's contract — see CLAUDE.md pitfall #9.
    """

    @property
    def task_name(self) -> str:
        """Short, kebab- or snake-case task identifier."""
        ...

    @property
    def num_classes(self) -> int | None:
        """Class count for classification / segmentation tasks; ``None`` for regression."""
        ...

    def forward(
        self,
        reconstruction: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run the downstream model on a reconstruction batch.

        Args:
            reconstruction: ``[B, C, H, W]`` (or ``[B, C, D, H, W]``)
                reconstruction. Adapter is responsible for any
                channel-shape coercion to fit the wrapped model.
            **kwargs: Adapter-specific keyword arguments.

        Returns:
            Raw model output (logits, segmentation map, regression
            target, …). Shape and semantics are task-specific.
        """
        ...

    def compute_metric(
        self,
        reconstruction: torch.Tensor,
        ground_truth: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Run the downstream model and return scalar task-aware metrics.

        Args:
            reconstruction: As in :py:meth:`forward`.
            ground_truth: Reference image (or label) the downstream
                task uses to compute its score.
            **kwargs: Adapter-specific keyword arguments.

        Returns:
            Mapping ``metric_name → float``. Keys must be stable across
            calls so the reporting pipeline can aggregate them.
        """
        ...
