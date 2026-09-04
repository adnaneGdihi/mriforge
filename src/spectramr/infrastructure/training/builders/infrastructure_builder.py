"""Infrastructure Builder

Creates metrics and ancillary training infrastructure components.
"""

from __future__ import annotations

import logging
from typing import Any

from spectramr.config.settings import TrainingSettings
from spectramr.core.metrics import MetricsRegistry, get_metric
from spectramr.core.metrics.flag_map import FLAG_TO_METRIC
from spectramr.domain.exceptions import ConfigurationError

from .base import Builder

logger = logging.getLogger(__name__)


class InfrastructureBuilder(Builder):
    """Builds metrics and related infrastructure from configuration."""

    def __init__(self, config: TrainingSettings, device: Any):
        """__init__.

        Args:
            config (TrainingSettings): Description.
            device (Any): Description.
        """
        self._config = config
        self._device = device
        self._metrics: dict[str, Any] = {}

    def build_metrics(self) -> InfrastructureBuilder:
        """Create metrics according to config.metrics flags."""
        metrics_cfg = self._config.metrics

        flags = {
            "compute_psnr": True,
            "compute_ssim": True,
            "compute_mse": True,
            "compute_mae": False,
        }

        if metrics_cfg is not None:
            if hasattr(metrics_cfg, "model_dump"):
                flags.update(metrics_cfg.model_dump())
            elif hasattr(metrics_cfg, "dict"):
                flags.update(metrics_cfg.dict())
            else:
                flags.update(vars(metrics_cfg))

        # metric_map is the SSOT flag -> registered-metric-name mapping for the
        # per-batch builder path (spectramr.core.metrics.flag_map). Coverage here is
        # deliberately a subset of the schema — offline / report / distribution
        # metrics are computed elsewhere; see KNOWN_SCHEMA_FLAGS_WITHOUT_MAP_ENTRY.
        metric_map = FLAG_TO_METRIC

        for flag, name in metric_map.items():
            enabled = flags.get(flag, False)
            if not enabled:
                continue

            if not MetricsRegistry.is_registered(name):
                # Pitfall #9/#18 — an explicitly-enabled metric that names no
                # registered metric would silently never be computed (and could
                # be the early-stopping / headline metric). Every real metric is
                # registered by import-time decorator regardless of optional
                # deps, so a miss here is a genuine config error: raise.
                raise ConfigurationError(
                    f"metrics.{flag} is enabled but '{name}' is not a registered "
                    f"metric — it would never be computed. Check the metric name."
                )

            try:
                self._metrics[name] = get_metric(name, device=self._device)
                logger.info("Metric created: %s", name)
            except Exception as exc:
                # Construction failure (e.g. an optional dependency like
                # pyradiomics/torchvision missing on this host) — the metric
                # exists but can't run here. Surface loudly rather than at DEBUG.
                logger.warning("Failed to create metric %s: %s", name, exc)

        return self

    def validate(self) -> InfrastructureBuilder:
        """validate.

        Returns:
            InfrastructureBuilder: Description.
        """
        if not self._metrics:
            logger.warning("No metrics created; training will proceed without metrics")
        return self

    def build(self) -> dict[str, Any]:
        """build.

        Returns:
            dict[str, Any]: Description.
        """
        return dict(self._metrics)
