"""Loss CSV logging configuration schema.

Three fields are consumed:

- ``frequency`` (``execution_engine.py``): log losses every N steps.
- ``output_dir`` (``pinn_strategy.py``, ``hpo.py``): directory for per-step loss CSV files.
- ``csv_path`` (``hpo.py``, ``paired_arms_audit.py``): the loss CSV itself.

``csv_path`` was dropped in the 2026-05 sweep as "never wired", but it has both a
producer and a consumer: ``hpo.py:438`` writes it into every trial YAML and
``paired_arms_audit.py:111`` audits it. Under ``extra='ignore'`` the write was
silently discarded on load, so the audit read a key that could not survive the
trip -- producer and consumer, no transport (issue #795). 323 arms declare it.

The remaining fields removed in that sweep (include_metrics, buffer_size,
flush_interval, compute_psnr/ssim/mae/mse/fid/lpips/hfen, metric_interval,
domain, transform, enabled, enable_tracking) genuinely have no reader and stay
out: declaring an unread knob is pitfall #15. ``extra='ignore'`` keeps the ~417
arms that still set ``enabled`` loadable; that corpus cleanup is tracked
separately.
"""

from pydantic import BaseModel, Field


class LossLoggingConfigSchema(BaseModel):
    """Loss CSV logging — frequency and output directory."""

    model_config = {
        "protected_namespaces": (),
        "extra": "ignore",
        "frozen": True,
    }

    frequency: int = Field(
        default=1,
        ge=1,
        description="Log losses every N steps",
    )
    output_dir: str | None = Field(
        default=None,
        description="Directory for per-step loss CSV files (used by HPO and PINN strategy)",
    )
    csv_path: str | None = Field(
        default=None,
        description="Per-step loss CSV file (written by hpo.py, audited by paired_arms_audit.py)",
    )


__all__ = ["LossLoggingConfigSchema"]
