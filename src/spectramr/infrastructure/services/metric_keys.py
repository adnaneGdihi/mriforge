"""Validation-metric key alias resolution (shared SSOT).

A YAML's monitored metric name frequently does not match the exact key the
validator emits (``val_`` prefix, loss aliases, cascade suffixes). Both the
early-stopping path (:mod:`spectramr.pipelines.train`) and the loss-schedule
controller (:mod:`spectramr.infrastructure.training.loss_schedule_controller`)
resolve a configured key against the live ``val_metrics`` dict through this one
ordered-candidate helper, so neither re-implements the alias rules. See the
``F-EARLYSTOP-*`` entries in ``docs/smoke_audit_20260521_fixes.rst``.

This module lives in the infrastructure layer (not ``pipelines/``) precisely so
the controller can import it without a leftward layer violation (CLAUDE.md #13).
``spectramr.pipelines.train`` re-exports :func:`early_stop_monitor_candidates`
for back-compat with existing call sites and tests.
"""

from __future__ import annotations

from collections.abc import Iterable

_CASCADE_SUFFIXES = ("_2x", "_8x", "_32x")


def early_stop_monitor_candidates(monitor_key: str) -> list[str]:
    """Ordered candidate ``val_metrics`` keys for a monitored metric.

    A YAML's monitor metric frequently does not match the exact key the
    validator emits. Rather than rewrite every YAML, callers try an ordered
    list of aliases (exact match always wins, preserving user intent). This
    helper centralises the four mismatch classes that cluster smoke runs have
    surfaced -- see the ``F-EARLYSTOP-*`` entries in
    ``docs/smoke_audit_20260521_fixes.rst``:

    * **prefix** -- YAMLs use the ``val_`` namespace; the validator may
      return the bare metric (``psnr``).
    * **loss-alias** -- ``monitor: val_loss`` against a validator emitting
      ``val_recon_loss`` / ``g_total_loss`` / (recon-only) only image
      metrics, where ``val_mse`` is the min-mode-compatible proxy.
    * **cascade-add** -- cascading-validation arms suffix the metric with
      the acceleration level (``val_psnr_2x``); bare monitors must gain it.
    * **cascade-strip** -- the inverse: a monitor carries ``_2x`` while a
      non-cascading validator emits the bare key.

    Args:
        monitor_key: The configured metric name.

    Returns:
        Ordered, de-duplicated candidate keys (exact monitor first).
    """
    candidates: list[str] = [monitor_key]

    def _add(key: str | None) -> None:
        if key and key not in candidates:
            candidates.append(key)

    if monitor_key.startswith("val_"):
        _add(monitor_key.removeprefix("val_"))
    if monitor_key in ("val_loss", "loss"):
        for alias in (
            "val_recon_loss",
            "recon_loss",
            "val_total_loss",
            "total_loss",
            "g_total_loss",
            "val_g_total_loss",
            # val_mse: min-mode proxy for recon-only validators that emit
            # no aggregate loss (lower is better, same direction as a loss).
            "val_mse",
            "mse",
        ):
            _add(alias)
    # cascade-add: bare monitor -> suffixed metric (prefer easiest level).
    for suffix in _CASCADE_SUFFIXES:
        bare = monitor_key.removeprefix("val_") if monitor_key.startswith("val_") else None
        _add(f"{monitor_key}{suffix}")
        if bare is not None:
            _add(f"{bare}{suffix}")
    # cascade-strip: suffixed monitor -> bare metric.
    for suffix in _CASCADE_SUFFIXES:
        if monitor_key.endswith(suffix):
            stripped = monitor_key[: -len(suffix)]
            _add(stripped)
            if stripped.startswith("val_"):
                _add(stripped.removeprefix("val_"))
    return candidates


def resolve_metric_key(monitor_key: str, available: Iterable[str] | None) -> str | None:
    """Return the first alias of ``monitor_key`` present in ``available``.

    Mirrors the early-stopping resolution so a configured ``val_ssim`` matches
    a validator that emits the bare ``ssim``. Returns ``None`` when no candidate
    is present (the caller surfaces that as a once-per-rule warning rather than
    a silent never-fire -- CLAUDE.md #10 / #15).

    Args:
        monitor_key: The configured metric name.
        available: The keys actually present in the metrics dict.

    Returns:
        The resolved key, or ``None`` if none of the candidates is available.
    """
    if not available:
        return None
    avail = set(available)
    for cand in early_stop_monitor_candidates(monitor_key):
        if cand in avail:
            return cand
    return None


__all__ = ["early_stop_monitor_candidates", "resolve_metric_key"]
