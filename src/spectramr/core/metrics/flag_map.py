"""Single source of truth for the ``metrics.compute_*`` flag -> metric-name mapping.

Three separate consumers historically hand-maintained their own ``compute_* -> name``
dict, and they drifted:

- ``InfrastructureBuilder.build_metrics`` — per-batch metric *construction*
  (fail-loud on an unregistered name).
- ``MetricsMixin._extract_metrics_from_config`` — the *name list* fed to the
  validation metrics computer.
- ``pipelines.training_loop`` — the *expected* ``train_``/``val_`` metric-name set
  used for column tracking.

The flag->name relationship is *identity* in every case (``compute_psnr`` -> ``psnr``)
with exactly one near-miss exception where identity names an UNREGISTERED metric:
``compute_neg_voxels`` -> the registered metric is ``negative_voxels``. (The former
``compute_spike_percent`` — a duplicate of ``compute_spike_percentage`` whose identity
target ``spike_percent`` was never registered — was removed from the schema in the same
change, so it needs no alias.)

:func:`metric_for_flag` is the ONE resolver. Each consumer keeps its own *coverage*
policy (which flags it honors — the per-batch builder deliberately excludes offline /
report / distribution metrics such as FID), but resolves the NAME through this function.
``tests/unit/core/metrics/test_flag_map.py`` locks all three consumer maps to it, so the
same flag can never again map to two different names (fatal on one path, silent skip on
another). This mirrors the loss-weight SSOT (``models/losses/weights.py``, pitfall #13b)
one layer over.

Naming was only half the drift, though: the *coverage* halves were hand-maintained too,
at 78 (CSV columns) and 43 (mixin selection) entries, and disagreed on **22 flags that
resolve to a registered metric**. Each of those declared a ``losses.csv`` column that no
code path could ever fill -- a header with a permanently empty column reads as "we
measured it and it was blank" (#340). :func:`schema_flag_to_metric` derives coverage from
``MetricsConfigSchema`` so a flag added to the schema is reachable by construction, and
neither consumer can silently fall behind it again.
"""

from __future__ import annotations

_COMPUTE_PREFIX = "compute_"

#: ``compute_*`` flags whose registered metric name is NOT the identity strip of the
#: flag. Keep this tiny — every entry is a historical naming near-miss, not a feature.
_FLAG_TO_METRIC_ALIASES: dict[str, str] = {
    "compute_neg_voxels": "negative_voxels",
}


def metric_for_flag(flag: str) -> str:
    """Return the registered metric name for a ``metrics.compute_*`` flag.

    Identity by default (``compute_psnr`` -> ``psnr``); the alias table handles the
    near-miss cases where a plain strip would name a metric that is not registered.
    """
    if not flag.startswith(_COMPUTE_PREFIX):
        raise ValueError(f"not a metrics compute_* flag: {flag!r}")
    return _FLAG_TO_METRIC_ALIASES.get(flag, flag[len(_COMPUTE_PREFIX) :])


#: ``compute_*`` booleans that do NOT select a metric. Legacy master switches whose
#: identity strip names nothing and never will, so a consumer must neither select
#: them nor report them as dangling.
#:
#: Keeping this separate from "names an unregistered metric" is load-bearing.
#: ``compute_advanced_metrics`` defaults **True** (#343), so it is live on every arm
#: whether or not the YAML mentions it. Feeding it to a registry filter would emit a
#: dangling-flag warning on EVERY run -- and warnings exit 2 under ``audit --strict``
#: (non-negotiable #4), which would turn a behaviour-neutral refactor into a
#: corpus-wide break. The other 16 danglers default False and SHOULD stay visible:
#: they name a metric that is simply not registered yet, which is #340's actual ask.
NON_METRIC_FLAGS: frozenset[str] = frozenset({"compute_advanced_metrics"})


def schema_compute_flags() -> frozenset[str]:
    """Every metric-selecting ``compute_*`` boolean declared on the metrics schema.

    Derived from ``MetricsConfigSchema.model_fields`` rather than hand-listed. The
    schema is a safe derivation source in a way a registry is not: ``model_fields``
    is fully determined once the class body executes, whereas a decorator-populated
    registry is import-order dependent.
    """
    from spectramr.config.schemas.metrics import MetricsConfigSchema

    return frozenset(
        f
        for f in MetricsConfigSchema.model_fields
        if f.startswith(_COMPUTE_PREFIX) and f not in NON_METRIC_FLAGS
    )


def schema_flag_to_metric() -> dict[str, str]:
    """``compute_*`` flag -> metric name, for every metric-selecting schema flag.

    The full-coverage map. Consumers that need a narrower policy (the per-batch
    builder excludes offline/report metrics) intersect this with their own flag
    set; they must not re-spell the NAMES, which is how the three maps drifted.
    """
    return {flag: metric_for_flag(flag) for flag in sorted(schema_compute_flags())}


#: Flags the per-batch :class:`InfrastructureBuilder` constructs a metric object for.
#: This is a CURATED SUBSET of the schema: offline / report / distribution metrics
#: (FID, precision/recall, ICC, Bland-Altman, the perfusion/flow/spectroscopy families,
#: ...) are deliberately excluded here and computed elsewhere. The exclusions are
#: catalogued in ``KNOWN_SCHEMA_FLAGS_WITHOUT_MAP_ENTRY``
#: (``tests/unit/config/test_metrics_schema_coverage.py``).
BUILDER_METRIC_FLAGS: frozenset[str] = frozenset(
    {
        # Basic
        "compute_psnr",
        "compute_ssim",
        "compute_mse",
        "compute_mae",
        "compute_nmse",
        "compute_nrmse",
        "compute_rmse",
        "compute_snr",
        # Perceptual / structural
        "compute_hfen",
        "compute_gmsd",
        "compute_fsim",
        "compute_vif",
        "compute_ms_ssim",
        "compute_uqi",
        "compute_gradient_entropy",
        "compute_gradient_error",
        # K-space / physics
        "compute_kspace_error",
        "compute_phase_mse",
        "compute_ipen",
        "compute_complex_hfen",
        # QA / artifacts
        "compute_qi1",
        "compute_efc",
        "compute_fber",
        "compute_cjv",
        "compute_cnr",
        "compute_wm2max",
        # Segmentation
        "compute_dice",
        "compute_iou",
        "compute_hd95",
        # Temporal
        "compute_tsnr",
        "compute_temporal_fidelity",
        # Generative
        "compute_kid",
        "compute_inception_score",
        "compute_lpips",
        # Radiomics
        "compute_frd",
        "compute_rfs",
    }
)

#: ``compute_*`` flag -> registered metric name, for the per-batch builder path.
#: Derived from :func:`metric_for_flag` so the naming has a single owner.
FLAG_TO_METRIC: dict[str, str] = {
    flag: metric_for_flag(flag) for flag in sorted(BUILDER_METRIC_FLAGS)
}


__all__ = [
    "BUILDER_METRIC_FLAGS",
    "FLAG_TO_METRIC",
    "NON_METRIC_FLAGS",
    "metric_for_flag",
    "schema_compute_flags",
    "schema_flag_to_metric",
]
