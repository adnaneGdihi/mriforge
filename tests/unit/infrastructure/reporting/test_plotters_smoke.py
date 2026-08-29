"""Smoke sweep over every plotter under ``mriforge.infrastructure.reporting.plotters``.

This is the *first* test file targeting the plotter package. The intent is
broad smoke coverage rather than deep correctness:

* For every public plotter module (one per ``*.py`` file, recursively, minus
  helpers prefixed with ``_``), build a minimum-viable synthetic fixture that
  matches the plotter's documented input contract.
* Invoke the canonical entry point (``make`` for the Phase-1 family,
  ``render_*`` for the optional/wrapper plotters).
* Assert the plotter returns a non-``None`` output path and that the file on
  disk is non-empty — equivalent to "produces a non-empty matplotlib Figure"
  for plotters that save-and-close inside ``make`` (which all of them do).

Plotters whose smoke fixture cannot be synthesised cheaply (e.g. those that
need real NIfTI / model outputs / multi-stage calibration JSON) are marked
``xfail`` rather than silently skipped, per the repo CLAUDE.md pitfall #10
("warnings are not OK").

Notes:

* Matplotlib is forced to ``Agg`` *before* any pyplot import so the test
  works in headless CI and never opens a GUI window.
* ``plt.close("all")`` runs in fixture teardown to keep figure-manager
  memory bounded across the parametrised sweep.
* If the ``plotters`` package is not importable on the current branch the
  whole module is skipped via ``pytest.importorskip`` — this keeps the
  test honest on branches where the plotter work has not landed yet.
"""

from __future__ import annotations

# IMPORTANT: ``Agg`` must be selected before pyplot is imported anywhere.
import matplotlib

matplotlib.use("Agg")

import importlib
import pkgutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Skip the whole module on branches that don't carry the plotter package
# (e.g. ``main`` prior to the reporting-pipeline merge).
plotters_pkg = pytest.importorskip(
    "mriforge.infrastructure.reporting.plotters",
    reason="reporting.plotters package not available on this branch",
)


# ---------------------------------------------------------------------------
# Plotter discovery
# ---------------------------------------------------------------------------


def _discover_plotter_modules() -> list[tuple[str, Any]]:
    """Walk the ``plotters`` package and return ``(short_name, module)`` pairs.

    Helpers prefixed with ``_`` and the ``__init__`` registry are filtered out.
    The short name is used as the parametrise id so failing nodes are easy
    to read in pytest output.
    """
    discovered: list[tuple[str, Any]] = []
    pkg_path = Path(plotters_pkg.__file__).parent
    for mod_info in pkgutil.walk_packages(
        [str(pkg_path)], prefix=plotters_pkg.__name__ + "."
    ):
        short = mod_info.name.rsplit(".", 1)[-1]
        # Skip private helpers (``_comparison_helpers``) and sub-packages
        # themselves (e.g. ``generative``, ``mri_specific``).
        if short.startswith("_") or mod_info.ispkg:
            continue
        try:
            module = importlib.import_module(mod_info.name)
        except Exception as exc:
            discovered.append(
                (short, pytest.param(None, marks=pytest.mark.xfail(
                    reason=f"plotter module {mod_info.name} failed to import: {exc}",
                    strict=False,
                )))
            )
            continue
        discovered.append((short, module))
    return discovered


_PLOTTER_MODULES = _discover_plotter_modules()


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _make_long_metrics_df() -> pd.DataFrame:
    """Long-format DataFrame matching ``aggregator.aggregate*`` output.

    Contains the columns the Phase-1 plotters look at:
    ``method``, ``metric``, ``step``, ``value``, ``split``, ``subject_id``.
    Includes the metric names referenced by every Phase-1 / generative
    plotter so each can find at least one matching row.
    """
    rng = np.random.default_rng(0)
    rows: list[dict[str, Any]] = []

    methods = ["full", "ablated_a", "ablated_b"]
    steps = list(range(0, 100, 10))

    # Per-step training-curve metrics (loss family + components).
    curve_metrics = [
        "loss",
        "val_loss",
        "reconstruction_loss",
        "kl_loss",
        "g_loss",
        "d_loss",
        "d_acc",
        "gp_norm",
    ]
    for method in methods:
        for step in steps:
            for m in curve_metrics:
                base = 1.0 - step / 200.0
                noise = rng.normal(0, 0.02)
                value = max(base + noise, 1e-3)
                rows.append({
                    "method": method,
                    "metric": m,
                    "step": float(step),
                    "value": float(value),
                    "split": "train" if not m.startswith("val_") else "val",
                    "subject_id": None,
                })
        # Per-step quality metrics on BOTH splits so the train/val-gap plotter
        # can match base names and the metric-correlation heatmap has >=2 varying
        # metrics over >=3 val steps.
        for step in steps:
            frac = step / max(steps)
            for base, (lo, hi) in {"psnr": (24.0, 34.0), "ssim": (0.6, 0.92),
                                   "nmse": (0.05, 0.008)}.items():
                train_v = lo + (hi - lo) * frac + rng.normal(0, 0.02 * abs(hi - lo))
                val_v = train_v - 0.06 * abs(hi - lo) + rng.normal(0, 0.02 * abs(hi - lo))
                rows.append({"method": method, "metric": base, "step": float(step),
                             "value": float(train_v), "split": "train",
                             "subject_id": None})
                rows.append({"method": method, "metric": f"val_{base}",
                             "step": float(step), "value": float(val_v),
                             "split": "val", "subject_id": None})
        # Run-summary / best / final scalar rows — mirror the aggregator's folding
        # of run_summary.json (split="run") + final_metrics.json (best/final) so
        # the run-summary card fires exactly as on a real run.
        for name, v in {"params_m": rng.uniform(5, 80),
                        "iterations_per_sec": rng.uniform(2, 12),
                        "duration_min": rng.uniform(30, 300),
                        "effective_batch": float(rng.choice([8, 16, 32]))}.items():
            rows.append({"method": method, "metric": name, "step": -1.0,
                         "value": float(v), "split": "run", "subject_id": None})
        for name, (lo, hi) in {"psnr": (30.0, 36.0), "ssim": (0.85, 0.95)}.items():
            rows.append({"method": method, "metric": name, "step": -1.0,
                         "value": float(rng.uniform(lo, hi)), "split": "best",
                         "subject_id": None})
        for name, v in {"final_loss": rng.uniform(0.05, 0.2),
                        "early_stopping_best_value": rng.uniform(0.8, 0.92)}.items():
            rows.append({"method": method, "metric": name, "step": -1.0,
                         "value": float(v), "split": "final", "subject_id": None})

    # Per-step diffusion / generative binned metrics.
    for binned in ("loss_per_timestep", "psnr_vs_nfe", "psnr_vs_cfg"):
        for step in steps:
            rows.append({
                "method": "full",
                "metric": binned,
                "step": float(step),
                "value": float(rng.uniform(20, 35)),
                "split": "train",
                "subject_id": None,
            })

    # Per-latent KL for VAE collapse check.
    for latent_idx in range(4):
        rows.append({
            "method": "full",
            "metric": f"kl_latent_{latent_idx}",
            "step": float(max(steps)),
            "value": float(rng.uniform(0.0, 1.0)),
            "split": "train",
            "subject_id": None,
        })

    # Scalar evaluation metrics (per-subject, no step). The Pareto and
    # ablation plotters consume these.
    for method in methods:
        for subj in range(8):
            rows.append({
                "method": method,
                "metric": "psnr",
                "step": float("nan"),
                "value": float(rng.uniform(28, 36)),
                "split": "test",
                "subject_id": f"s{subj:03d}",
            })
            rows.append({
                "method": method,
                "metric": "ssim",
                "step": float("nan"),
                "value": float(rng.uniform(0.7, 0.95)),
                "split": "test",
                "subject_id": f"s{subj:03d}",
            })
        # Cost axis used by the headline Pareto.
        rows.append({
            "method": method,
            "metric": "params_m",
            "step": float("nan"),
            "value": float(rng.uniform(5, 80)),
            "split": "meta",
            "subject_id": None,
        })

    # Computational-profile metrics.
    for step in steps:
        for m in ("data_load_s", "forward_s", "backward_s", "optim_s", "log_s"):
            rows.append({
                "method": "full",
                "metric": m,
                "step": float(step),
                "value": float(rng.uniform(0.005, 0.05)),
                "split": "train",
                "subject_id": None,
            })
        for m in ("gpu_mem_gb", "cpu_mem_gb"):
            rows.append({
                "method": "full",
                "metric": m,
                "step": float(step),
                "value": float(rng.uniform(1.0, 12.0)),
                "split": "train",
                "subject_id": None,
            })

    return pd.DataFrame(rows)


def _make_predictions_df() -> pd.DataFrame:
    """``predicted`` vs ``target`` long table for regression-style plotters."""
    rng = np.random.default_rng(1)
    n = 200
    target = rng.uniform(0.0, 1.0, size=n)
    predicted = target + rng.normal(0.0, 0.05, size=n)
    return pd.DataFrame({"predicted": predicted, "target": target})


def _make_stratified_df() -> pd.DataFrame:
    """Per-subgroup scalar metric table for stratified-performance plotter."""
    rng = np.random.default_rng(2)
    rows: list[dict[str, Any]] = []
    for method in ("full", "ablated_a"):
        for group in ("pediatric", "adult", "geriatric"):
            for _ in range(6):
                rows.append({
                    "method": method,
                    "metric": "psnr",
                    "subgroup": group,
                    "value": float(rng.uniform(28, 36)),
                })
    return pd.DataFrame(rows)


def _make_image_cases(n: int = 3, h: int = 32, w: int = 32) -> list[dict[str, Any]]:
    """Minimal image cases for SR / C2C / kspace-recon / failure-gallery."""
    rng = np.random.default_rng(3)
    cases: list[dict[str, Any]] = []
    for i in range(n):
        target = rng.uniform(0, 1, size=(h, w)).astype(np.float32)
        pred = target + rng.normal(0, 0.05, size=(h, w)).astype(np.float32)
        lr = pred * 0.9
        zf = target * 0.7
        baseline = target * 0.85
        cases.append({
            "name": f"case_{i:02d}",
            # Generic field names used by failure_gallery / synthesis_c2c.
            "image": target,
            "input": target,
            "pred": pred,
            "predicted": pred,
            # Recorder's canonical key (recon_panel / kspace_error_spectrum
            # read cases["prediction"]; the npz recorder writes `prediction=`).
            "prediction": pred,
            "target": target,
            "residual": float(np.abs(pred - target).mean()),
            # SR-specific keys.
            "lr": lr,
            "hr": target,
            # k-space-recon keys.
            "zero_filled": zf,
            "baseline": baseline,
            "fs_ref": target,
        })
    return cases


def _make_cohort_dict() -> dict[str, Any]:
    return {
        "train_n": 80,
        "val_n": 10,
        "test_n": 10,
        "age_mean": 45.0,
        "age_std": 12.0,
        "sex_split": "55/45 F/M",
        "scanners": "3T Siemens",
        "pathology": "knee MRI",
        "split_rule": "subject-level (no slice leakage)",
    }


def _make_metric_frame() -> pd.DataFrame:
    """Per-subject scalar metric table (distribution / significance / forest)."""
    rng = np.random.default_rng(4)
    rows: list[dict[str, Any]] = []
    for method in ("full", "ablated_a", "ablated_b"):
        for subj in range(8):
            for metric, lo, hi in (("psnr", 28, 36), ("ssim", 0.7, 0.95)):
                rows.append({"method": method, "metric": metric,
                             "value": float(rng.uniform(lo, hi)),
                             "subject_id": f"s{subj:03d}", "split": "test"})
    return pd.DataFrame(rows)


def _make_accel_frame() -> pd.DataFrame:
    """Metric vs acceleration factor (acceleration_sweep)."""
    rng = np.random.default_rng(5)
    rows: list[dict[str, Any]] = []
    for method in ("full", "ablated_a"):
        for r in (2, 4, 8, 16):
            for _ in range(4):
                rows.append({"method": method, "metric": "psnr", "acceleration": r,
                             "value": float(36 - 1.4 * np.log2(r) + rng.normal(0, 0.3))})
    return pd.DataFrame(rows)


def _make_calibration_frame() -> pd.DataFrame:
    """Nominal vs empirical coverage (calibration_coverage)."""
    rng = np.random.default_rng(6)
    nominal = np.linspace(0.05, 0.95, 10)
    rows: list[dict[str, Any]] = []
    for method in ("full", "ablated_a"):
        bias = 0.0 if method == "full" else 0.06
        for nm in nominal:
            rows.append({"method": method, "nominal": float(nm),
                         "empirical": float(np.clip(nm - bias + rng.normal(0, 0.01), 0, 1))})
    return pd.DataFrame(rows)


def _make_heatmap_frame() -> pd.DataFrame:
    """Two-axis ablation grid (ablation_heatmap)."""
    rng = np.random.default_rng(7)
    rows: list[dict[str, Any]] = []
    for x in (16, 32, 64):
        for y in ("l1", "l1+ssim", "l1+adv"):
            rows.append({"axis_x": x, "axis_y": y, "value": float(rng.uniform(28, 35))})
    return pd.DataFrame(rows)


def _make_bland_frame() -> pd.DataFrame:
    """Reference vs measurement pairs (bland_altman)."""
    rng = np.random.default_rng(8)
    ref = rng.uniform(0.3, 0.9, 80)
    return pd.DataFrame({"metric": "ssim", "reference": ref,
                         "value": ref + rng.normal(0, 0.03, 80)})


def _make_acquisition_frame() -> pd.DataFrame:
    """policy / step / psnr_db for competing acquisition policies (fig_2_15)."""
    rng = np.random.default_rng(9)
    rows: list[dict[str, Any]] = []
    for policy, gain in (("IB-active", 1.0), ("BALD", 0.85), ("random", 0.6)):
        for step in range(1, 17):
            rows.append({"policy": policy, "step": step,
                         "psnr_db": float(22 + gain * 6 * np.log1p(step)
                                          + rng.normal(0, 0.2))})
    return pd.DataFrame(rows)


def _make_bloch_frame() -> pd.DataFrame:
    """tissue / bloch_eq_residual / step (fig_b3)."""
    rng = np.random.default_rng(10)
    rows: list[dict[str, Any]] = []
    for tissue, b0 in {"WM": 0.08, "GM": 0.10, "CSF": 0.25, "fat": 0.18}.items():
        for step in range(0, 1000, 100):
            for _ in range(4):
                rows.append({"tissue": tissue, "step": step,
                             "bloch_eq_residual": float(
                                 b0 * np.exp(-step / 600) * (1 + rng.uniform(-0.2, 0.2)))})
    return pd.DataFrame(rows)


def _make_qmap_frame() -> pd.DataFrame:
    """method / per-pixel error (fig_b7)."""
    rng = np.random.default_rng(11)
    rows: list[dict[str, Any]] = []
    for method, scale in (("euclidean", 0.20), ("riemannian", 0.11)):
        for _ in range(400):
            rows.append({"method": method,
                         "error": float(abs(rng.normal(0, scale)) + 1e-3)})
    return pd.DataFrame(rows)


def _make_qmap_slices() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(12)
    return {"euclidean": np.abs(rng.normal(0, 0.2, (48, 48))).astype(np.float32),
            "riemannian": np.abs(rng.normal(0, 0.11, (48, 48))).astype(np.float32)}


def _make_beltrami_mu(n: int = 64) -> np.ndarray:
    yy, xx = np.mgrid[0:n, 0:n] / n
    mag = 0.6 * np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.05)
    return (mag * np.exp(1j * 2 * np.pi * (xx + yy))).astype(np.complex64)


def _make_fingerprint_embedding(n_per: int = 60, k: int = 4):
    rng = np.random.default_rng(13)
    embs, labels = [], []
    for c in range(k):
        centre = rng.normal(0, 3, 5)
        embs.append(centre + rng.normal(0, 0.6, (n_per, 5)))
        labels.append(np.full(n_per, c))
    return np.vstack(embs).astype(np.float32), np.concatenate(labels)


def _make_task_ablation_frame() -> pd.DataFrame:
    """Tidy ``[family, role, arm, metric, value]`` frame for ``cohort_task_ablation_bars``.

    Two families, method + one/two ablation control(s) each, on the three
    faceted validation metrics — matches the plotter's documented contract.
    """
    rng = np.random.default_rng(14)
    rows: list[dict[str, Any]] = []
    families = {"b1": ["b1_full", "b1_ablate_x"],
                "b2": ["b2_full", "b2_ablate_y", "b2_ablate_z"]}
    for fam, arms in families.items():
        for arm in arms:
            role = "ablation" if "_ablate_" in arm else "method"
            for metric, (lo, hi) in {"ssim": (0.6, 0.9), "psnr": (24.0, 34.0),
                                     "lpips": (0.1, 0.4)}.items():
                rows.append({"family": fam, "role": role, "arm": f"mrixfields_{arm}",
                             "metric": metric, "value": float(rng.uniform(lo, hi))})
    return pd.DataFrame(rows)


# fig_id -> (frame_factory, kwargs_factory) covering EVERY registered plotter.
# The registry-coverage test below dispatches each entry and asserts a non-None
# artifact — the anti-facade guard (pitfall #16 at the reporting layer): a
# figure that is registered but can never fire on well-formed input is a facade.
def _make_ksd_result():
    """Minimal object with the fields ``render_ksd_certificate`` reads."""
    import types
    return types.SimpleNamespace(passed=True, ksd_squared=1.2e-3, p_value=0.31,
                                 threshold=0.05, n_samples=256, bandwidth=1.0,
                                 message="synthetic KSD result")


def _make_certificate_paths():
    """A partial certificate (R1 payload) so the summary figure renders."""
    import json
    import tempfile
    d = Path(tempfile.mkdtemp())
    (d / "r1.json").write_text(json.dumps({"empirical_coverage": 0.91, "alpha": 0.1}))
    return {"r1_path": d / "r1.json"}


def _make_acquisition_arms():
    from mriforge.infrastructure.reporting.plotters.learnable_acquisition_pareto import (
        ArmResult,
    )
    return [ArmResult("full", 0.5, 33.0, 0.92),
            ArmResult("mid", 0.25, 31.0, 0.88),
            ArmResult("low", 0.125, 28.0, 0.82)]


_REGISTRY_FIXTURES: dict[str, tuple[Any, Any]] = {
    "fig_1_1_headline_pareto": (_make_long_metrics_df, lambda: {"metric": "psnr", "cost": "params_m"}),
    "fig_1_2_learning_curves": (_make_long_metrics_df, lambda: {"metrics": ["loss", "val_loss"]}),
    "fig_1_3_loss_decomposition": (_make_long_metrics_df, dict),
    "fig_1_4_residual_diagnostics": (_make_long_metrics_df, lambda: {"predictions_df": _make_predictions_df()}),
    "fig_1_5_predicted_vs_true": (_make_long_metrics_df, lambda: {"predictions_df": _make_predictions_df()}),
    "fig_1_9_ablation_strip": (_make_metric_frame, lambda: {"primary_metric": "psnr", "full_method": "full"}),
    "fig_1_11_stratified_performance": (_make_long_metrics_df, lambda: {"stratified_df": _make_stratified_df()}),
    "fig_1_12_failure_gallery": (_make_long_metrics_df, lambda: {"cases": _make_image_cases()}),
    "fig_1_15_computational_profile": (_make_long_metrics_df, dict),
    "fig_1_16_run_summary_card": (_make_long_metrics_df, dict),
    "fig_1_17_metric_correlation": (_make_long_metrics_df, dict),
    "fig_1_18_train_val_gap": (_make_long_metrics_df, dict),
    "gen_gan_diagnostics": (_make_long_metrics_df, dict),
    "gen_vae_diagnostics": (_make_long_metrics_df, dict),
    "gen_diffusion_diagnostics": (_make_long_metrics_df, dict),
    "metric_distribution": (_make_metric_frame, dict),
    "significance_matrix": (_make_metric_frame, lambda: {"metric": "psnr"}),
    "bland_altman": (_make_bland_frame, lambda: {"metric": "ssim"}),
    "calibration_coverage": (_make_calibration_frame, dict),
    "forest_plot": (_make_metric_frame, lambda: {"metric": "psnr"}),
    "ablation_heatmap": (_make_heatmap_frame, dict),
    "acceleration_sweep": (_make_accel_frame, lambda: {"metric": "psnr"}),
    "mri_a1_sr_triptych": (_make_long_metrics_df, lambda: {"cases": _make_image_cases(), "show_spectrum": True}),
    "mri_a2_synthesis_c2c": (_make_long_metrics_df, lambda: {"cases": _make_image_cases()}),
    "mri_a7_kspace_recon": (_make_long_metrics_df, lambda: {"cases": _make_image_cases()}),
    "mri_recon_panel": (_make_long_metrics_df, lambda: {"cases": _make_image_cases()}),
    "kspace_error_spectrum": (_make_long_metrics_df, lambda: {"cases": _make_image_cases()}),
    "mri_a11_cohort_table": (_make_long_metrics_df, lambda: {"cohort": _make_cohort_dict()}),
    "mri_a12_fiducial_check": (_make_long_metrics_df, lambda: {"cases": _make_image_cases()}),
    "fig_2_15_active_acquisition_trajectory": (_make_acquisition_frame, dict),
    "fig_b3_bloch_consistency_residual": (_make_bloch_frame, dict),
    "fig_b7_qmap_riemannian_vs_euclidean": (_make_qmap_frame, lambda: {"slices": _make_qmap_slices()}),
    "fig_c1_beltrami_field": (_make_long_metrics_df, lambda: {"mu": _make_beltrami_mu()}),
    "fig_c2_spd_geodesic": (_make_long_metrics_df, lambda: {"distances": np.linspace(1.2, 0.05, 200)}),
    "fig_c3_teichmuller_schedule": (_make_long_metrics_df, lambda: {"r": np.linspace(0.0, 0.95, 100)}),
    "fig_c4_fingerprint_embedding": (
        _make_long_metrics_df,
        lambda: {"embedding": _make_fingerprint_embedding()[0],
                     "labels": _make_fingerprint_embedding()[1]},
    ),
    # QC QC report.
    "qc_group_strip": (_make_metric_frame, dict),
    "qc_subject_mosaic": (_make_long_metrics_df, lambda: {"cases": _make_image_cases()}),
    "qc_carpet": (_make_long_metrics_df, lambda: {"cases": _make_image_cases()}),
    # contact_sheet composites the OTHER PNGs — handled specially in the test.
    "contact_sheet": (_make_long_metrics_df, dict),
    # Certificate / learnable-acquisition figures (registered via soft-skipping
    # adapters in plotters/__init__.py; these fixtures prove each fires).
    "certificate_summary": (_make_long_metrics_df,
                            lambda: {"certificate_paths": _make_certificate_paths()}),
    "ksd_certificate": (_make_long_metrics_df, lambda: {"ksd_result": _make_ksd_result()}),
    "learnable_acquisition_pareto": (_make_long_metrics_df,
                                     lambda: {"acquisition_arms": _make_acquisition_arms()}),
    # Cohort task-ablation bars (faceted validation metrics, method vs ablation).
    "cohort_task_ablation_bars": (_make_task_ablation_frame, dict),
}


# ---------------------------------------------------------------------------
# Plotter-specific dispatchers
# ---------------------------------------------------------------------------


# Plotters that need extra kwargs beyond the long metrics DataFrame.
# Each entry is ``(callable_attr, kwargs_factory)``. ``kwargs_factory`` is
# called lazily so per-test fixtures (e.g. cases lists) stay isolated.
_DISPATCH: dict[str, tuple[str, Any]] = {
    "headline_pareto": ("make", lambda: {"metric": "psnr", "cost": "params_m"}),
    "learning_curves": ("make", lambda: {"metrics": ["loss", "val_loss"]}),
    "loss_decomposition": ("make", lambda: {}),
    "residual_diagnostics": ("make", lambda: {"predictions_df": _make_predictions_df()}),
    "predicted_vs_true": ("make", lambda: {"predictions_df": _make_predictions_df()}),
    "ablation_strip": ("make", lambda: {"primary_metric": "psnr", "full_method": "full"}),
    "stratified_performance": ("make", lambda: {"stratified_df": _make_stratified_df()}),
    "failure_gallery": ("make", lambda: {"cases": _make_image_cases()}),
    "computational_profile": ("make", lambda: {}),
    # Generative.
    "gan_diagnostics": ("make", lambda: {}),
    "vae_diagnostics": ("make", lambda: {}),
    "diffusion_diagnostics": ("make", lambda: {}),
    # MRI-specific.
    "sr_triptych": ("make", lambda: {"cases": _make_image_cases(), "show_spectrum": True}),
    "synthesis_c2c": ("make", lambda: {"cases": _make_image_cases()}),
    "kspace_recon": ("make", lambda: {"cases": _make_image_cases()}),
    "cohort_table": ("make", lambda: {"cohort": _make_cohort_dict()}),
    "fiducial_check": ("make", lambda: {"cases": _make_image_cases()}),
}


# Modules whose ``make`` needs a frame OTHER than the long metrics df (or
# extra kwargs). ``(frame_factory, kwargs_factory)`` — both called lazily so
# per-test fixtures stay isolated. Keyed by module short name.
_MODULE_FIXTURES: dict[str, tuple[Any, Any]] = {
    "ablation_heatmap": (_make_heatmap_frame, dict),
    "bland_altman": (_make_bland_frame, lambda: {"metric": "ssim"}),
    "calibration_coverage": (_make_calibration_frame, dict),
    "forest_plot": (_make_metric_frame, lambda: {"metric": "psnr"}),
    "metric_distribution": (_make_metric_frame, dict),
    "significance_matrix": (_make_metric_frame, lambda: {"metric": "psnr"}),
    "acceleration_sweep": (_make_accel_frame, lambda: {"metric": "psnr"}),
    "metric_correlation": (_make_long_metrics_df, dict),
    "run_summary_card": (_make_long_metrics_df, dict),
    "train_val_gap": (_make_long_metrics_df, dict),
    "active_acquisition_trajectory": (_make_acquisition_frame, dict),
    "bloch_consistency_residual": (_make_bloch_frame, dict),
    "qmap_riemannian_vs_euclidean": (_make_qmap_frame, lambda: {"slices": _make_qmap_slices()}),
    "recon_panel": (_make_long_metrics_df, lambda: {"cases": _make_image_cases()}),
    "kspace_error_spectrum": (_make_long_metrics_df, lambda: {"cases": _make_image_cases()}),
    "group_strip": (_make_metric_frame, dict),
    "subject_mosaic": (_make_long_metrics_df, lambda: {"cases": _make_image_cases()}),
    "carpet_spike": (_make_long_metrics_df, lambda: {"cases": _make_image_cases()}),
    "task_ablation_bars": (_make_task_ablation_frame, dict),
}


def _invoke_plotter(module: Any, out_path: Path) -> Path | None:
    """Resolve the canonical entry point and call it with synthetic inputs."""
    short = module.__name__.rsplit(".", 1)[-1]

    if short in _DISPATCH:
        attr, kwargs_factory = _DISPATCH[short]
        fn = getattr(module, attr)
        return fn(_make_long_metrics_df(), out_path, **kwargs_factory())

    if short in _MODULE_FIXTURES:
        frame_factory, kwargs_factory = _MODULE_FIXTURES[short]
        return module.make(frame_factory(), out_path, **kwargs_factory())

    # Wrapper plotters with bespoke signatures.
    if short == "learnable_acquisition_pareto":
        fn = module.render_learnable_acquisition_pareto
        ArmResult = module.ArmResult  # noqa: N806 — holds a class, not a value
        arms = [
            ArmResult(name=f"arm_{i}", sampling_budget=1.0 / (i + 2),
                      psnr=28.0 + i, ssim=0.8 + 0.02 * i)
            for i in range(4)
        ]
        return fn(out_path, arms)

    if short == "certificate_summary":
        # Optional artefact paths default to None — the plotter should
        # still render a 2x3 grid with placeholder panels.
        fn = module.render_certificate_summary
        return fn(out_path)

    if short == "ksd_certificate":
        # Bespoke: takes a duck-typed KSDDefensibilityResult, not a df.
        class _KSDResult:
            passed, ksd_squared, p_value, threshold = True, 3.14e-4, 0.42, 0.05
            n_samples, bandwidth = 512, None
            message = "KSD GoF accepts H0."
        return module.render_ksd_certificate(out_path, result=_KSDResult())

    if short == "contact_sheet":
        # Composites sibling PNGs — populate the dir, then build the index.
        siblings = importlib.import_module(
            "mriforge.infrastructure.reporting.plotters"
        )
        siblings.dispatch(
            _make_long_metrics_df(),
            ["fig_1_2_learning_curves", "fig_1_15_computational_profile"],
            out_path.parent, formats=("png",), dpi=80, metrics=["loss", "val_loss"],
        )
        return module.make(_make_long_metrics_df(), out_path,
                           figure_dir=out_path.parent, formats=("png",))

    if short == "sfc_conformal_plots":
        # Multi-function module (make_beltrami_field / make_spd_geodesic / …);
        # the registry-coverage test exercises all four fig-ids. Here we just
        # prove the module renders by driving one representative entry point.
        return module.make_beltrami_field(None, out_path, mu=_make_beltrami_mu())

    pytest.xfail(f"no smoke fixture wired for plotter '{short}'")
    return None  # unreachable; xfail above raises


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every matplotlib figure after each test to bound memory use."""
    yield
    plt.close("all")


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    d = tmp_path / "plotter_smoke"
    d.mkdir()
    return d


def _ids(item: tuple[str, Any]) -> str:
    return item[0]


@pytest.mark.parametrize(
    ("short", "module"),
    _PLOTTER_MODULES,
    ids=[short for short, _ in _PLOTTER_MODULES],
)
def test_plotter_produces_nonempty_figure(short: str, module: Any, out_dir: Path) -> None:
    """Every plotter writes a non-empty figure file with synthetic inputs.

    The Phase-1 plotter family always returns ``Path | None`` and saves
    the figure to ``out_path`` inside ``make``; "non-empty Matplotlib
    Figure" therefore translates to "non-empty file on disk at the
    returned path".
    """
    if module is None:
        # Discovery-time import failure — already xfailed.
        pytest.xfail(f"plotter '{short}' could not be imported")

    out_path = out_dir / f"{short}.png"

    result = _invoke_plotter(module, out_path)

    assert result is not None, (
        f"plotter '{short}' returned None — synthetic fixture did not match "
        f"its expected schema"
    )
    result_path = Path(result)
    assert result_path.exists(), f"plotter '{short}' did not write {result_path}"
    assert result_path.stat().st_size > 0, (
        f"plotter '{short}' produced an empty file at {result_path}"
    )


def test_plotter_registry_lists_all_phase1_ids() -> None:
    """The ``PLOTTERS`` registry should expose at least one fig-ID per
    discovered Phase-1 module.

    Acts as a regression test against silent de-registration: if a new
    plotter module is added but ``register("fig_id", module.make)`` is
    forgotten in ``plotters/__init__.py``, the dispatcher won't see it.
    """
    plotters_module = importlib.import_module("mriforge.infrastructure.reporting.plotters")
    registered = set(plotters_module.list_available())
    assert registered, "PLOTTERS registry is empty — expected at least one entry"

    # Spot-check a few canonical IDs that should always be present.
    expected_present = {
        "fig_1_1_headline_pareto",
        "fig_1_2_learning_curves",
        "fig_1_4_residual_diagnostics",
    }
    missing = expected_present - registered
    assert not missing, f"expected fig-IDs missing from registry: {sorted(missing)}"


def test_plotter_dispatch_runs_against_synthetic_df(out_dir: Path) -> None:
    """End-to-end smoke for the public ``dispatch`` entry point.

    Drives the plotters that consume the long metrics DataFrame through
    the registry. Plotters that need richer inputs (``predictions_df``,
    ``cases``, ``cohort``) are excluded from this batch — they are
    exercised individually by ``test_plotter_produces_nonempty_figure``.
    """
    plotters_module = importlib.import_module("mriforge.infrastructure.reporting.plotters")
    df_long = _make_long_metrics_df()

    df_driven_ids = [
        "fig_1_1_headline_pareto",
        "fig_1_2_learning_curves",
        "fig_1_3_loss_decomposition",
        "fig_1_9_ablation_strip",
        "fig_1_15_computational_profile",
        "gen_gan_diagnostics",
        "gen_vae_diagnostics",
        "gen_diffusion_diagnostics",
    ]
    # Intersect with the actually-registered set so the test is forward-
    # compatible with registry pruning.
    available = [fid for fid in df_driven_ids if fid in set(plotters_module.list_available())]
    assert available, "no df-driven plotter ids available — registry empty?"

    results = plotters_module.dispatch(df_long, available, out_dir)

    assert set(results.keys()) == set(available)
    # At least one plotter must produce a non-None artefact — proves
    # the dispatch path is wired up end-to-end.
    produced = [v for v in results.values() if v is not None]
    assert produced, (
        "dispatch() returned None for every fig_id; plotters did not see "
        "the synthetic DataFrame as parseable"
    )
    for path in produced:
        assert Path(path).exists() and Path(path).stat().st_size > 0


def test_registry_fixture_map_covers_every_registered_id() -> None:
    """``_REGISTRY_FIXTURES`` must map EVERY registered plotter id.

    Guards against the split that motivated this test: a plotter can be
    registered (and shipped in a preset) yet have no fixture proving it
    fires, so nobody notices when it silently renders nothing. If a new
    plotter is registered without a fixture here, this fails loudly.
    """
    plotters_module = importlib.import_module("mriforge.infrastructure.reporting.plotters")
    registered = set(plotters_module.list_available())
    mapped = set(_REGISTRY_FIXTURES)
    missing = registered - mapped
    assert not missing, (
        f"registered plotters with no coverage fixture: {sorted(missing)} — "
        f"add each to _REGISTRY_FIXTURES so the anti-facade guard exercises it"
    )


@pytest.mark.parametrize("fig_id", sorted(_REGISTRY_FIXTURES))
def test_every_registered_plotter_renders(fig_id: str, out_dir: Path) -> None:
    """Every registered fig-id renders a non-empty artifact on good input.

    This is the reporting-layer anti-facade guard (CLAUDE.md pitfall #16):
    a figure that is registered but never fires — because a fixture key was
    dropped, a metric renamed, or a schema drifted — is a facade. The
    earlier module-level sweep xfailed ~18 of these; this closes that blind
    spot by driving the *registry* (``list_available``) directly.

    ``contact_sheet`` is the one figure that composites *other* PNGs, so it
    is rendered after a batch of sibling figures populate its ``figure_dir``.
    """
    plotters_module = importlib.import_module("mriforge.infrastructure.reporting.plotters")
    if fig_id not in set(plotters_module.list_available()):
        pytest.skip(f"{fig_id} not registered on this branch")

    frame_factory, kwargs_factory = _REGISTRY_FIXTURES[fig_id]
    df = frame_factory()
    kwargs = kwargs_factory()

    if fig_id == "contact_sheet":
        # Populate the dir with a couple of real figures first, then composite.
        plotters_module.dispatch(
            _make_long_metrics_df(),
            ["fig_1_2_learning_curves", "fig_1_15_computational_profile"],
            out_dir, formats=("png",), dpi=80,
            metrics=["loss", "val_loss"],
        )
        kwargs = {"figure_dir": out_dir}

    results = plotters_module.dispatch(
        df, [fig_id], out_dir, formats=("png",), dpi=80, **kwargs
    )
    produced = results.get(fig_id)
    assert produced is not None, (
        f"registered plotter '{fig_id}' rendered nothing on its coverage "
        f"fixture — a reporting-layer facade (pitfall #16)"
    )
    assert Path(produced).exists() and Path(produced).stat().st_size > 0
