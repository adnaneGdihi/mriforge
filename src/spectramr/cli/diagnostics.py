"""``spectramr doctor`` — environment & wiring diagnostics for cluster use.

A single command that answers "is my environment set up to train here?" without
launching a job: version, Python/platform, torch + CUDA (devices, memory),
determinism flags, the resolved cache/data roots, the relevant ``SPECTRAMR_*``
env knobs, and (optionally) whether a given config loads and a checkpoint exists.

Designed to be import-safe: torch is imported lazily and guarded so ``doctor``
still runs (and reports the failure) on a broken/CPU-only environment — exactly
when you most need it.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import sys
from pathlib import Path
from typing import Any

# The env knobs the framework actually reads (kept in sync with env_resolver +
# the pitfall-#15 knob table). Shown so a cluster user can confirm what's set.
_ENV_KNOBS = (
    "SPECTRAMR_CACHE_ROOT",
    "SPECTRAMR_CLUSTER_ROOT",
    "SPECTRAMR_DATA_ROOT",
    "SPECTRAMR_DEVICE",
    "SPECTRAMR_ROOT",
    "SPECTRAMR_NO_GPU_PROBE",
    "SPECTRAMR_SUPPRESS_CLINICAL_WARNING",
    "SIM2RANK_MODE",
    "TMPDIR",
    "CUDA_VISIBLE_DEVICES",
    "SLURM_JOB_ID",
    "SLURM_NODELIST",
)


def _version() -> str:
    try:
        return importlib.metadata.version("spectramr")
    except Exception:  # pragma: no cover - editable installs without metadata
        return "unknown"


def _torch_info() -> dict[str, Any]:
    """Probe torch/CUDA without crashing if torch is broken or absent.

    Thin delegator to the canonical probe in
    :func:`spectramr.infrastructure.logging.provenance.torch_runtime` (the SSOT
    shared with run-provenance capture). Kept as a module-level function so
    callers and tests can patch ``diagnostics._torch_info`` directly.
    """
    from spectramr.infrastructure.logging.provenance import torch_runtime

    return torch_runtime()


def collect_diagnostics() -> dict[str, Any]:
    """Gather a structured snapshot of the runtime environment (pure / testable)."""
    from spectramr.infrastructure.config.env_resolver import resolve_cache_root

    try:
        cache_root = str(resolve_cache_root())
    except Exception as exc:  # pragma: no cover
        cache_root = f"<error: {exc}>"

    data_root = os.environ.get("SPECTRAMR_DATA_ROOT", "databases/")
    return {
        "version": _version(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "torch": _torch_info(),
        "cache_root": cache_root,
        "data_root": {"path": data_root, "exists": Path(data_root).exists()},
        "env": {k: os.environ.get(k) for k in _ENV_KNOBS if os.environ.get(k) is not None},
        "in_slurm": "SLURM_JOB_ID" in os.environ,
    }


def render_diagnostics(diag: dict[str, Any]) -> str:
    """Human-readable rendering of :func:`collect_diagnostics`."""
    lines = ["spectramr doctor", "=" * 40]
    lines.append(f"version       : {diag['version']}")
    lines.append(f"python        : {diag['python']}")
    lines.append(f"platform      : {diag['platform']}")
    lines.append(f"executable    : {diag['executable']}")

    t = diag["torch"]
    if not t.get("available"):
        lines.append(f"torch         : NOT IMPORTABLE ({t.get('import_error')})")
    else:
        lines.append(f"torch         : {t['version']} (cuda build: {t.get('cuda_compiled')})")
        if t.get("cuda_available"):
            lines.append(f"cuda          : available — {len(t['devices'])} device(s)")
            for d in t["devices"]:
                lines.append(
                    f"  [{d['index']}] {d['name']} "
                    f"({d['total_mem_gb']} GB, sm_{d['capability'].replace('.', '')})"
                )
        else:
            lines.append("cuda          : NOT available (CPU-only)")
        lines.append(
            f"cudnn         : benchmark={t['cudnn_benchmark']} "
            f"deterministic={t['cudnn_deterministic']}"
        )

    lines.append(f"cache_root    : {diag['cache_root']}")
    dr = diag["data_root"]
    lines.append(f"data_root     : {dr['path']} ({'present' if dr['exists'] else 'MISSING'})")
    lines.append(f"in SLURM job  : {diag['in_slurm']}")
    if diag["env"]:
        lines.append("env knobs set :")
        for k, v in diag["env"].items():
            lines.append(f"  {k}={v}")
    else:
        lines.append("env knobs set : (none)")
    return "\n".join(lines)


def run_doctor(args: argparse.Namespace) -> int:
    """``doctor`` subcommand handler.

    Returns 0 when healthy. Returns 1 when ``--config`` is given and fails to
    load, or when ``--require-cuda`` is set and CUDA is unavailable — so the
    command doubles as a cluster pre-flight gate (``spectramr doctor --require-cuda``).
    """
    diag = collect_diagnostics()

    if getattr(args, "json", False):
        import json

        print(json.dumps(diag, indent=2))
    else:
        print(render_diagnostics(diag))

    rc = 0

    config_path = getattr(args, "config", None)
    if config_path is not None:
        from spectramr.config.settings import TrainingSettings

        if not Path(config_path).exists():
            print(f"\n[config] MISSING: {config_path}")
            rc = 1
        else:
            try:
                TrainingSettings.from_yaml(str(config_path))
                print(f"\n[config] OK: {config_path} loads against the schema.")
            except Exception as exc:
                print(f"\n[config] INVALID: {config_path}\n  {exc}")
                rc = 1

    if getattr(args, "require_cuda", False) and not diag["torch"].get("cuda_available"):
        print("\n[require-cuda] FAILED: no CUDA device visible.")
        rc = 1

    return rc
