"""Compute-device SSOT — the accelerated-run contract.

Single source of truth for turning a device *spec* (``--device``, YAML
``device:``, ``SPECTRAMR_DEVICE``) into a concrete device, and for enforcing the
project rule:

    **Heavy pipelines run on an accelerator. If none is available, they RAISE
    — they never silently degrade to CPU.**

"Heavy" is anything that consumes real GPU-hours: training, inference,
validation, HPO, ablation, benchmarking, distributed runs, campaigns, and the
Tier-2 audit probe (:data:`HEAVY_PIPELINES`).

Why this exists: eight call sites each grew their own ``if not
torch.cuda.is_available(): device = "cpu"`` fallback. A cluster job that landed
on a GPU-less node, or whose CUDA driver faulted, would quietly run ~100x slower
on CPU and report success — burning the wall-clock allocation while looking
green. That is pitfall #9 (silent fallback) at the *hardware* layer.

The user may still dictate CPU — deliberately, never by accident:

* ``--device cpu`` on the CLI,
* ``device: cpu`` in the experiment YAML,
* ``SPECTRAMR_DEVICE=cpu``,
* ``FORCE_CPU=true`` in the environment (the CI / co-simulation escape hatch).

An explicit ``cuda`` request on a host without CUDA always raises, even under
``FORCE_CPU`` — that is a broken environment, not a preference.

The policy is a **pure function** of ``(requested, pipeline, cuda_available,
mps_available, environ)`` so every branch is unit-testable without a GPU. Torch
is imported lazily, only by the :func:`resolve_torch_device` shell.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "HEAVY_PIPELINES",
    "AcceleratorRequiredError",
    "DeviceDecision",
    "cpu_opt_in_from_env",
    "resolve_compute_device",
    "resolve_torch_device",
]


#: Pipelines that consume real GPU-hours and must therefore run accelerated.
#: A caller not in this set (a config lint, a manifest build) may run on CPU.
HEAVY_PIPELINES: frozenset[str] = frozenset(
    {
        "train",
        "sanity_check",
        "experiment",
        "infer",
        "infer_dataset",
        "predict",
        "validate",
        "evaluate",
        "hpo",
        "ablation",
        "benchmark",
        "distributed",
        "campaign",
        "probe",
        # PGGS reconstruction (Gaussian splatting + latent diffusion + test-time
        # optimisation) runs models; ungated it fell back to CPU without an opt-in.
        "pggs",
    }
)

#: Env knob: explicitly accept a CPU run for a heavy pipeline. Already
#: registered in ``spectramr.core.env``; consumed here and by ``initialize_device``.
FORCE_CPU_ENV = "FORCE_CPU"

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off", ""})


class AcceleratorRequiredError(RuntimeError):
    """A heavy pipeline was asked to run without an accelerator.

    Raised instead of degrading to CPU. See the module docstring for the
    sanctioned ways to opt into a CPU run.
    """


@dataclass(frozen=True)
class DeviceDecision:
    """The resolved device plus the provenance of *why* (pitfall #15c)."""

    device: str  # concrete: "cuda", "cuda:1", "mps", "cpu"
    accelerated: bool  # False only for cpu
    source: str  # "cli" | "config.device" | "env:FORCE_CPU" | ...
    cpu_opt_in: bool  # True when the user explicitly dictated CPU

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def cpu_opt_in_from_env(environ: Mapping[str, str] | None = None) -> bool:
    """Read + validate ``FORCE_CPU``. Raises on an unrecognized value.

    Pitfall #15: an advertised knob is read, validated against the accepted
    set, and raises on anything else — it never degrades to a default.
    """
    env = os.environ if environ is None else environ
    raw = env.get(FORCE_CPU_ENV, "")
    value = str(raw).strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    raise ValueError(
        f"{FORCE_CPU_ENV}={raw!r} is not a recognized boolean. "
        f"Use one of {sorted(_TRUTHY)} or {sorted(_FALSY - {''})}."
    )


def _validate_cuda_spec(spec: str) -> None:
    """Reject ``cuda:x`` — a malformed index must not resolve to device 0."""
    if ":" not in spec:
        return
    _, _, index = spec.partition(":")
    if not index.isdigit():
        raise ValueError(f"Malformed CUDA device {spec!r}; expected 'cuda' or 'cuda:<int>'.")


def resolve_compute_device(
    requested: str | None,
    *,
    pipeline: str,
    cuda_available: bool,
    mps_available: bool = False,
    environ: Mapping[str, str] | None = None,
    source: str = "unspecified",
) -> DeviceDecision:
    """Resolve a device spec under the accelerated-run contract.

    Args:
        requested: ``"auto"`` / ``"cuda"`` / ``"cuda:N"`` / ``"mps"`` / ``"cpu"``.
            ``None`` or blank means ``"auto"``.
        pipeline: Caller name. Gated when in :data:`HEAVY_PIPELINES`.
        cuda_available: ``torch.cuda.is_available()`` — injected so the policy
            stays pure and testable on a CPU-only box.
        mps_available: ``torch.backends.mps.is_available()``.
        environ: Environment mapping (defaults to ``os.environ``).
        source: Where ``requested`` came from, stamped into the decision.

    Returns:
        The :class:`DeviceDecision`.

    Raises:
        AcceleratorRequiredError: A heavy pipeline would have run on CPU
            without the user having dictated it.
        ValueError: The spec is unrecognized or malformed.
    """
    spec = (requested or "auto").strip().lower() or "auto"
    heavy = pipeline in HEAVY_PIPELINES
    force_cpu = cpu_opt_in_from_env(environ)

    if spec == "cpu":
        # Explicit user dictation — allowed, but never quiet on a heavy run.
        if heavy:
            logger.warning(
                "[DEVICE] %s will run on CPU by explicit request (device=cpu). "
                "Expect a ~100x slowdown vs GPU; no AMP, no CUDA OOM coverage.",
                pipeline,
            )
        return DeviceDecision(device="cpu", accelerated=False, source=source, cpu_opt_in=True)

    if spec.startswith("cuda"):
        _validate_cuda_spec(spec)
        if not cuda_available:
            # Deliberate: NOT relaxed by FORCE_CPU. Asking for cuda on a host
            # with no cuda is a broken environment, not a preference.
            raise AcceleratorRequiredError(
                f"device={spec!r} was requested for the {pipeline!r} pipeline, "
                "but CUDA is not available on this host. Refusing to fall back "
                "to CPU: a silent CPU run burns the GPU allocation at ~100x "
                "slowdown while reporting success. Fix the environment (check "
                "the sbatch --gres=gpu request and CUDA_VISIBLE_DEVICES), or "
                "run on CPU deliberately with --device cpu (or device: cpu "
                "in the YAML / run.device: cpu). FORCE_CPU does NOT apply "
                "here: an explicit cuda request on a CUDA-less host is a "
                "broken environment, not a preference, so it is not "
                "relaxed by the env opt-in."
            )
        return DeviceDecision(device=spec, accelerated=True, source=source, cpu_opt_in=False)

    if spec == "mps":
        if not mps_available:
            raise AcceleratorRequiredError(
                f"device='mps' was requested for the {pipeline!r} pipeline, but "
                "MPS is not available on this host."
            )
        return DeviceDecision(device="mps", accelerated=True, source=source, cpu_opt_in=False)

    if spec != "auto":
        raise ValueError(
            f"device={requested!r} unrecognized. Choose from: auto, cpu, cuda, cuda:<idx>, mps."
        )

    # ── auto ─────────────────────────────────────────────────────────
    if cuda_available:
        return DeviceDecision(device="cuda", accelerated=True, source=source, cpu_opt_in=False)
    if mps_available:
        return DeviceDecision(device="mps", accelerated=True, source=source, cpu_opt_in=False)

    if heavy and not force_cpu:
        raise AcceleratorRequiredError(
            f"The {pipeline!r} pipeline requires an accelerator but no "
            "accelerator is available (no CUDA, no MPS). Refusing to fall back "
            "to CPU: a silent CPU run burns the wall-clock allocation at ~100x "
            "slowdown while reporting success. Either run on a GPU node, or "
            "opt into a CPU run deliberately with --device cpu (or "
            f"device: cpu in the YAML, or {FORCE_CPU_ENV}=true)."
        )

    if heavy:
        logger.warning(
            "[DEVICE] %s will run on CPU: no accelerator available and "
            "%s is set. Expect a ~100x slowdown.",
            pipeline,
            FORCE_CPU_ENV,
        )
    return DeviceDecision(
        device="cpu",
        accelerated=False,
        source=f"env:{FORCE_CPU_ENV}" if force_cpu else source,
        cpu_opt_in=force_cpu,
    )


def resolve_torch_device(
    requested: str | None,
    *,
    pipeline: str,
    source: str = "unspecified",
) -> DeviceDecision:
    """:func:`resolve_compute_device` bound to the live torch runtime.

    The imperative shell: queries torch for availability, then defers every
    decision to the pure policy above. Torch is imported lazily so importing
    this module stays cheap (see the entry-point startup-perf note).
    """
    import torch

    try:
        mps_available = bool(torch.backends.mps.is_available())
    except AttributeError:  # torch build without MPS
        mps_available = False

    return resolve_compute_device(
        requested,
        pipeline=pipeline,
        cuda_available=bool(torch.cuda.is_available()),
        mps_available=mps_available,
        source=source,
    )
