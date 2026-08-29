"""Node CPU/memory/GPU inventory — the resource half of the topology SSOT.

These probes used to live in ``infrastructure.logging.provenance``, where they
were written as *reporting* helpers: fail-open, best-effort, stamped into
``provenance.json`` and never consulted again. ``cpu_resources()`` even said in
its own docstring that ``usable_cores`` is "what dataloader-worker decisions
should be judged against" — and nothing judged anything against it. The worker
count was a flat read of the YAML, so a 4-rank job on a 16-core allocation
launched ``4 x num_workers`` decoder processes and thrashed.

They live in ``core`` now so the number stamped into provenance and the number
used to make a decision come from **one implementation**. ``core`` is the
rightmost layer, so both ``infrastructure/`` (provenance) and ``data/``
(dataloader construction) import them rightward, which is the legal direction.

Two contracts that pull in opposite directions, deliberately:

* **Reporting stays fail-open.** Every probe here swallows its own errors and
  returns ``None`` for what it could not learn. A missing ``psutil``, an
  unreadable cgroup file, or a non-Linux host must never abort a run.
* **Decisions must not.** A caller that *acts* on ``usable_cores`` has to treat
  ``None`` as an explicit branch, because silently substituting a default is
  non-negotiable 3. :func:`mriforge.core.topology.resolve_run_topology` is the
  one such caller; see the ``cpus_on_node is None`` handling there.

The GPU probes (:func:`visible_gpu_count`, :data:`ALLOC_GPU_ENV_PER_NODE`) were
added for the same reason the CPU ones moved here. ``SLURM_GPUS_ON_NODE`` had been
read in exactly one place -- ``provenance.gpu_resources``, as ``allocated_count``
-- and stamped beside the live world size in the same JSON record, where nothing
ever subtracted the two. A ``--gpus=4`` job therefore ran one rank for 41 minutes
with three cards idle and every layer of the package silent (#1274). The refusal
that consumes them lives at ``pipelines/distributed.py``, not here.

``usable_cores`` may be a **float**: :func:`cgroup_cpu_quota` reports fractional
cores (a ``cpu.max`` of ``150000 100000`` is 1.5 cores), and the ``min()`` that
picks the tightest constraint can therefore return one. Callers that need an
integer worker count must floor it themselves rather than assume an int.
"""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import re
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "ALLOC_CPU_ENV",
    "ALLOC_GPU_ENV",
    "ALLOC_GPU_ENV_JOB",
    "ALLOC_GPU_ENV_PER_NODE",
    "allocated_gpus_per_node",
    "cgroup_cpu_quota",
    "cgroup_memory_limit_gb",
    "cpu_model",
    "cpu_resources",
    "env_int",
    "env_int_source",
    "read_first_line",
    "visible_gpu_count",
]

#: Scheduler env vars carrying the CPU allocation, in priority order (per-task
#: first: that is this process's share, whereas ``*_ON_NODE`` covers every task).
ALLOC_CPU_ENV = ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "NSLOTS", "PBS_NP")

#: Scheduler env vars carrying the GPU grant **for one node**.
ALLOC_GPU_ENV_PER_NODE = ("SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE")
#: Scheduler env vars carrying the GPU grant **for the whole job**. ``--gpus=N``
#: populates only this one, and it is a job total: on a 2-node allocation
#: ``SLURM_GPUS=8`` means 4 per node. Comparing it to a per-node rank count
#: without dividing is the mistake that makes a correct multi-node run look
#: wrong, so the two tuples are kept apart rather than merged into one lookup.
ALLOC_GPU_ENV_JOB = ("SLURM_GPUS",)
#: Priority order for a bare "how many GPUs?" read, per-node first. Reporting
#: callers that do not care about the scope use this; callers that compare the
#: number against a rank count must use the two tuples above instead.
ALLOC_GPU_ENV = ALLOC_GPU_ENV_PER_NODE + ALLOC_GPU_ENV_JOB

_BYTES_PER_GB = 1024**3


def read_first_line(path: str) -> str | None:
    """First line of ``path``, or ``None`` if unreadable (fail-open)."""
    try:
        with open(path) as fh:
            return fh.readline().strip()
    except Exception:
        return None


def env_int_source(
    names: tuple[str, ...], environ: Mapping[str, str] | None = None
) -> tuple[int | None, str | None]:
    """First env var in ``names`` that parses as a positive int, **and its name**.

    The name matters wherever the number is later compared against something:
    ``SLURM_GPUS_ON_NODE`` and ``SLURM_GPUS`` are both "the GPU count" but one is
    per node and one is per job, so a caller that does not know which it got
    cannot know what it may compare it to. :func:`env_int` is the same lookup for
    callers that only need the value.

    Args:
        names: Variable names to try, in priority order.
        environ: Environment mapping (defaults to ``os.environ``). Injected so
            topology resolution stays a pure function of its inputs.

    Returns:
        ``(value, variable_name)``, or ``(None, None)`` when nothing parsed.
    """
    env = os.environ if environ is None else environ
    for key in names:
        raw = (env.get(key) or "").strip()
        # SLURM_CPUS_ON_NODE can be "72(x2)" on heterogeneous allocations.
        match = re.match(r"^(\d+)", raw)
        if match:
            val = int(match.group(1))
            if val > 0:
                return val, key
    return None, None


def env_int(names: tuple[str, ...], environ: Mapping[str, str] | None = None) -> int | None:
    """First env var in ``names`` that parses as a positive int.

    Args:
        names: Variable names to try, in priority order.
        environ: Environment mapping (defaults to ``os.environ``). Injected so
            topology resolution stays a pure function of its inputs.
    """
    return env_int_source(names, environ)[0]


def allocated_gpus_per_node(
    num_nodes: int, environ: Mapping[str, str] | None = None
) -> tuple[int | None, str]:
    """The scheduler's GPU grant **for one node**, and where the number came from.

    Two consumers need this and must not disagree:
    :func:`mriforge.core.topology.resolve_run_topology`, which stamps it into
    provenance, and ``pipelines.distributed.run_distributed_training``, which
    refuses a launch that would waste it. The launcher deliberately calls this
    rather than resolving a whole topology -- it already knows its own rank
    counts, and re-validating the environment there would make an unrelated
    misconfiguration fail in a new place.

    ``--gpus=N`` populates only ``SLURM_GPUS``, which is a **job total**. On a
    2-node job ``SLURM_GPUS=8`` means 4 per node, and comparing the raw 8 against
    one node's rank count is precisely the per-node-vs-global confusion that makes
    a correct multi-node run look broken. So the total is divided -- and when it
    does not divide, the allocation is heterogeneous and no per-node number is
    honest, so ``None`` is returned rather than a rounded guess (non-negotiable 3:
    a plausible substitute is worse than an admitted unknown).

    Args:
        num_nodes: Nodes in this job, used only to divide a job total.
        environ: Environment mapping (defaults to ``os.environ``).

    Returns:
        ``(gpus_on_this_node, source)``. ``source`` is always a non-empty string
        naming the variable, the derivation, or why the answer is ``None``.
    """
    per_node, var = env_int_source(ALLOC_GPU_ENV_PER_NODE, environ)
    if per_node is not None:
        return per_node, f"env:{var}"

    job_total, var = env_int_source(ALLOC_GPU_ENV_JOB, environ)
    if job_total is None:
        return None, "unset:no-scheduler-gpu-grant"
    nodes = max(1, num_nodes)
    if job_total % nodes == 0:
        return job_total // nodes, f"derived:{var}/{nodes}nodes"
    logger.warning(
        "%s=%d does not divide by num_nodes=%d, so the per-node GPU grant cannot"
        " be derived. Leaving it unknown rather than rounding: the allocation is"
        " probably heterogeneous, and a plausible-looking per-node count is worse"
        " than none.",
        var,
        job_total,
        nodes,
    )
    return None, f"undividable:{var}={job_total}"


def visible_gpu_count() -> int | None:
    """How many CUDA devices *this process* can address, or ``None`` if unknown.

    **Visible, not allocated.** The two coincide on a plain ``--gpus=N`` job and
    diverge the moment ``CUDA_VISIBLE_DEVICES`` masks something -- which is a
    legitimate pattern (``srun --gpu-bind`` hands each task one device out of the
    node's four), so a caller deciding whether devices are going to waste has to
    look at both. The allocation side is :data:`ALLOC_GPU_ENV_PER_NODE` /
    :data:`ALLOC_GPU_ENV_JOB`.

    Fail-open per this module's reporting contract: a missing torch, a broken
    driver or a CPU-only host yields ``None``, never an exception and never a
    ``0`` standing in for "could not tell". ``0`` is returned only when torch
    answered and the answer was genuinely zero.

    Torch is imported lazily, exactly as :mod:`mriforge.core.compute_device` does
    it, so importing ``core`` stays cheap and torch-free.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except Exception:  # pragma: no cover - torch absent / driver broken
        logger.debug("visible GPU probe failed", exc_info=True)
        return None


def cgroup_cpu_quota() -> float | None:
    """Container CPU limit in whole cores, or ``None`` when unlimited/absent.

    Docker/k8s (and the ``mriforge launch`` docker backend) cap CPU via a cgroup
    quota that ``psutil`` cannot see — it still reports every host core.

    May return a fractional value (``150000 100000`` is 1.5 cores).
    """
    v2 = read_first_line("/sys/fs/cgroup/cpu.max")  # "<quota|max> <period>"
    if v2:
        parts = v2.split()
        if len(parts) == 2 and parts[0] != "max":
            try:
                quota, period = int(parts[0]), int(parts[1])
                if quota > 0 and period > 0:
                    return round(quota / period, 2)
            except ValueError:
                pass
    quota_s = read_first_line("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period_s = read_first_line("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    try:
        quota, period = int(quota_s or -1), int(period_s or -1)
    except ValueError:
        return None
    if quota > 0 and period > 0:
        return round(quota / period, 2)
    return None


def cgroup_memory_limit_gb() -> float | None:
    """Container memory ceiling in GB, or ``None`` when unlimited/absent.

    Same trap as the CPU quota: inside a container ``psutil.virtual_memory()``
    reports the *host's* RAM, so a run that OOM'd at 8 GB looks like it had 1 TB.
    """
    for path in (
        "/sys/fs/cgroup/memory.max",  # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
    ):
        raw = read_first_line(path)
        if not raw or raw == "max":
            continue
        try:
            limit = int(raw)
        except ValueError:
            continue
        # v1 encodes "unlimited" as a near-2^63 sentinel; treat >1 PB as unset.
        if 0 < limit < 1024**5:
            return round(limit / _BYTES_PER_GB, 2)
    return None


def cpu_model() -> str | None:
    """Human-readable CPU model (``/proc/cpuinfo`` first, then ``platform``)."""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or None


def cpu_resources(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Node CPU inventory **and** the slice this process may actually use.

    The distinction is load-bearing on a cluster: a 128-core node where the job
    was allocated 8 cores must not read as a 128-core run. ``usable_cores`` is
    the honest number (affinity mask n scheduler allocation n cgroup quota) and
    is what dataloader-worker decisions are judged against.

    Args:
        environ: Environment mapping (defaults to ``os.environ``).

    Returns:
        A dict whose ``usable_cores`` is ``None`` when nothing could be probed.
        It may be a **float** when a fractional cgroup quota is the tightest
        constraint — floor it before using it as a process count.
    """
    info: dict[str, Any] = {
        "model": None,
        "physical_cores": None,
        "logical_cores": os.cpu_count(),
        "affinity_cores": None,
        "allocated_cores": env_int(ALLOC_CPU_ENV, environ),
        "cgroup_quota_cores": cgroup_cpu_quota(),
        "usable_cores": None,
        "max_frequency_mhz": None,
    }
    with contextlib.suppress(Exception):  # pragma: no cover - defensive
        info["model"] = cpu_model()
    try:
        import psutil

        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["logical_cores"] = psutil.cpu_count(logical=True) or os.cpu_count()
        freq = psutil.cpu_freq()
        if freq is not None and freq.max:
            info["max_frequency_mhz"] = round(float(freq.max), 1)
    except Exception:  # psutil absent/unsupported -> keep the os.cpu_count view
        logger.debug("psutil cpu probe failed", exc_info=True)
    # The affinity mask is what the kernel will actually schedule us on — cgroup
    # cpusets and `taskset` both show up here, `os.cpu_count()` does not.
    with contextlib.suppress(AttributeError, OSError):  # non-Linux / unsupported
        info["affinity_cores"] = len(os.sched_getaffinity(0))
    candidates = [
        info["affinity_cores"],
        info["allocated_cores"],
        info["cgroup_quota_cores"],
        info["logical_cores"],
    ]
    usable = [c for c in candidates if c]
    if usable:
        info["usable_cores"] = min(usable)
    return info
