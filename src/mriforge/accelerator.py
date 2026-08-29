import logging
import os
import random

import numpy as np
import torch

from mriforge.core.compute_device import resolve_torch_device
from mriforge.core.worker_seeding import seed_worker
from mriforge.infrastructure.config.env_resolver import configure_cache_environment

logger = logging.getLogger(__name__)

# ``seed_worker`` is the canonical per-worker RNG hook, now homed in
# ``core.worker_seeding`` so ``data`` and ``infrastructure`` share one copy. It
# is re-exported here for the historical ``mriforge.accelerator.seed_worker``
# import path.
__all__ = ["seed_worker"]


def seed_everything(
    seed: int,
    *,
    rank: int = 0,
    deterministic: bool = True,
    allow_tf32: bool = True,
) -> int:
    """Seed torch / numpy / random (+ CUDA) and apply the determinism policy.

    Single Source of Truth for process-level seeding, shared by BOTH
    :func:`initialize_accelerator` (the canonical CLI entry point) and the
    training pipeline's seed call (``shared.utils.seed_control.set_global_seed``
    delegates here). Previously the pipeline re-seeded WITHOUT
    ``use_deterministic_algorithms`` or the DDP rank offset, silently diverging
    from the accelerator's policy — this collapses both onto one implementation.

    Args:
        seed: Base seed.
        rank: DDP rank. The effective seed is ``seed + rank`` so each rank draws
            different data augmentations, while weight init (seeded before the
            rank split, or with the same seed across ranks) stays shared.
        deterministic: True → cuDNN ``benchmark=False`` / ``deterministic=True``
            plus ``torch.use_deterministic_algorithms(warn_only=True)``. False →
            enable the cuDNN autotuner (``benchmark=True``) for speed. Never both
            at once (a contradiction: the autotuner re-introduces the run-to-run
            variance the determinism flag removes).
        allow_tf32: TensorFloat-32 for cuDNN convolutions and fp32 matmuls on
            Ampere+ — a large fp32 throughput win for ~10 bits of mantissa.
            Defaults to True, which is the behaviour this codebase already had;
            it is a parameter here so it has ONE owner and a logged value.

            It previously had none. ``training_utils.initialize_device`` set
            ``matmul.allow_tf32 = True`` *and* ``cudnn.benchmark = True``
            unconditionally, making it a second writer to globals this function
            owns — and the two disagreed about ``benchmark``. Because that
            function is reached lazily through ``_DEFAULT_DEVICE``, whichever
            ran last won, so a run that asked for determinism could be silently
            un-determinized. Those two lines are gone; this is the only writer.

            Note TF32 is *not* the same axis as ``deterministic``: TF32 is
            reproducible, merely lower-precision. It is kept separate so an arm
            can have bit-for-bit reproducibility and TF32 throughput at once.
            ``SafeComplexAutocast`` still disables TF32 for the duration of a
            complex autocast block and restores it on exit — a scoped override
            of this global, not a competing owner.

    Returns:
        The effective (rank-offset) seed that was applied.
    """
    worker_seed = seed + rank
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    if torch.cuda.is_available():
        # ``manual_seed_all`` covers every visible CUDA device; harmless when the
        # run is CPU-only-in-practice but CUDA happens to be importable.
        torch.cuda.manual_seed_all(worker_seed)

    # Determinism policy is device-agnostic: CPU ops (scatter / index reductions,
    # some pooling backward passes) also have non-deterministic variants, so the
    # policy cannot live behind a cuda-available gate. A single coherent choice.
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # warn_only: a non-deterministic op warns instead of hard-failing.
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # Orthogonal to the determinism policy above: TF32 is reproducible, just
    # lower-precision. Set unconditionally so the value is never left at
    # whatever a previous process-level writer happened to leave it at.
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    logger.info(
        "[Accelerator] determinism=%s cudnn.benchmark=%s allow_tf32=%s",
        deterministic,
        torch.backends.cudnn.benchmark,
        allow_tf32,
    )

    return worker_seed


def initialize_accelerator(
    device_str: str | None,
    seed: int | None = None,
    rank: int = 0,
    deterministic: bool = True,
    *,
    pipeline: str = "train",
) -> torch.device:
    """
    Initialize the accelerator (GPU/CPU), set seeds, and configure precision.

    Device resolution is delegated to the SSOT policy in
    :mod:`mriforge.core.compute_device`: a heavy pipeline that cannot get an
    accelerator RAISES ``AcceleratorRequiredError`` rather than degrading to
    CPU. The old "CUDA not available. Falling back to CPU." branch is gone — it
    turned a broken GPU allocation into a silent ~100x-slower run that still
    reported success.

    Args:
        device_str: Device string ('cuda', 'cpu', 'cuda:0', 'auto', or None→auto)
        seed: Random seed for reproducibility. Defaults to 42 if None.
        rank: DDP rank for multi-GPU training (default 0 for single GPU)
        pipeline: Caller name; the accelerated-run contract applies when it is
            in ``HEAVY_PIPELINES``.
        deterministic: If True (default), configure cuDNN for bit-for-bit
            reproducibility (``deterministic=True``, ``benchmark=False``, plus
            ``torch.use_deterministic_algorithms(warn_only=True)``). If False,
            enable the cuDNN autotuner (``benchmark=True``) for speed at the
            cost of run-to-run reproducibility. The previous SSOT set BOTH
            ``benchmark=True`` and ``deterministic=True`` — a contradiction
            (the autotuner re-introduces run-to-run variance the determinism
            flag is meant to remove).

    Returns:
        torch.device object

    Note:
        For DDP training, each rank gets a seed offset (seed + rank) to ensure
        different data augmentations across GPUs. Model weights should be
        initialized BEFORE calling this with rank > 0, or use the same seed
        across ranks for weight init.
    """
    # Use default seed if None
    if seed is None:
        seed = 42

    # Env-var fallbacks for callers that didn't pre-set them at module
    # import time (the canonical pre-set lives in src/main.py before
    # ``import torch``, which is too late by the time we get here).
    # Keep the alloc-config string in lock-step with main.py's — see
    # ``TODO/audit/15_entry_points_bootstrap_cli.md`` F13.
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:512",
    )

    # Configure the library caches for cluster/local environments through the
    # shared SSOT, so this entry point gets the SAME set main.py gets.
    #
    # This call site used to set three of the six by hand — TMPDIR, TORCH_HOME,
    # CUDA_CACHE_CONFIG — and it is the ONLY one that runs on the documented
    # multi-GPU path (`torchrun -m mriforge.cli train-distributed` never imports
    # main.py). The three it omitted included both caches DeepSpeed writes during
    # its own import: TRITON_CACHE_DIR (Triton ignores XDG_CACHE_HOME and defaults
    # to ~/.triton) and XDG_CACHE_HOME itself (torch's JIT extension build root
    # follows it, else ~/.cache/torch_extensions). So the cluster fix that added
    # TRITON_CACHE_DIR to main.py never applied to a distributed run, and
    # `import deepspeed` kept writing into a cluster $HOME — an opaque
    # PermissionError/OSError out of a third-party import, before any config is
    # read or any device is selected.
    configure_cache_environment()

    # DDP-aware seeding + determinism policy via the shared SSOT, so the
    # training pipeline's seed call (set_global_seed → seed_everything) cannot
    # drift from this canonical entry point.
    worker_seed = seed_everything(seed, rank=rank, deterministic=deterministic)

    # Device resolution + the accelerated-run contract live in ONE place.
    # Raises AcceleratorRequiredError instead of degrading to CPU.
    decision = resolve_torch_device(device_str, pipeline=pipeline, source="cli/config")

    logger.info(
        "[DEVICE] %s -> %s (accelerated=%s, source=%s) | Rank: %s, Seed: %s, Deterministic: %s",
        pipeline,
        decision.device,
        decision.accelerated,
        decision.source,
        rank,
        worker_seed,
        deterministic,
    )
    if not decision.accelerated and torch.cuda.is_available():
        logger.warning(
            "[DEVICE] CUDA is available but this %s run was pinned to CPU by "
            "explicit request — the GPU will sit idle.",
            pipeline,
        )

    return torch.device(decision.device)
