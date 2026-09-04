"""Per-DataLoader-worker RNG seeding — the single canonical hook.

Lives in ``core`` (the rightmost layer) so that both the data-loading SSOT
layer (``data/``) and ``infrastructure/`` import it *rightward* — neither
duplicates the body, and ``data`` avoids a forbidden leftward dependency on
``infrastructure``. This replaces three byte-identical copies that had drifted
apart in docstring only (``data/builders/torchio_queue_builder``,
``infrastructure/builders/leaf/data_builders``, and the dead
``accelerator.seed_worker``).
"""

from __future__ import annotations

__all__ = ["seed_worker"]


def seed_worker(worker_id: int) -> None:
    """Seed NumPy + stdlib ``random`` per DataLoader worker from torch's seed.

    PyTorch seeds each worker's *torch* RNG from ``base_seed + worker_id`` but
    leaves NumPy's and the stdlib ``random`` generators untouched, so a transform
    that uses ``np.random`` / ``random`` draws the *same* values in every worker.
    This forwards torch's own per-worker seed (``torch.initial_seed()``) to both,
    keeping draws distinct across workers yet reproducible across runs. It does
    NOT call ``torch.manual_seed`` (which would fight ``initialize_accelerator``);
    it only mirrors torch's per-worker seed onto the other two RNGs.

    Use as ``DataLoader(..., worker_init_fn=seed_worker)``. ``worker_id`` is part
    of the DataLoader hook signature and intentionally unused (the seed is read
    from ``torch.initial_seed()``, which already encodes the worker offset).
    """
    import random

    import numpy as np
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
