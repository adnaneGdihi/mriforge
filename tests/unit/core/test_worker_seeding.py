"""Unit tests for the canonical per-worker seeding hook (WS5).

``seed_worker`` used to be copy-pasted three times (two live builders + a dead
``accelerator`` copy). It now lives once in ``core.worker_seeding`` and is
imported rightward by both the ``data`` and ``infrastructure`` layers. These
tests pin the reproducibility contract and guard against a re-divergence into
per-layer copies.
"""

import random

import numpy as np
import torch

from mriforge.core.worker_seeding import seed_worker


def _draw_after_seed(worker_id: int, torch_seed: int = 0):
    torch.manual_seed(torch_seed)
    seed_worker(worker_id)
    return float(np.random.rand()), random.random()


class TestReproducibility:
    def test_same_torch_seed_reproduces_numpy_and_random(self):
        assert _draw_after_seed(0) == _draw_after_seed(0)

    def test_different_torch_seed_decorrelates(self):
        torch.manual_seed(1)
        a = _draw_after_seed(0, torch_seed=1)
        b = _draw_after_seed(0, torch_seed=2)
        assert a != b

    def test_does_not_touch_torch_rng(self):
        # seed_worker must NOT call torch.manual_seed (that would fight
        # initialize_accelerator); it only mirrors torch's seed onto np/random.
        torch.manual_seed(123)
        before = torch.rand(1).item()
        torch.manual_seed(123)
        seed_worker(0)
        after = torch.rand(1).item()
        assert before == after


class TestSingleSourceOfTruth:
    def test_all_layers_share_one_object(self):
        """Every consumer must reference the ONE core hook — a re-introduced
        local copy would make these identity checks fail."""
        from mriforge.accelerator import seed_worker as acc
        from mriforge.data.builders.torchio_queue_builder import seed_worker as data_q
        from mriforge.infrastructure.builders.leaf.data_builders import (
            seed_worker as infra,
        )

        assert seed_worker is acc
        assert seed_worker is data_q
        assert seed_worker is infra
