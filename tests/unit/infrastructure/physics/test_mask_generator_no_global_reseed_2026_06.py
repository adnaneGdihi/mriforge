"""Regression test for the 2026-06 audit fix to ``MaskGenerator``.

``MaskGenerator.__init__(seed=...)`` used to reseed the *global* ``random`` /
``numpy`` / ``torch`` RNGs. Built inside the per-step SSL-pretraining loss path
(``MaskGenerator(seed=epoch)``), that froze the global RNG to ``epoch`` every
step and clobbered dataloader-shuffle / dropout entropy. The reseed is removed;
masks stay reproducible via the local ``self.seed``-based RNGs.
"""
from __future__ import annotations

import numpy as np
import torch

from mriforge.infrastructure.physics.sampling import MaskGenerator


def test_construction_does_not_touch_global_rng() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    torch_before = torch.random.get_rng_state().clone()
    np_before = np.random.get_state()

    MaskGenerator(seed=12345)  # must NOT reseed the global generators

    assert torch.equal(torch.random.get_rng_state(), torch_before)
    # numpy global state unchanged (compare the 624-word MT19937 key array)
    assert np.array_equal(np.random.get_state()[1], np_before[1])


def test_masks_still_reproducible_via_local_seed() -> None:
    shape = (32, 32)
    m1 = MaskGenerator(seed=42).generate_mask("random", shape, acceleration_factor=4.0)
    m2 = MaskGenerator(seed=42).generate_mask("random", shape, acceleration_factor=4.0)
    assert torch.equal(m1, m2)
