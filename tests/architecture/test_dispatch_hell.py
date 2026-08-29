"""Fitness function: no NEW string-literal component dispatch outside registries.

Cached-cascade Phase 0. Flags ``if x == "a" / elif x == "b" / ...`` dispatch
chains (>=3 string branches on one target) that select a component outside a
registry/factory home — these should route through a registry (``create_loss`` /
``create_optimizer`` / ``SCHEDULER_REGISTRY`` / ...). Ratcheted: existing sites
(e.g. the four diffusion loss ``if/elif`` blocks + ``create_single_optimizer``)
are baselined; only NEW chains fail. WS-D Tier-1 converts the baselined sites
and tightens the baseline.
"""

from __future__ import annotations

import pytest

from ._fitness_lib import ratchet, scan_dispatch_hell

pytestmark = pytest.mark.architecture


def test_no_new_string_dispatch_chains() -> None:
    current = scan_dispatch_hell(min_branches=3)
    new = ratchet(
        "dispatch_hell.txt",
        current,
        header="String-literal dispatch chains (>=3 branches) outside registry homes",
    )
    assert not new, (
        "New string-literal component dispatch detected — route it through a "
        "registry (create_loss/create_optimizer/SCHEDULER_REGISTRY) instead of "
        "an if/elif chain:\n  " + "\n  ".join(sorted(new))
    )
