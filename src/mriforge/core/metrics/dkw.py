r"""Dvoretzky-Kiefer-Wolfowitz slack — the SSOT for the DKW finite-sample band.

``dkw_slack`` used to live in ``infrastructure/calibration/chd.py``. Its one
``core/`` consumer (:mod:`mriforge.core.metrics.trajectory_metrics`) reached
upward for it with a function-local import, which is a non-negotiable 5
violation that ``scripts/ci/check_layering.sh`` cannot see (every grep there is
``^``-anchored, so an indented import is invisible) and which kept
``tests/architecture/test_layer_direction.py`` red — see issue #1183.

The function is twelve lines of pure :mod:`math` with no torch, no IO and no
configuration, so it belongs in ``core/`` at source rather than in an
allow-list. ``chd`` re-exports it, so every existing import path keeps working
and there is exactly one implementation (non-negotiable 17).

The inverse, ``StatisticalTests.dkw_required_n``, lives in
:mod:`mriforge.core.metrics.statistical_tests` and is round-tripped against this
function by ``tests/unit/core/metrics/test_statistical_tests.py``.
"""

from __future__ import annotations

import math

__all__ = ["dkw_slack"]


def dkw_slack(n: int, delta: float) -> float:
    r"""DKW slack term :math:`\sqrt{\ln(2/\delta) / (2 n)}`.

    Args:
        n: Number of calibration samples.
        delta: Confidence parameter ``δ ∈ (0, 1)``.

    Raises:
        ValueError: ``n ≤ 0`` or ``δ`` outside ``(0, 1)``.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0; got {n}")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must lie in (0, 1); got {delta}")
    return math.sqrt(math.log(2.0 / delta) / (2.0 * n))
