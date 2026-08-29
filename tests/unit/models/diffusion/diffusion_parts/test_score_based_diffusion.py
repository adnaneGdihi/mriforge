"""Tests for score_based_diffusion (WS-6).

Two regressions guarded here:

1. ``p_losses`` silently defaulted any unknown ``loss_type`` to MSE
   (pitfall #9); the dispatch is now registry-routed and raises.
2. The module-scope ``from mriforge.core.metrics import get_metric`` created
   an import cycle (core.metrics walk-discovery → models.losses →
   models.diffusion → this module → partially-initialized core.metrics)
   that aborted the losses-package init, silently leaving the loss registry
   half-populated (l1/l2 unresolvable) for the whole process. The import is
   now lazy; importing this module must not pull in core.metrics.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from mriforge.models.diffusion.diffusion_parts.score_based_diffusion import (  # noqa: E402
    ScoreBasedDiffusion,
)


class _HalfInput(torch.nn.Module):
    def forward(self, x, t):
        return 0.5 * x


def test_p_losses_unknown_loss_type_raises() -> None:
    sbd = ScoreBasedDiffusion(timesteps=8, device="cpu")
    x_start = torch.randn(1, 1, 8, 8)
    t = torch.tensor([2])
    with pytest.raises(ValueError, match="Unsupported loss type"):
        sbd.p_losses(_HalfInput(), x_start, t, noise=torch.randn_like(x_start), loss_type="nope")


def test_p_losses_computes_finite_composite() -> None:
    sbd = ScoreBasedDiffusion(timesteps=8, device="cpu")
    torch.manual_seed(0)
    x_start = torch.randn(2, 1, 16, 16)
    t = torch.tensor([1, 5])
    loss = sbd.p_losses(_HalfInput(), x_start, t, noise=torch.randn_like(x_start), loss_type="l1")
    assert torch.isfinite(loss)


def test_import_after_core_metrics_keeps_loss_registry_complete() -> None:
    """The get_metric import must stay lazy (cycle guard, see module
    docstring). Pin the poisoning order: core.metrics first, then this
    module, then the elementary losses must still resolve — with the old
    eager import this order left ``l1`` unresolvable.

    (An assertion that importing this module pulls no core.metrics at all is
    NOT possible: ``mriforge.models.diffusion.__init__`` loads core.metrics
    through other modules; only this module's import had the cycle-forming
    package-attribute form.)
    """
    code = (
        "import mriforge.core.metrics\n"
        "import mriforge.models.diffusion.diffusion_parts.score_based_diffusion\n"
        "from mriforge.models.losses.elementary import resolve_elementary_loss\n"
        "for name in ('l1', 'l2', 'huber'):\n"
        "    resolve_elementary_loss(name)\n"
    )
    env = {**os.environ, "MRIFORGE_SUPPRESS_CLINICAL_WARNING": "1"}
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
