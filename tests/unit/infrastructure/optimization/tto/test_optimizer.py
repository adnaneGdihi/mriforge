"""Tests for ``TestTimeOptimizer`` (TTO refine loop).

Covers the compute-hoist regression: the k-space coordinate grid is a pure
function of the (constant) measured-k-space shape and trajectory, so it must
be built ONCE per ``refine`` call, not rebuilt on every optimization step.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from spectramr.infrastructure.optimization.tto.optimizer import TestTimeOptimizer


def _make_optimizer_with_fakes(num_steps: int) -> TestTimeOptimizer:
    """Build a TTO with real regularizer (lambdas=0) but faked IO collaborators."""
    opt = TestTimeOptimizer(
        projector=MagicMock(),  # not an nn.Module → no .to() call
        num_steps=num_steps,
        lambda_sparsity=0.0,  # all regularizers off → empty loss_dict, no syncs
    )

    # A single optimizable leaf, disconnected from the (faked) render graph —
    # the loop must still run num_steps iterations without error.
    leaf = torch.zeros(4, 3, requires_grad=True)
    params = MagicMock()
    params.is_valid.return_value = True
    params.to_list.return_value = [leaf]
    params.means = torch.zeros(4, 3)  # dim()==2 path
    params.opacity = None
    params.features = None

    opt.param_extractor = MagicMock()
    opt.param_extractor.extract.return_value = params

    opt.renderer = MagicMock()
    opt.renderer.render.return_value = torch.zeros(1, 1, 8, 8, requires_grad=True)
    opt.renderer.match_target_shape.side_effect = lambda pred, _tgt: pred
    opt.renderer._prepare_kspace_coords.return_value = torch.zeros(64, 3)
    return opt


def test_kspace_coords_built_once_not_per_step() -> None:
    """The constant coordinate grid is prepared exactly once per refine().

    Regression: it used to be rebuilt inside the per-step loop, an O(steps)
    meshgrid waste on a value that never changes.
    """
    num_steps = 5
    opt = _make_optimizer_with_fakes(num_steps)
    kspace = torch.zeros(1, 1, 8, 8)

    with patch(
        "spectramr.infrastructure.optimization.tto.optimizer.GaussianUpdater.update",
        return_value="refined",
    ):
        _refined, history = opt.refine(MagicMock(), kspace)

    assert opt.renderer._prepare_kspace_coords.call_count == 1
    # The loop still ran every step (loss recorded per iteration).
    assert len(history) == num_steps
