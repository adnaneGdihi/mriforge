"""``SobolevKSpaceLoss`` must expose the order its own docstring advertises.

The class documents ``(1 + k_x^2 + k_y^2)^{s/2}`` and hardcoded the ``s=2`` case
as the literal ``1.0 + xx**2 + yy**2``. 56 exp_11 arms declared ``sobolev_order:
1`` in a block ``extra="ignore"`` discarded, so the order was dead at every
layer and nobody could tell (#560, #615).

Note ``sobolev_frequency`` is a *separate* registered loss whose weight is
``(1 + |k|)`` -- the two are two fixed points of one family, which is why the
arms' declared intent was expressible all along, just not that way.
"""

import torch

from spectramr.models.losses.kspace_physics_losses import SobolevKSpaceLoss


def _expected_weight(size: int, order: float) -> torch.Tensor:
    y = (torch.arange(size, dtype=torch.float32) - size / 2.0) / (size / 2.0)
    yy, xx = torch.meshgrid(y, y, indexing="ij")
    return ((1.0 + xx**2 + yy**2) ** (order / 2.0)).unsqueeze(0).unsqueeze(0)


def test_default_order_reproduces_the_historical_hardcoded_weight():
    """s=2 is what the body computed as `1.0 + xx**2 + yy**2`.

    The default must not move, or every arm already using this loss silently
    changes objective.
    """
    loss = SobolevKSpaceLoss()
    assert loss.order == 2.0
    w = loss._get_weight_map(8, 8, torch.device("cpu"), torch.float32)
    torch.testing.assert_close(w, _expected_weight(8, 2.0))


def test_order_one_gives_the_square_root_weighting():
    loss = SobolevKSpaceLoss(order=1.0)
    w = loss._get_weight_map(8, 8, torch.device("cpu"), torch.float32)
    torch.testing.assert_close(w, _expected_weight(8, 1.0))


def test_order_zero_is_the_unweighted_l1_consistency_check():
    """At s=0 the weight map is all ones, so the loss degenerates to plain L1.

    Probing at the parameter value where both sides must agree, rather than at
    the operating point.
    """
    loss = SobolevKSpaceLoss(order=0.0)
    w = loss._get_weight_map(8, 8, torch.device("cpu"), torch.float32)
    torch.testing.assert_close(w, torch.ones_like(w))


def test_order_changes_the_weight_map_so_it_is_not_a_dead_knob():
    a = SobolevKSpaceLoss(order=2.0)._get_weight_map(
        8, 8, torch.device("cpu"), torch.float32
    )
    b = SobolevKSpaceLoss(order=1.0)._get_weight_map(
        8, 8, torch.device("cpu"), torch.float32
    )
    assert not torch.allclose(a, b)


def test_shape_only_cache_key_is_sufficient_because_order_is_per_instance():
    """The cache key is ``(height, width)``, which no longer uniquely determines
    the map now that ``order`` varies -- it is safe only because ``order`` is
    fixed per instance. Pin that, so making ``order`` mutable fails here rather
    than silently serving a stale map.
    """
    loss = SobolevKSpaceLoss(order=1.0)
    first = loss._get_weight_map(8, 8, torch.device("cpu"), torch.float32)
    assert set(loss._weight_map_cache) == {(8, 8)}

    loss.order = 2.0  # not supported; the cache must not silently honour it
    second = loss._get_weight_map(8, 8, torch.device("cpu"), torch.float32)
    torch.testing.assert_close(first, second)


def test_order_is_reachable_through_the_registry_kwargs_path():
    """The whole point: `kwargs: {order: 1.0}` must now reach the constructor."""
    from spectramr.models.losses.registry import LossRegistry

    loss = LossRegistry.create("sobolev_kspace", order=1.0)
    assert loss.order == 1.0


def test_forward_still_runs_and_differs_by_order():
    pred = torch.randn(1, 2, 8, 8)
    target = torch.randn(1, 2, 8, 8)
    lo = SobolevKSpaceLoss(order=0.0)(pred, target)
    hi = SobolevKSpaceLoss(order=4.0)(pred, target)
    assert torch.isfinite(lo) and torch.isfinite(hi)
    assert not torch.isclose(lo, hi), "order must change the computed loss"
