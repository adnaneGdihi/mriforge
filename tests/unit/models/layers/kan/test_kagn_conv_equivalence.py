"""Equivalence tests for the KAGN group-loop ``.clone()`` removal (2026-07-01).

The group loops in ``kagn_conv.py`` / ``kagn_bottleneck_conv.py`` appended
``y.clone()`` before ``torch.cat`` — but ``cat`` copies its inputs anyway and
``y`` is a fresh tensor from ``forward_kag`` each iteration, so the clone was
a pure extra activation allocation + copy per group per forward. The group
loop itself is deliberately untouched (folding groups into ``conv(groups=)``
would reshuffle the per-group ``ModuleList`` parameter layout and break
checkpoints; all repo call sites use ``groups=1`` anyway).
"""

from __future__ import annotations

import pytest
import torch

from mriforge.models.layers.kan.kan_convs.kagn_conv import KAGNConv2DLayer


@pytest.mark.unit
@pytest.mark.parametrize("groups", [1, 2])
def test_forward_equals_explicit_clone_reference(groups: int) -> None:
    """forward(x) must equal the old clone-per-group construction bitwise."""
    torch.manual_seed(0)
    conv = KAGNConv2DLayer(
        input_dim=4, output_dim=6, kernel_size=3, degree=3, groups=groups, padding=1
    )
    conv.eval()
    x = torch.randn(2, 4, 8, 8)

    with torch.no_grad():
        out = conv(x)
        split_x = torch.split(x, conv.inputdim // conv.groups, dim=1)
        ref = torch.cat(
            [
                conv.forward_kag(_x, group_ind).clone()
                for group_ind, _x in enumerate(split_x)
            ],
            dim=1,
        )

    assert torch.equal(out, ref), "clone removal changed the forward output"


@pytest.mark.unit
def test_group_loop_has_no_clone() -> None:
    """Pin the perf contract at source level for both KAGN conv modules."""
    import inspect

    from mriforge.models.layers.kan.kan_convs import kagn_bottleneck_conv, kagn_conv

    for mod in (kagn_conv, kagn_bottleneck_conv):
        src = inspect.getsource(mod)
        assert ".clone())" not in src, (
            f"redundant pre-cat .clone() reappeared in {mod.__name__}"
        )


@pytest.mark.unit
def test_gradients_flow_through_all_groups() -> None:
    torch.manual_seed(1)
    conv = KAGNConv2DLayer(
        input_dim=4, output_dim=4, kernel_size=3, degree=2, groups=2, padding=1
    )
    x = torch.randn(1, 4, 6, 6, requires_grad=True)

    conv(x).sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    grads = [p.grad for p in conv.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
