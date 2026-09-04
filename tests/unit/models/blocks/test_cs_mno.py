"""Unit tests for CS-MNO building blocks.

Covers the four invariants the integration plan calls out as
merge-blocking:

1. ``ImageTopologyLinearizer.physical_arc_delta()`` returns a
   sequence whose length equals the grid size and whose maximum
   step is the Hilbert Hölder bound (much smaller than the
   raster row-jump).
2. ``SpectralBranch2d`` / ``SpectralBranch3d`` preserve shape and
   pass autograd cleanly.
3. ``ContinuousSFCMamba`` raises when the use_physical_arc / Δt
   contract is broken (no silent fallback) and produces different
   outputs when Δt actually changes (the physical scaling is wired
   through, not ignored).
4. ``CSMNOLayer`` end-to-end shape contract on both 2-D and 3-D
   inputs.

Pytest collects these even on CPU-only machines because the existing
conftest stubs ``torch`` if it is missing — but most assertions
require a real torch, so we ``importorskip``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # noqa: E402

from spectramr.models.blocks.continuous_sfc_mamba import ContinuousSFCMamba  # noqa: E402
from spectramr.models.blocks.cs_mno_layer import CSMNOLayer  # noqa: E402
from spectramr.models.blocks.spectral_branch import (  # noqa: E402
    SpectralBranch2d,
    SpectralBranch3d,
)
from spectramr.models.blocks.topology_linearizer import ImageTopologyLinearizer  # noqa: E402

# ── physical_arc_delta ─────────────────────────────────────────────


def test_physical_arc_hilbert_2d_is_locality_preserving() -> None:
    """Hilbert at L=16 has every step exactly 1/16 (lattice neighbours)."""
    lin = ImageTopologyLinearizer((16, 16), mode="hilbert_2d")
    delta = lin.physical_arc_delta()
    assert delta.shape == (16 * 16,)
    assert delta[0].item() == 0.0  # convention: first token has no predecessor
    # All Hilbert steps are unit-lattice neighbours, normalised by L=16.
    assert torch.allclose(delta[1:], torch.full((255,), 1.0 / 16.0), atol=1e-6)


def test_physical_arc_raster_has_row_break_jump() -> None:
    """Raster scan exhibits long jumps at row boundaries."""
    lin = ImageTopologyLinearizer((8, 8), mode="raster_2d")
    delta = lin.physical_arc_delta()
    # Row breaks (every 8th step) jump diagonally across the grid:
    # ‖((0,7) → (1,0))‖ / 8 = sqrt(1 + 49)/8 ≈ 0.884
    assert delta.max().item() > 0.8


def test_physical_arc_3d_shape() -> None:
    """3-D Hilbert returns the right-shaped Δt buffer."""
    lin = ImageTopologyLinearizer((4, 4, 4), mode="hilbert_3d")
    delta = lin.physical_arc_delta()
    assert delta.shape == (64,)


# ── SpectralBranch ────────────────────────────────────────────────


@pytest.mark.parametrize("dim,modes", [(8, 4), (16, 8)])
def test_spectral_branch_2d_shape(dim: int, modes: int) -> None:
    block = SpectralBranch2d(dim=dim, modes=modes)
    x = torch.randn(2, dim, 32, 32)
    y = block(x)
    assert y.shape == x.shape


def test_spectral_branch_2d_autograd() -> None:
    block = SpectralBranch2d(dim=4, modes=2)
    x = torch.randn(1, 4, 16, 16, requires_grad=True)
    y = block(x).sum()
    y.backward()
    assert x.grad is not None and x.grad.norm().item() > 0


def test_spectral_branch_3d_shape() -> None:
    block = SpectralBranch3d(dim=4, modes=2)
    x = torch.randn(1, 4, 8, 8, 8)
    y = block(x)
    assert y.shape == x.shape


# ── ContinuousSFCMamba ────────────────────────────────────────────


def test_mamba_raises_when_physical_arc_required_but_missing() -> None:
    """No silent fallback: missing Δt is a ValueError, not a default."""
    block = ContinuousSFCMamba(dim=4, d_state=4, use_physical_arc=True)
    with pytest.raises(ValueError, match="delta_phys"):
        block(torch.randn(1, 8, 4), delta_phys=None)


def test_mamba_raises_on_delta_length_mismatch() -> None:
    block = ContinuousSFCMamba(dim=4, d_state=4, use_physical_arc=True)
    x = torch.randn(1, 8, 4)
    bad_delta = torch.zeros(7)  # one short
    with pytest.raises(ValueError, match="length"):
        block(x, delta_phys=bad_delta)


def test_mamba_output_changes_when_delta_changes() -> None:
    """Sanity: physical-arc Δt is actually wired through, not ignored."""
    torch.manual_seed(0)
    block = ContinuousSFCMamba(dim=4, d_state=4, use_physical_arc=True)
    x = torch.randn(1, 8, 4)
    y_small = block(x, delta_phys=torch.full((8,), 0.01))
    y_large = block(x, delta_phys=torch.full((8,), 0.5))
    assert not torch.allclose(y_small, y_large, atol=1e-3)


def test_mamba_scalar_delta_fallback_runs() -> None:
    block = ContinuousSFCMamba(dim=4, d_state=4, use_physical_arc=False)
    x = torch.randn(1, 8, 4)
    y = block(x, delta_phys=None)  # ignored in fallback mode
    assert y.shape == x.shape


# ── CSMNOLayer ────────────────────────────────────────────────────


def test_layer_2d_shape_and_autograd() -> None:
    layer = CSMNOLayer(dim=8, modes=4, d_state=4, spatial_dims=2)
    lin = ImageTopologyLinearizer((16, 16), mode="hilbert_2d")
    x = torch.randn(2, 8, 16, 16, requires_grad=True)
    y = layer(x, lin.forward_idx, lin.inverse_idx, lin.physical_arc_delta())
    assert y.shape == x.shape
    y.mean().backward()
    assert x.grad is not None and x.grad.norm().item() > 0


def test_layer_rejects_unknown_activation() -> None:
    with pytest.raises(ValueError, match="activation"):
        CSMNOLayer(dim=8, modes=4, d_state=4, spatial_dims=2, activation="mish")


def test_layer_3d_non_cubic_shape() -> None:
    """A non-cubic grid catches an N miscomputation the cubic case hides.

    ``N`` is the product of the spatial extents. With a square/cubic grid a
    wrong reduction (a sum, a first-axis pick, a transposed shape) can still
    happen to produce a reshape-compatible number; ``6*4*8`` does not.
    """
    layer = CSMNOLayer(dim=4, modes=2, d_state=4, spatial_dims=3)
    lin = ImageTopologyLinearizer((6, 4, 8), mode="raster_3d")
    x = torch.randn(2, 4, 6, 4, 8)
    y = layer(x, lin.forward_idx, lin.inverse_idx, lin.physical_arc_delta())
    assert y.shape == x.shape


def test_layer_forward_makes_no_host_device_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``CSMNOLayer.forward`` must not call ``.item()`` (non-negotiable 9).

    The token count used to be computed as ``torch.tensor(spatial).prod()
    .item()``. ``x.shape[2:]`` is a ``torch.Size`` of Python ints, so that
    allocated a throwaway CPU tensor on every forward -- cheap, but it is the
    exact shape non-negotiable 9 forbids in the loop, and it is a real sync on
    any input whose shape is carried on-device. ``math.prod`` is exact and
    allocation-free.
    """
    calls: list[str] = []
    original = torch.Tensor.item

    def counting(self: torch.Tensor, *args: object, **kwargs: object) -> object:
        calls.append(str(tuple(self.shape)))
        return original(self, *args, **kwargs)

    layer = CSMNOLayer(dim=8, modes=4, d_state=4, spatial_dims=2)
    lin = ImageTopologyLinearizer((16, 16), mode="hilbert_2d")
    x = torch.randn(2, 8, 16, 16)
    args = (lin.forward_idx, lin.inverse_idx, lin.physical_arc_delta())

    monkeypatch.setattr(torch.Tensor, "item", counting)
    layer(x, *args)
    assert calls == []
