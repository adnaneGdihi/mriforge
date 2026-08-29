"""Unit tests for MetricSFCLinearizer.

Pin the foreground-only invariant, the MST + DFS-preorder traversal (a
deterministic, connected, heuristic content-aware curve — NOT a
bounded-approximation TSP tour; the module docstring explains why the
MST-doubling guarantee does not apply to this sparse graph), the round-trip
identity, and the on-disk caching behaviour.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from mriforge.models.blocks.metric_sfc import MetricSFCLinearizer  # noqa: E402


def _toy_data(D: int = 4, H: int = 8, W: int = 8, mask_radius: int = 2) -> tuple:
    """Build a small foreground sphere with a random intensity field."""
    rng = np.random.default_rng(42)
    intensity = rng.standard_normal((D, H, W)).astype(np.float32)
    zz, yy, xx = np.meshgrid(
        np.arange(D), np.arange(H), np.arange(W), indexing="ij"
    )
    cz, cy, cx = (D - 1) / 2.0, (H - 1) / 2.0, (W - 1) / 2.0
    r2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
    mask = (r2 <= mask_radius**2).astype(np.uint8)
    return intensity, mask


def test_permutation_visits_only_foreground(tmp_path) -> None:
    intensity, mask = _toy_data()
    lin = MetricSFCLinearizer(intensity, mask, beta=5.0, cache_dir=str(tmp_path))
    fg_flat = np.argwhere(mask.ravel() > 0).ravel()
    perm = lin.forward_idx.cpu().numpy()
    assert sorted(perm.tolist()) == sorted(fg_flat.tolist())
    assert len(perm) == lin.n_foreground == int(mask.sum())


def _two_disjoint_blobs(D: int = 4, H: int = 8, W: int = 8) -> tuple:
    """Two spatially disjoint 2x2x2 foreground blobs with a background gap —
    the connected-components count is 2 (nothing bridges them at any
    connectivity)."""
    rng = np.random.default_rng(7)
    intensity = rng.standard_normal((D, H, W)).astype(np.float32)
    mask = np.zeros((D, H, W), dtype=np.uint8)
    mask[0:2, 0:2, 0:2] = 1  # blob A (corner)
    mask[2:4, 5:7, 5:7] = 1  # blob B (far corner)
    return intensity, mask


def test_disconnected_mask_visits_all_foreground(tmp_path) -> None:
    """A disconnected foreground mask must still visit EVERY foreground voxel.

    Pre-fix, depth_first_order(mst, i_start=0) only traversed node-0's
    connected component, silently dropping the other blob's voxels and
    under-reporting n_foreground (pitfall #16). The stitch must cover all
    components."""
    intensity, mask = _two_disjoint_blobs()
    lin = MetricSFCLinearizer(intensity, mask, beta=5.0, cache_dir=str(tmp_path))
    fg_flat = np.argwhere(mask.ravel() > 0).ravel()
    perm = lin.forward_idx.cpu().numpy()
    assert lin.n_foreground == int(mask.sum())
    assert sorted(perm.tolist()) == sorted(fg_flat.tolist())


def test_disconnected_mask_round_trips(tmp_path) -> None:
    """Forward+reverse over a disconnected mask reconstructs every voxel."""
    intensity, mask = _two_disjoint_blobs()
    lin = MetricSFCLinearizer(intensity, mask, beta=5.0, cache_dir=str(tmp_path))
    x = torch.randn(1, 1, *intensity.shape)
    recon = lin.reverse(lin(x), background=x)
    assert torch.allclose(recon, x, atol=1e-6)


def test_round_trip_identity(tmp_path) -> None:
    """Forward + reverse with the original background must reconstruct input."""
    intensity, mask = _toy_data()
    lin = MetricSFCLinearizer(intensity, mask, beta=5.0, cache_dir=str(tmp_path))
    x = torch.randn(2, 1, *intensity.shape)
    seq = lin(x)
    assert seq.shape == (2, 1, lin.n_foreground)
    # Round-trip with original background to recover the input exactly.
    recon = lin.reverse(seq, background=x)
    assert torch.allclose(recon, x, atol=1e-6)


def test_disk_cache_round_trip(tmp_path) -> None:
    """Building the same linearizer twice must hit the cache."""
    intensity, mask = _toy_data()
    lin1 = MetricSFCLinearizer(
        intensity, mask, beta=5.0, cache_dir=str(tmp_path)
    )
    cache_files = list(tmp_path.glob("sfc_*.pkl"))
    assert len(cache_files) == 1, "Expected a single cached permutation"

    lin2 = MetricSFCLinearizer(
        intensity, mask, beta=5.0, cache_dir=str(tmp_path)
    )
    assert torch.equal(lin1.forward_idx, lin2.forward_idx)
    assert torch.equal(lin1.inverse_idx, lin2.inverse_idx)


def test_beta_changes_permutation(tmp_path) -> None:
    """Increasing beta must in general change the visit order."""
    intensity, mask = _toy_data()
    lin_low = MetricSFCLinearizer(
        intensity, mask, beta=0.0, cache_dir=str(tmp_path / "low")
    )
    lin_high = MetricSFCLinearizer(
        intensity, mask, beta=100.0, cache_dir=str(tmp_path / "high")
    )
    # Different beta → different cache directory → different permutation.
    assert not torch.equal(lin_low.forward_idx, lin_high.forward_idx)


def test_rejects_empty_mask() -> None:
    intensity, _ = _toy_data()
    empty_mask = np.zeros_like(intensity, dtype=np.uint8)
    with pytest.raises(ValueError, match="empty"):
        MetricSFCLinearizer(intensity, empty_mask, beta=5.0, cache_dir=None)


def test_rejects_shape_mismatch() -> None:
    intensity = np.zeros((4, 8, 8), dtype=np.float32)
    mask = np.zeros((4, 8, 16), dtype=np.uint8)
    with pytest.raises(ValueError, match="shapes must match"):
        MetricSFCLinearizer(intensity, mask, cache_dir=None)


def test_rejects_wrong_input_shape(tmp_path) -> None:
    intensity, mask = _toy_data()
    lin = MetricSFCLinearizer(intensity, mask, beta=5.0, cache_dir=str(tmp_path))
    bad_x = torch.zeros(1, 1, 8, 8, 8)  # wrong spatial shape
    with pytest.raises(ValueError, match="spatial shape mismatch"):
        lin(bad_x)
