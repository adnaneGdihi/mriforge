"""Unit tests for D2MambaBlock domain dispatch (pitfall #9 contract).

Regression coverage for the silent-fallback fix in
``spectramr.models.mamba.d2_mamba``: an unknown ``domain`` value must raise a
clear ``ValueError`` naming the offending option instead of silently being
swept into the ``dual`` branch.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.mamba.d2_mamba import D2MambaBlock
from tests.utils.optional_backends import requires_cuda_for_mamba


@pytest.fixture
def block() -> D2MambaBlock:
    torch.manual_seed(0)
    # d_model must equal the channel dim of the input (the scan returns (B, L, C)).
    return D2MambaBlock(d_model=8, d_state=4, d_conv=2, expand=2, dropout=0.0)


def _make_input() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(2, 8, 8, 8, device="cpu")


def test_unknown_domain_raises_valueerror(block: D2MambaBlock) -> None:
    """An unrecognized domain must raise ValueError, not degrade to 'dual'.

    On pre-fix code the typo'd value fell into the bare ``else: # dual`` branch
    and silently produced an output; post-fix it must raise loudly.
    """
    x = _make_input()
    with pytest.raises(ValueError, match="unknown domain"):
        block(x, domain="kspce")  # deliberate typo


def test_unknown_domain_message_names_offending_value(block: D2MambaBlock) -> None:
    """The error names the bad value and the advertised set."""
    x = _make_input()
    with pytest.raises(ValueError) as exc_info:
        block(x, domain="k-space")
    msg = str(exc_info.value)
    assert "k-space" in msg
    assert "kspace" in msg and "image" in msg and "dual" in msg


@requires_cuda_for_mamba
@pytest.mark.parametrize("domain", ["kspace", "image", "dual"])
def test_legal_domains_preserve_shape(block: D2MambaBlock, domain: str) -> None:
    """Each of the three legal domains returns a (B, C, H, W) tensor.

    Guards against the explicit-elif refactor accidentally dropping the dual
    path or any legal branch.
    """
    x = _make_input()
    out = block(x, domain=domain)
    assert out.shape == x.shape


@requires_cuda_for_mamba
def test_default_domain_is_dual(block: D2MambaBlock) -> None:
    """Calling forward with no domain arg uses the 'dual' branch unchanged."""
    x = _make_input()
    out = block(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Scan-index cache (perf, 2026-07-01)
# ---------------------------------------------------------------------------


def test_scan_indices_cached_once_and_match_builders(block: D2MambaBlock) -> None:
    """PERF regression (2026-07-01): the pure-Python Morton build (~2M Python
    ops at 256²) and the concentric build + a fresh ``argsort`` used to re-run
    on EVERY forward (twice: scan + unscan). ``_get_scan_indices`` memoizes
    (indices, inverse) per (kind, H, W, device); the static builders remain
    the single source of the orderings."""
    device = torch.device("cpu")
    for kind, builder in (
        ("zorder", D2MambaBlock._get_zorder_indices),
        ("concentric", D2MambaBlock._get_concentric_indices),
    ):
        idx, inv = block._get_scan_indices(kind, 8, 8, device)
        assert torch.equal(idx, builder(8, 8, device)), f"{kind} ordering changed"
        assert torch.equal(inv, torch.argsort(idx)), f"{kind} inverse wrong"

        idx2, inv2 = block._get_scan_indices(kind, 8, 8, device)
        assert idx2 is idx and inv2 is inv, f"{kind} rebuilt on cache hit"


def test_scan_unscan_roundtrips_are_identity(block: D2MambaBlock) -> None:
    """The cached inverse must exactly undo each scan ordering."""
    x = _make_input()

    scanned = block._image_scan(x)
    assert torch.equal(block._image_unscan(scanned, 8, 8), x)

    scanned_k = block._kspace_scan(x)
    assert torch.equal(block._kspace_unscan(scanned_k, 8, 8), x)


def test_unknown_scan_kind_raises(block: D2MambaBlock) -> None:
    with pytest.raises(ValueError, match="Unknown scan kind"):
        block._get_scan_indices("raster", 8, 8, torch.device("cpu"))


@requires_cuda_for_mamba
def test_forward_deterministic_with_cache(block: D2MambaBlock) -> None:
    """Caching must not change results: two identical eval forwards agree."""
    block.eval()
    x = _make_input()
    with torch.no_grad():
        out1 = block(x, domain="dual")
        out2 = block(x, domain="dual")
    assert torch.equal(out1, out2)
