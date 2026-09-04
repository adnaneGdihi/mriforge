"""Perf regressions for BlochMambaV2 (2026-07-01).

Two fixes pinned here:

* ``_hilbert_indices`` used to key its cache on ``(h, w)`` and store the
  tensor on CPU, so every cache *hit* paid a fresh host→device ``.to(device)``
  copy — a recurring H2D transfer on every forward. The cache is now keyed on
  ``(h, w, device)`` and stores the device-resident tensor (moved exactly
  once).

* ``BlochSSM.parallel_scan_step`` allocated a full ``zeros_like(B_u)`` and
  wrote its first row in place, then discarded the buffer by overwriting
  ``h`` with the stacked list — dead alloc/copy removed. Values and grads are
  identical (the old inline math is kept here as the reference).
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.mamba.bloch_mamba_v2 import BlochMambaV2, BlochSSM


@pytest.mark.unit
def test_hilbert_indices_cache_hit_returns_same_object() -> None:
    model = BlochMambaV2(in_channels=2, base_dim=8, num_layers=1)
    device = torch.device("cpu")

    first = model._hilbert_indices(8, 8, device)
    second = model._hilbert_indices(8, 8, device)

    assert first.device == device
    assert second is first, (
        "cache hit must return the stored device tensor, not a fresh copy "
        "(the old code stored CPU + .to(device) on every hit)"
    )
    # Sanity: it is a permutation of the flat grid.
    assert torch.equal(torch.sort(first).values, torch.arange(64))


@pytest.mark.unit
def test_hilbert_indices_cache_is_device_keyed() -> None:
    """Different (h, w) entries coexist; keys carry the device so a moved
    module can never serve stale-device indices."""
    model = BlochMambaV2(in_channels=2, base_dim=8, num_layers=1)
    device = torch.device("cpu")

    a = model._hilbert_indices(4, 4, device)
    b = model._hilbert_indices(8, 8, device)
    assert a.numel() == 16 and b.numel() == 64
    assert (4, 4, device) in model._hilbert_cache
    assert (8, 8, device) in model._hilbert_cache


def _reference_parallel_scan_step(
    log_A: torch.Tensor, B_u: torch.Tensor
) -> torch.Tensor:
    """The pre-2026-07-01 loop, verbatim (incl. the dead buffer)."""
    _B, L, _N = log_A.shape
    A = torch.exp(log_A)
    h = torch.zeros_like(B_u)
    h[:, 0] = B_u[:, 0]
    current_h = h[:, 0]
    h_list = [current_h]
    for t in range(1, L):
        current_h = A[:, t] * current_h + B_u[:, t]
        h_list.append(current_h)
    return torch.stack(h_list, dim=1)


@pytest.mark.unit
def test_parallel_scan_step_matches_old_loop() -> None:
    torch.manual_seed(0)
    ssm = BlochSSM(8, d_state=4)
    # Decaying |A| < 1 for a stable recurrence, complex like the live path.
    log_A = -torch.rand(2, 6, 4) - 0.1j * torch.rand(2, 6, 4)
    B_u = torch.randn(2, 6, 4, dtype=torch.complex64)

    out = ssm.parallel_scan_step(log_A, B_u)
    ref = _reference_parallel_scan_step(log_A, B_u)

    assert torch.equal(out, ref), "scan diverged from the pre-fix loop"


@pytest.mark.unit
def test_parallel_scan_step_gradients_match_old_loop() -> None:
    torch.manual_seed(1)
    ssm = BlochSSM(8, d_state=4)

    log_A = (-torch.rand(1, 5, 4) - 0.1j * torch.rand(1, 5, 4)).requires_grad_(True)
    B_u_new = torch.randn(1, 5, 4, dtype=torch.complex64, requires_grad=True)
    B_u_ref = B_u_new.detach().clone().requires_grad_(True)
    log_A_ref = log_A.detach().clone().requires_grad_(True)

    ssm.parallel_scan_step(log_A, B_u_new).abs().sum().backward()
    _reference_parallel_scan_step(log_A_ref, B_u_ref).abs().sum().backward()

    assert torch.allclose(B_u_new.grad, B_u_ref.grad, rtol=1e-6, atol=1e-6)
    assert torch.allclose(log_A.grad, log_A_ref.grad, rtol=1e-6, atol=1e-6)
