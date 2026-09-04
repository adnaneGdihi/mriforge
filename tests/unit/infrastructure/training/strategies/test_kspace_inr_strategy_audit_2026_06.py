"""Regression tests for the 2026-06 audit fixes to ``KSpaceINRStrategy``.

Covers the kspace_inr_strategy section of the 2026-05-31 strategy audit:

* **Silent zero-loss fallback** — ``_compute_losses_impl`` previously returned
  ``{"loss_total": tensor(0.0)}`` when the generator or ``batch['kspace']`` was
  missing, presenting a zero-gradient step that looks like training (CLAUDE.md
  pitfall #9/#10). Both branches must now ``raise RuntimeError``.
* **Docstring divergence** — the module docstring advertised a per-subject SIREN
  coordinate-MLP fit with an iFFT-to-image step that is not implemented; the
  loss is k-space-only. The docstring must no longer claim an image-domain
  ``iFFT`` step, and the happy path must still produce a k-space MAE loss.

Strategy methods are invoked on an ``object.__new__`` instance with tiny fakes
(no DI/env build needed), mirroring the sibling dispatch regression test.
"""

from __future__ import annotations

import types

import pytest
import torch

from spectramr.infrastructure.training.strategies import kspace_inr_strategy
from spectramr.infrastructure.training.strategies.kspace_inr_strategy import (
    KSpaceINRStrategy,
)


def _make(generator) -> KSpaceINRStrategy:
    """Build a strategy bypassing ``__init__`` (no DI/env build needed)."""
    s = object.__new__(KSpaceINRStrategy)
    s.env = types.SimpleNamespace(generator=generator)
    s.device = torch.device("cpu")
    return s


# ---------------------------------------------------------------------------
# Finding 1: silent zero-loss fallback -> RuntimeError
# ---------------------------------------------------------------------------


def test_missing_generator_raises():
    """generator is None -> loud RuntimeError, not a zero-loss step."""
    s = _make(generator=None)
    with pytest.raises(RuntimeError, match=r"requires self\.env\.generator"):
        s._compute_losses_impl(
            input_batch={"kspace": torch.randn(1, 2, 4, 4)},
            target_batch=None,
            epoch=0,
        )


def test_missing_kspace_raises():
    """batch lacks 'kspace' -> loud RuntimeError, not a zero-loss step."""
    s = _make(generator=torch.nn.Identity())
    with pytest.raises(RuntimeError, match=r"kspace.*is missing"):
        s._compute_losses_impl(
            input_batch={"x": torch.randn(1, 2, 4, 4)},
            target_batch=None,
            epoch=0,
        )


def test_no_silent_zero_loss_fallback_in_source():
    """The removed silent fallback must not reappear in the source."""
    import inspect

    src = inspect.getsource(KSpaceINRStrategy._compute_losses_impl)
    # The exact silent-fallback literal that was removed.
    assert 'return {"loss_total": torch.tensor(0.0' not in src


# ---------------------------------------------------------------------------
# Happy path: k-space-only mask-weighted L1 still works (no image-domain iFFT)
# ---------------------------------------------------------------------------


class _MaskAwareGen(torch.nn.Module):
    """k-space generator accepting the optional ``mask`` kwarg the strategy
    threads (a bare ``Conv2d`` would reject ``gen(kspace, mask=mask)``)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, kernel_size=1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.conv(x)


def test_happy_path_kspace_mae_loss():
    """With generator + kspace present, returns a grad-bearing k-space loss."""
    gen = _MaskAwareGen()
    s = _make(generator=gen)
    kspace = torch.randn(1, 2, 4, 4)
    mask = torch.ones(1, 1, 4, 4)
    out = s._compute_losses_impl(
        input_batch={"kspace": kspace, "mask": mask},
        target_batch=None,
        epoch=0,
    )
    assert "loss_total" in out
    assert "loss_kspace_mae" in out
    # The objective must carry a gradient back to the generator.
    assert out["loss_total"].requires_grad


# ---------------------------------------------------------------------------
# Finding 3 / 2: docstring no longer advertises the unimplemented iFFT-to-image
# ---------------------------------------------------------------------------


def test_module_docstring_drops_unbacked_ifft_claim():
    """The module docstring must not claim an (unimplemented) iFFT-to-image step
    as the strategy's behaviour."""
    doc = kspace_inr_strategy.__doc__ or ""
    # The old docstring asserted "Then iFFT to image." as actual behaviour.
    assert "Then iFFT to image." not in doc
    # It must now state the k-space-only scope explicitly.
    assert "k-space" in doc.lower()
