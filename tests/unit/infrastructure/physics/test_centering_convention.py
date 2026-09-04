"""Regression tests for the centered-k-space convention (CLAUDE.md pitfall #2).

Locks ``TODO/backlog_ssot_and_layering_cleanup.md`` Phase 3:

- ``src/models/diffusion/ssad.py::SSADInterface._fft2`` previously
  applied a spurious ``torch.fft.fftshift`` *after* ``fft2c``, producing
  corner-DC output the SSAD loss then compared against the centered
  ``kspace`` input — a silent k-space-convention mismatch.

- ``src/core/metrics/evaluation_metrics.py::RadialProfile.radial_profile``
  did the same: ``fft2c`` returned centered k-space, then ``fftshift``
  un-centered ``power`` while the radial-bin grid still assumed DC at
  ``(H//2, W//2)`` — misaligning every reported headline number.

Both are source-level pins (we don't have the SSAD trainer loop or the
metric integration test set up here). The test asserts the spurious
``torch.fft.fftshift(...)`` lines are gone and the comment explaining
why is in place — the regression is structural.
"""

from __future__ import annotations

import inspect

import torch


def _calls_fft_shift(src: str) -> bool:
    """Return True if ``src`` actually *invokes* ``torch.fft.fftshift(...)``.

    Mere mention (in a comment or docstring explaining the past bug) is
    fine; what we forbid is the active call.
    """
    return "torch.fft.fftshift(" in src


def test_ssad_fft2_does_not_apply_spurious_fftshift() -> None:
    """``SSADInterface._fft2`` no longer corner-shifts the centered output."""
    from spectramr.models.diffusion import ssad

    src = inspect.getsource(ssad.SSADInterface._fft2)
    assert not _calls_fft_shift(src), (
        "ssad._fft2 reintroduced a corner-shift after fft2c — that "
        "produces corner-DC k-space and breaks the SSAD loss diff "
        "against centered ``kspace``. See TODO/backlog_ssot_and_layering_cleanup.md Phase 3."
    )
    # The new explanatory comment is in place.
    assert "already centers" in src


def test_radial_profile_does_not_un_center_power_spectrum() -> None:
    """``PowerSpectrumConsistency.radial_profile`` no longer fftshifts the power spectrum.

    Same convention-mismatch bug as in ssad — the radial-bin grid uses
    ``cy, cx = H//2, W//2`` which assumes DC at center, so any
    additional ``fftshift`` on ``power`` misaligns every bin.
    """
    from spectramr.core.metrics import evaluation_metrics

    cls = getattr(evaluation_metrics, "PowerSpectrumConsistency", None)
    assert cls is not None, "PowerSpectrumConsistency metric must remain in the module"
    src = inspect.getsource(cls.radial_profile)
    assert not _calls_fft_shift(src)


def test_ssad_fft2_actually_round_trips_correctly() -> None:
    """``_fft2(image)`` followed by ``ifft2c`` on the result recovers the image.

    This is the *behavioural* test that the source-level pins enable.
    Bug pattern: if ``_fft2`` accidentally re-introduced an extra shift,
    the inverse round-trip would produce a phase-shifted image rather
    than the original.
    """
    from spectramr.infrastructure.physics.fft_ops import ifft2c
    from spectramr.models.diffusion.ssad import SSADInterface

    torch.manual_seed(0)
    # Build a stand-alone instance — _fft2 is a pure-tensor method so
    # we can monkey-instantiate the class without the model.
    iface = SSADInterface.__new__(SSADInterface)
    torch.nn.Module.__init__(iface)
    # Real/imag stacked input [B=1, C=2, H, W]
    x_complex = torch.randn(1, 8, 8, dtype=torch.complex64)
    x_stacked = torch.stack([x_complex.real, x_complex.imag], dim=1)
    k_real = iface._fft2(x_stacked)
    # Round-trip through the canonical ifft2c.
    k_complex = torch.complex(k_real[:, 0], k_real[:, 1])
    recon = ifft2c(k_complex)
    assert torch.allclose(recon, x_complex.squeeze(0), rtol=1e-4, atol=1e-5), (
        "SSAD _fft2 → ifft2c does NOT recover the input — the centering "
        "convention has drifted again."
    )
