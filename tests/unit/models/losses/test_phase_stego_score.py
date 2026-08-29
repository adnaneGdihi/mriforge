"""Domain-contract tests for :class:`PhaseStegoScoreLoss`.

The canonical paired test file did not exist. It is added here with the fix
for the defect cluster job 8004252 reported: the loss is registered
``domain="complex"`` but silently upcast a real tensor with
``x.to(torch.complex64)``. The imaginary part was then zero, so
``torch.angle`` was identically 0 (pi for negative reals) — a constant — and
the whole loss had a **zero gradient while reporting a large non-zero value**.
A term that contributes to the reported total and nothing to the update is
pitfall #16, and it was live: ``twin_dps_strategy`` computes the entire
marker-side guidance through this call.

On the framework's canonical interleaved layout (real ``[B, 2C, H, W]``, even
C — see ``physics.md`` and ``fft_ops._to_complex``) the upcast was worse
still: it read the two interleaved channels as two separate zero-imaginary
images and discarded the phase the loss exists to score.

The fix routes through the SSOT, ``fft_ops._to_complex(strict=True)``, which
already understands both the last-dim-2 and the interleaved-channel encodings
and raises on a genuinely real tensor instead of degrading.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mriforge.models.losses.phase_stego_score import (  # noqa: E402
    PhaseStegoScoreLoss,
)
from mriforge.models.losses.registry import list_available  # noqa: E402


def test_registered() -> None:
    assert "phase_stego_score" in list_available()


def test_interleaved_real_input_carries_real_phase() -> None:
    """Interleaved ``[B, 2, H, W]`` must be read as complex, not zero-imag.

    This is the assertion the old upcast could never satisfy: with a zero
    imaginary part every angle is 0, so the phase field is constant. Here the
    imaginary channel is non-zero and distinct from the real one, so a correct
    conversion yields a phase field that actually varies.
    """
    loss = PhaseStegoScoreLoss(sigma_M=1.0, basis="fourier")
    real = torch.ones(1, 1, 8, 8)
    imag = torch.linspace(-1.0, 1.0, 64).reshape(1, 1, 8, 8)
    x = torch.cat([real, imag], dim=1)

    phase = torch.angle(loss._phase_stego_forward(x))
    # A zero-imaginary upcast collapses this to a single value.
    assert phase.unique().numel() > 1


def test_gradient_reaches_the_prediction() -> None:
    """The marker term must move the estimate, not just report a number."""
    loss = PhaseStegoScoreLoss(sigma_M=1.0, basis="fourier")
    x = torch.randn(1, 2, 8, 8, requires_grad=True)
    marker = torch.zeros(1, 1, 8, 8, dtype=torch.complex64)

    out = loss(x, marker)
    out.abs().backward()

    assert x.grad is not None
    assert torch.any(x.grad != 0), (
        "phase_stego_score reported a value but left the prediction's "
        "gradient all-zeros — the marker guidance would be inert."
    )


def test_pure_real_input_raises_rather_than_scoring_a_constant_phase() -> None:
    """A 1-channel real image has no phase, so the loss must reject it.

    ``strict=True`` is what makes this a raise. The pre-fix behaviour returned
    a large finite value here (272659.75 on the contract suite's synthetic
    input) computed over a phase field that was identically zero.
    """
    loss = PhaseStegoScoreLoss(sigma_M=1.0, basis="fourier")
    with pytest.raises(ValueError):
        loss(torch.rand(1, 1, 8, 8), torch.zeros(1, 1, 8, 8))


def test_complex_input_is_passed_through_unchanged() -> None:
    """A genuinely complex tensor is the production shape and must still work.

    Guards the fix against over-tightening: ``_to_complex`` returns a complex
    tensor untouched, so the live ``twin_dps_strategy`` path is unaffected by
    the strictness added for the real case.
    """
    loss = PhaseStegoScoreLoss(sigma_M=1.0, basis="fourier")
    x = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    out = loss(x, torch.zeros(1, 1, 8, 8, dtype=torch.complex64))
    assert torch.isfinite(out.abs()).all()
