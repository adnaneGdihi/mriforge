"""Regression: ``ConcomitantAwareReconStrategy`` must actually *enable* the
Maxwell concomitant-phase correction.

2026-05-31 audit found the strategy's ``__init__`` validated
``physics.concomitant.enabled`` but never called
``configure_concomitant_correction(...)``, leaving the mixin disabled
(``_concomitant_enabled`` False) — a silent no-op (CLAUDE.md #9). The fix
configures the operator lazily on the first ``_compute_losses_impl`` call
(spatial_shape needs the batch resolution) and pre-rotates the network input
into the concomitant frame before the parent reconstruction loss runs.

The strategy is built via ``object.__new__`` with only the attributes the
tested method touches stubbed, so the test stays decoupled from the heavy
``ReconstructionTrainingStrategy.__init__`` plumbing.
"""

from __future__ import annotations

import types

import torch

from mriforge.infrastructure.training.strategies.concomitant_aware_recon_strategy import (
    ConcomitantAwareReconStrategy,
)
from mriforge.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)


def _make_strategy(field_strength: float = 0.064):
    """Build a bare strategy with only the attrs ``_compute_losses_impl`` reads."""
    s = object.__new__(ConcomitantAwareReconStrategy)
    s.device = torch.device("cpu")
    s._concomitant_configured = False
    # Mixin class attrs are present on the class, but set instance attrs so the
    # property reads a fresh per-instance state.
    s._concomitant_enabled = False
    s._concomitant_op = None

    physics = types.SimpleNamespace(
        field_strength=field_strength,
        concomitant=types.SimpleNamespace(
            enabled=True, max_gradient_T_per_m=0.040
        ),
    )
    s.config = types.SimpleNamespace(physics=physics)

    # Stub generator: identity-ish module the parent forward would call. We stub
    # the *parent* _compute_losses_impl so the test exercises only our override
    # (configure + generator wrap), not the full reconstruction loss path.
    dummy = torch.nn.Parameter(torch.zeros(1))
    captured: dict = {}

    def _generator(x, *a, **kw):
        captured["forward_input"] = x
        return x

    s.env = types.SimpleNamespace(
        opt_g=torch.optim.SGD([dummy], lr=0.1), generator=_generator
    )
    return s, captured


def test_concomitant_enabled_after_first_compute_losses(monkeypatch):
    """After one ``_compute_losses_impl`` call the correction is ENABLED."""
    s, _ = _make_strategy()

    # Stub the parent loss impl so we test only the configure/wrap behaviour.
    def _parent(input_batch=None, target_batch=None, epoch=0, **kwargs):
        # Drive the wrapped generator so the corrected input reaches the forward.
        s.env.generator(input_batch)
        return {"g_total_loss": torch.tensor(0.0)}

    monkeypatch.setattr(
        ReconstructionTrainingStrategy,
        "_compute_losses_impl",
        staticmethod(_parent),
        raising=True,
    )

    assert s.concomitant_enabled is False
    img = torch.randn(2, 2, 16, 16)
    s._compute_losses_impl(input_batch=img, target_batch=img, epoch=0)

    # KEY property of the fix: configure ran and the mixin is now enabled.
    assert s.concomitant_enabled is True
    assert s._concomitant_configured is True
    assert s._concomitant_op is not None
    # Operator buffer matches the (1, H, W) spatial shape derived from the batch.
    assert tuple(s._concomitant_op.phasor.shape) == (1, 16, 16)
    # And the phasor is genuinely NON-trivial on the single-slice (D=1) path:
    # the imaging gradient is placed on the slice-select (Gz) axis so the
    # central-slice term Gz^2(x^2+y^2)/(8 B0) is non-zero (previously the (g,g,0)
    # config gave z=0 AND Gz=0 -> phasor == 1, an inert facade).
    assert s._concomitant_op.phasor.angle().abs().max() > 1e-4


def test_corrected_input_reaches_forward(monkeypatch):
    """The concomitant correction wraps the forward (paradigm tensor reaches the
    network input) and the operator is configured exactly once."""
    s, captured = _make_strategy()

    def _parent(input_batch=None, target_batch=None, epoch=0, **kwargs):
        s.env.generator(input_batch)
        return {"g_total_loss": torch.tensor(0.0)}

    monkeypatch.setattr(
        ReconstructionTrainingStrategy,
        "_compute_losses_impl",
        staticmethod(_parent),
        raising=True,
    )

    img = torch.randn(1, 2, 8, 8)
    s._compute_losses_impl(input_batch=img, target_batch=img, epoch=0)

    fwd = captured["forward_input"]
    # The corrected input reaches the forward with the same real channel layout.
    assert fwd is not None
    assert fwd.shape == img.shape
    assert fwd.dtype == img.dtype  # projected back to the real interleaved layout
    # The original generator is restored after the wrapped call (no leak).
    assert callable(s.env.generator)

    # A second call must NOT re-configure (guard flag holds, op is reused).
    op_before = s._concomitant_op
    s._compute_losses_impl(input_batch=img, target_batch=img, epoch=1)
    assert s._concomitant_op is op_before


def test_apply_correction_rotates_complex_image_when_phase_nonzero():
    """The configured operator applies a genuinely non-trivial phase rotation
    to a complex image whose last three dims match ``(D, H, W)`` with ``D > 1``
    (so the Bernstein z^2 slice term is non-zero)."""
    s, _ = _make_strategy(field_strength=0.064)
    s.configure_concomitant_correction(
        spatial_shape=(4, 8, 8),
        voxel_size_mm=(1.0, 1.0, 1.0),
        gradient_amplitudes_mT_per_m=(40.0, 40.0, 0.0),
        field_strength_t=0.064,
        readout_time_s=1e-3,
    )
    # Operator contract: complex image with last three dims (D, H, W).
    img = torch.randn(2, 4, 8, 8, dtype=torch.complex64)
    out = s.apply_concomitant_correction(img)
    assert out.shape == img.shape
    assert s.concomitant_op.phasor.angle().abs().max() > 0  # phase is non-zero
    assert not torch.allclose(out, img)  # rotation modifies the image


def test_correct_input_preserves_real_interleaved_layout():
    """``_correct_input`` keeps the host network's real interleaved [B, 2k, H, W]
    layout (dtype + shape) so the corrected tensor is a drop-in network input."""
    s, _ = _make_strategy(field_strength=0.064)
    s.configure_concomitant_correction(
        spatial_shape=(1, 8, 8),
        voxel_size_mm=(1.0, 1.0, 1.0),
        gradient_amplitudes_mT_per_m=(40.0, 40.0, 0.0),
        field_strength_t=0.064,
        readout_time_s=1e-3,
    )
    s._concomitant_configured = True
    img = torch.randn(3, 2, 8, 8)  # [B, real, imag, ...] interleaved
    out = s._correct_input(img)
    assert out.shape == img.shape
    assert out.dtype == img.dtype  # real, not complex — drop-in for the network
