"""Regression tests for ConcreteTTOStrategy's data-consistency loss.

VF-smoke-2026-05-25: ``experiment_vf_tto_v2`` diverged with
``loss_dc=7673`` and rising (``gradient_explosion`` health flag) because
the DC residual was computed in un-normalized raw-k-space units. The fix
normalises by the measurement energy so the loss is O(1) and
scale-invariant. These tests pin that contract on the extracted
``_normalized_dc_loss`` helper.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from mriforge.infrastructure.training.strategies.tto_strategy import (  # noqa: E402
    ConcreteTTOStrategy,
)


def test_normalized_dc_loss_is_scale_invariant() -> None:
    """Scaling both measurement and prediction leaves the loss unchanged."""
    torch.manual_seed(0)
    y = torch.randn(2, 1, 8, 8, dtype=torch.complex64)
    pred = y + 0.1 * torch.randn(2, 1, 8, 8, dtype=torch.complex64)

    loss_unit = ConcreteTTOStrategy._normalized_dc_loss(pred, y)
    # Raw M4Raw-like scale: multiply by 1000.
    loss_scaled = ConcreteTTOStrategy._normalized_dc_loss(pred * 1000.0, y * 1000.0)

    assert torch.allclose(loss_unit, loss_scaled, atol=1e-5), (
        f"DC loss must be scale-invariant; got {loss_unit.item()} vs "
        f"{loss_scaled.item()}"
    )


def test_normalized_dc_loss_stays_order_one_at_large_scale() -> None:
    """The headline bug: raw-scale k-space must NOT yield an O(1e3) loss."""
    torch.manual_seed(1)
    y = 2000.0 * torch.randn(2, 1, 16, 16, dtype=torch.complex64)
    pred = y + 100.0 * torch.randn(2, 1, 16, 16, dtype=torch.complex64)

    loss = ConcreteTTOStrategy._normalized_dc_loss(pred, y)
    assert loss.item() < 1.0, (
        f"normalized DC loss should be O(1) even at raw k-space scale; "
        f"got {loss.item()} (the pre-fix unnormalized loss was ~7673)"
    )


def test_normalized_dc_loss_zero_on_perfect_match() -> None:
    y = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
    loss = ConcreteTTOStrategy._normalized_dc_loss(y.clone(), y)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_normalized_dc_loss_handles_zero_measurement() -> None:
    """The 1e-8 clamp prevents a divide-by-zero on an all-zero measurement."""
    y = torch.zeros(1, 1, 4, 4, dtype=torch.complex64)
    pred = torch.ones(1, 1, 4, 4, dtype=torch.complex64)
    loss = ConcreteTTOStrategy._normalized_dc_loss(pred, y)
    assert torch.isfinite(loss).all()


# ---------------------------------------------------------------------------
# _compute_losses_impl regression coverage
#
# Two fixes pinned here:
#   * Perf: the per-step train loss dict must NOT carry ``max_translation`` /
#     ``max_rotation_deg`` — those read ``MotionTrajectory`` properties that are
#     ``.item()``-backed (D2H GPU host-syncs) and must not run in the hot loop.
#     They are reported once per epoch in ``validation_step`` instead.
#   * Grad path: the bridge forward must NOT be wrapped in ``torch.no_grad()``,
#     so ``g_total_loss`` stays grad-connected to the trajectory θ̂ and the
#     trajectory actually trains.
# ---------------------------------------------------------------------------


class _IdentityBridgeGenerator(torch.nn.Module):
    """Tiny stand-in for the frozen HyperMambaUNet bridge.

    The prediction depends on BOTH the input k-space and the motion-corrupted
    fiducial, so gradients can flow back to θ̂ through the ``corrupted_fiducial``
    argument. Backbone weights are frozen (requires_grad=False) exactly as the
    real strategy freezes them.
    """

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Conv2d(2, 2, kernel_size=1, bias=False)
        for p in self.parameters():
            p.requires_grad = False

    def forward(
        self, input_batch: torch.Tensor, corrupted_fiducial: torch.Tensor
    ) -> torch.Tensor:
        return self.proj(input_batch) + corrupted_fiducial


def _make_tto_strategy(img_size: tuple[int, int] = (16, 16)) -> ConcreteTTOStrategy:
    """Build a TTO strategy with only the attrs ``_compute_losses_impl`` reads.

    Bypasses the DI training environment via ``__new__`` and installs the
    minimal attributes the method touches. ``generator_model`` is a base
    property returning ``self.env.generator`` (base.py:154/167), so a fake
    ``env`` exposing ``.generator`` and ``.device`` is installed.
    """
    from mriforge.infrastructure.physics.kinematic_forward import (
        KinematicForwardOperator,
    )
    from mriforge.infrastructure.physics.virtual_fiducial import (
        MotionTrajectory,
        VirtualFiducial,
    )

    strat = ConcreteTTOStrategy.__new__(ConcreteTTOStrategy)
    device = torch.device("cpu")
    # Plain device attribute (strategy is NOT an nn.Module).
    strat.device = device
    strat.kinematic_op = KinematicForwardOperator(im_size=img_size).to(device)
    strat.fiducial = VirtualFiducial(
        im_size=img_size, grid_spacing=4, sigma=2.0, learnable=False
    ).to(device)
    strat.motion_traj = MotionTrajectory(
        num_readout_lines=img_size[0], init_mode="zero"
    ).to(device)
    # Make θ̂ non-zero so the corrupted fiducial differs from the clean one and
    # gradients are non-degenerate.
    with torch.no_grad():
        strat.motion_traj.theta.add_(0.05)
    gen = _IdentityBridgeGenerator().to(device)
    # generator_model is a read-only base property reading env.generator.
    strat.env = type("_Env", (), {"generator": gen, "device": device})()
    strat._lambda_dc = 1.0
    strat._lambda_tv = 0.1
    return strat


def _make_kspace_input(img_size: tuple[int, int] = (16, 16)) -> torch.Tensor:
    """Real-valued [B, 2, H, W] interleaved real/imag k-space-shaped input."""
    torch.manual_seed(0)
    return torch.randn(1, 2, img_size[0], img_size[1])


def test_train_loss_dict_excludes_item_backed_motion_diagnostics() -> None:
    """Regression (perf): hot-path dict must drop ``.item()``-backed keys.

    Pre-fix code returned ``max_translation`` and ``max_rotation_deg`` (each
    built from a ``.item()`` property → D2H GPU host-sync) every gen step. This
    test FAILS on the pre-fix code and PASSES after their removal.
    """
    strat = _make_tto_strategy()
    inp = _make_kspace_input()

    losses = strat._compute_losses_impl(inp, inp, epoch=0)

    assert "max_translation" not in losses, (
        "max_translation must not be emitted from the per-step train loss dict "
        "(it is .item()-backed → GPU host-sync in the hot loop)."
    )
    assert "max_rotation_deg" not in losses, (
        "max_rotation_deg must not be emitted from the per-step train loss dict "
        "(it is .item()-backed → GPU host-sync in the hot loop)."
    )
    # The hot-path-safe keys remain.
    assert set(losses) == {"g_total_loss", "loss_dc", "loss_tv"}
    # Detached diagnostics must not carry grad.
    assert not losses["loss_dc"].requires_grad
    assert not losses["loss_tv"].requires_grad


def test_g_total_loss_is_grad_connected_to_motion_trajectory() -> None:
    """Regression (grad path): θ̂ must receive gradient from the total loss.

    Verifies the bridge forward is NOT wrapped in ``torch.no_grad()`` (the dead
    block that was removed) — grads flow θ̂ → fiducial → bridge, keeping
    ``g_total_loss`` grad-connected to ``motion_traj.theta``. FAILS on a
    ``no_grad``-wrapped forward (gradient would be ``None`` / zero).
    """
    strat = _make_tto_strategy()
    inp = _make_kspace_input()

    losses = strat._compute_losses_impl(inp, inp, epoch=0)
    g_total = losses["g_total_loss"]
    assert g_total.requires_grad, "g_total_loss must be grad-connected (trainable)."

    theta = strat.motion_traj.theta
    assert theta.requires_grad
    g_total.backward()
    assert theta.grad is not None, "motion_traj.theta received no gradient."
    assert torch.isfinite(theta.grad).all()
    assert theta.grad.abs().sum().item() > 0.0, (
        "θ̂ gradient is exactly zero — the forward path is detached from the "
        "trajectory (regression: torch.no_grad() around the bridge forward)."
    )


def test_validation_step_still_reports_motion_diagnostics() -> None:
    """The motion diagnostics moved entirely to ``validation_step``.

    Ensures the diagnostics were not lost — they are still reported outside the
    hot loop, keyed ``max_translation_px`` / ``max_rotation_deg``.
    """
    strat = _make_tto_strategy()
    # validation_step uses self._to_device; provide a passthrough.
    strat._to_device = lambda x: x  # type: ignore[method-assign]
    inp = _make_kspace_input()

    metrics = strat.validation_step(inp, inp)

    assert "max_translation_px" in metrics
    assert "max_rotation_deg" in metrics
    assert "val_psnr" in metrics
    assert metrics["max_translation_px"] >= 0.0
    assert metrics["max_rotation_deg"] == pytest.approx(
        strat.motion_traj.max_rotation * 180 / math.pi, rel=1e-5
    )
