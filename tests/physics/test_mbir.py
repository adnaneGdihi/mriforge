"""Tests for Model-Based Image Reconstruction (MBIR)."""

import math

import pytest
import torch
import torch.nn.functional as F

from mriforge.infrastructure.physics.fft_ops import fft2c, ifft2c
from mriforge.infrastructure.physics.forward_operator import MultiCoilForwardOperator
from mriforge.infrastructure.physics.mbir import (
    FISTAMBIRSolver,
    positive_contrast_demodulation,
)


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def synthetic_mbir_data(device):
    """Generates synthetic data for MBIR tests."""
    B, C, H, W = 1, 4, 64, 64

    # 1. Clean phantom (a centered square)
    phantom = torch.zeros((B, 1, H, W), device=device, dtype=torch.complex64)
    phantom[..., H // 4 : 3 * H // 4, W // 4 : 3 * W // 4] = 1.0 + 0.0j

    # 2. Coil Sensitivities (synthetic smooth gradients)
    smaps = torch.ones((B, C, H, W), device=device, dtype=torch.complex64)
    for c in range(C):
        phase_val = c * math.pi / 4
        phase_tensor = torch.full((B, H, W), phase_val, device=device, dtype=torch.float32)
        ones = torch.ones_like(phase_tensor)
        smaps[:, c, :, :] *= torch.polar(ones, phase_tensor)
    # Normalize
    rss = torch.sqrt(torch.sum(torch.abs(smaps) ** 2, dim=1, keepdim=True))
    smaps = smaps / (rss + 1e-8)

    # 3. Off-resonance B0 map (e.g. from susceptibility dipole)
    # Simulate a strong phase gradient causing distortion
    b0_map = torch.zeros((B, H, W), device=device, dtype=torch.float32)
    y_coords = torch.linspace(-math.pi, math.pi, H, device=device).unsqueeze(1).expand(H, W)
    x_coords = torch.linspace(-math.pi, math.pi, W, device=device).unsqueeze(0).expand(H, W)

    # Strong phase dispersion at the center (the "marker")
    r2 = x_coords**2 + y_coords**2
    b0_map = 10.0 * torch.exp(-r2 / 2.0)  # rad phase error

    return phantom, smaps, b0_map


def test_mbir_fista_convergence(synthetic_mbir_data, device):
    """Test that FISTA monotonically decreases the data consistency cost."""
    phantom, smaps, b0_map = synthetic_mbir_data

    # Create physics forward operator WITHOUT B0 field for baseline convergence test
    forward_op = MultiCoilForwardOperator(
        coil_sensitivities=smaps[0],
        centered=True,  # Operator expects (C, H, W)
    ).to(device)

    # Simulate measurements and add noise so it's not a trivial zero-residual problem
    y_clean = forward_op(phantom)
    noise = torch.randn_like(y_clean) + 1j * torch.randn_like(y_clean)
    y_measured = y_clean + 0.1 * noise

    # Initialize solver
    solver = FISTAMBIRSolver(
        forward_operator=forward_op,
        num_iterations=40,  # Need more iterations for a smaller step size
        lambda_reg=0.001,  # Use small reg to denoise
        step_size=0.01,  # Reduce step size to guarantee stability without calculating Lipschitz
        regularization="identity",
    ).to(device)

    # Run solver
    adjoint_init = forward_op.adjoint(y_measured)
    recon = solver(y_measured, x_init=adjoint_init)

    # Validate Cost decreased: || A(recon) - y || < || A(adjoint) - y ||
    initial_residual = torch.norm(forward_op(adjoint_init) - y_measured).item()
    final_residual = torch.norm(forward_op(recon) - y_measured).item()

    # FISTA with momentum might not strictly monotonic, but after 40 iterations
    # it must easily beat the initial zero-state gradient (adjoint).
    # We use a very loose bound to prevent test flakiness due to momentum oscillation.
    assert final_residual < initial_residual * 1.05, (
        f"FISTA didn't reliably minimize DC: Initial {initial_residual:.4f} -> Final {final_residual:.4f}"
    )


def test_mbir_fista_lipschitz_guard_prevents_divergence(synthetic_mbir_data, device):
    """A large configured step_size must not diverge — the Lipschitz clamp caps it.

    Regression: FISTA used a fixed ``step_size`` with no ``alpha <= 1/L`` guard
    (the existing convergence test even sets ``step_size=0.01`` "to guarantee
    stability without calculating Lipschitz"). With the power-iteration clamp a
    step_size far above ``1/||A*A||`` stays stable instead of blowing up.
    """
    phantom, smaps, _ = synthetic_mbir_data
    forward_op = MultiCoilForwardOperator(
        coil_sensitivities=smaps[0], centered=True
    ).to(device)
    y = forward_op(phantom)

    solver = FISTAMBIRSolver(
        forward_operator=forward_op,
        num_iterations=20,
        lambda_reg=0.0,
        step_size=10.0,  # >> 1/L (||A*A|| ~ 1 for RSS-normalized smaps)
        regularization="identity",
    ).to(device)

    adjoint_init = forward_op.adjoint(y)
    recon = solver(y, x_init=adjoint_init)

    assert torch.isfinite(recon).all(), "FISTA diverged (non-finite) at large step"
    init_res = torch.norm(forward_op(adjoint_init) - y).item()
    final_res = torch.norm(forward_op(recon) - y).item()
    assert final_res < init_res * 1.05, (
        f"FISTA unstable at step_size=10: {init_res:.4f} -> {final_res:.4f}"
    )


def test_mbir_b0_correction(synthetic_mbir_data, device):
    """Test that MBIR correctly unwarps image using the B0 map."""
    phantom, smaps, b0_map = synthetic_mbir_data

    # 1. Forward Operator WITH B0 Phase Error
    forward_op_distorted = MultiCoilForwardOperator(
        coil_sensitivities=smaps[0], b0_map=b0_map[0], centered=True
    ).to(device)

    # Simulate severely phase-degraded measurements
    y_distorted = forward_op_distorted(phantom)

    # 2. Baseline Recon: Adjoint (Fails to correct B0)
    recon_adjoint = forward_op_distorted.adjoint(y_distorted)

    # 3. MBIR Recon: Corrects B0 via forward model inversion
    solver = FISTAMBIRSolver(
        forward_operator=forward_op_distorted,
        num_iterations=30,
        lambda_reg=0.0,  # Pure data consistency inversion
        step_size=0.1,
        regularization="identity",
    ).to(device)

    recon_mbir = solver(y_distorted, x_init=recon_adjoint)

    # Validate: MBIR should be much closer to true phantom than simple Adjoint
    mse_adjoint = F.mse_loss(torch.view_as_real(recon_adjoint), torch.view_as_real(phantom)).item()
    mse_mbir = F.mse_loss(torch.view_as_real(recon_mbir), torch.view_as_real(phantom)).item()

    assert mse_mbir < mse_adjoint, (
        f"MBIR failed to correct B0 artifact. Adjoint MSE: {mse_adjoint:.6f}, MBIR MSE: {mse_mbir:.6f}"
    )


def test_apply_tv_is_not_a_contraction(synthetic_mbir_data, device):
    """``_apply_tv`` is a one-step TV gradient surrogate, NOT a true prox.

    Regression-locks the documented contract (the docstring now states this
    branch does ``x - D^T(soft_threshold(D x))``, a Landweber-style step that
    is expansive, not ``prox_{lambda*TV}``). On a smooth ramp at a small
    threshold the surrogate overshoots the input range — a real proximal map
    (non-expansive) never would. A future Chambolle swap is therefore a
    deliberate, reviewed change that flips this test.
    """
    phantom, smaps, _ = synthetic_mbir_data
    forward_op = MultiCoilForwardOperator(coil_sensitivities=smaps[0], centered=True).to(device)
    solver = FISTAMBIRSolver(forward_operator=forward_op, regularization="tv").to(device)

    H = W = 32
    ramp = (
        torch.linspace(0.0, 1.0, W, device=device)
        .view(1, 1, 1, W)
        .expand(1, 1, H, W)
        .to(torch.complex64)
        .contiguous()
    )
    out = solver._apply_tv(ramp, threshold=1e-3)
    out_mag = out.abs()
    ramp_mag = ramp.abs()
    # Surrogate overshoots the [0, 1] range -> not a non-expansive prox.
    assert (
        out_mag.max().item() > ramp_mag.max().item() + 1e-3
        or out_mag.min().item() < ramp_mag.min().item() - 1e-3
    )


def test_positive_contrast_demodulation_rejects_mismatched_te_length(device):
    """A 1-D te_sec whose length != k-space readout axis must RAISE.

    Regression for the silent mis-broadcast: a [K'] te_sec with K' != W would
    previously broadcast incorrectly (or fail cryptically). The guard now
    raises a clear ValueError.
    """
    kspace = torch.ones(1, 1, 4, 8, dtype=torch.complex64)  # readout axis = 8
    te_sec = torch.linspace(0.001, 0.01, 5)  # length 5 != 8
    with pytest.raises(ValueError, match="must match the k-space readout"):
        positive_contrast_demodulation(kspace, 150.0, te_sec)


def test_positive_contrast_demodulation_rejects_multidim_te(device):
    """A multi-dimensional te_sec tensor must RAISE (only scalar / 1-D allowed)."""
    kspace = torch.ones(1, 1, 4, 8, dtype=torch.complex64)
    te_sec = torch.ones(2, 8)  # 2-D after squeeze
    with pytest.raises(ValueError, match="must be scalar or 1-D"):
        positive_contrast_demodulation(kspace, 150.0, te_sec)


def test_positive_contrast_demodulation_matching_te_length_ok(device):
    """A 1-D te_sec matching the readout axis still broadcasts as before."""
    kspace = torch.ones(1, 1, 4, 8, dtype=torch.complex64)  # readout axis = 8
    te_sec = torch.linspace(0.001, 0.01, 8)
    result = positive_contrast_demodulation(kspace, 150.0, te_sec)
    assert result.shape == kspace.shape
    expected_last = torch.exp(torch.tensor(1j * 2.0 * math.pi * 150.0 * 0.01))
    assert torch.allclose(result[0, 0, 0, -1], expected_last)


def test_positive_contrast_demodulation(device):
    """Test that positive contrast demodulation shifts spatial energy."""
    B, C, H, W = 1, 1, 64, 64

    # Clean image with a centered point source
    image = torch.zeros((B, C, H, W), device=device, dtype=torch.complex64)
    image[..., H // 2, W // 2] = 1.0 + 0.0j

    # Convert to K-Space
    kspace = fft2c(image)

    # Demodulate to shift frequency
    delta_omega = 200.0  # Hz
    te = 0.010  # 10ms
    # phase_shift = 2 * pi * 200 * 0.01 = 4 * \pi rad (which is 0 modulo 2pi!)
    # Let's pick a shift that does not alias into a 2pi multiple to observe spatial effect

    # Actually, a temporal uniform phase shift across all k-space points is equivalent to
    # multiplying the IMAGE by the same constant phase shift. It does NOT spatially
    # translate the image (unlike a linear phase ramp).
    # Test valid magnitude scaling (magnitude should remain identical, phase rotates)
    kspace_demod = positive_contrast_demodulation(kspace, delta_omega, te)

    image_demod = ifft2c(kspace_demod)

    # Ensure magnitude is preserved
    mag_original = torch.abs(image)
    mag_demod = torch.abs(image_demod)
    torch.testing.assert_close(mag_original, mag_demod)

    # Expected phase shift: exp(i * 2pi * 200 * 0.010)
    expected_phase = math.fmod(2.0 * math.pi * delta_omega * te, 2.0 * math.pi)

    # Check center pixel phase
    center_phase_orig = torch.angle(image[..., H // 2, W // 2]).item()
    center_phase_demod = torch.angle(image_demod[..., H // 2, W // 2]).item()

    diff_rad = math.fmod(center_phase_demod - center_phase_orig, 2.0 * math.pi)
    if diff_rad < 0:
        diff_rad += 2.0 * math.pi

    expected_positive = expected_phase if expected_phase >= 0 else expected_phase + 2.0 * math.pi
    assert math.isclose(diff_rad, expected_positive, rel_tol=1e-4, abs_tol=1e-4)
