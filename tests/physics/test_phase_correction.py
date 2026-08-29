import torch
import torch.fft


def test_phase_correlation_and_shift_theorem():
    """
    Mathematically prove that our Phase Correlation + Fourier Shift Theorem
    accurately aligns repetitions subjected to translational motion,
    preventing destructive interference in k-space.
    """
    torch.manual_seed(42)

    # 1. Create a simulated ground truth spatial image (64x64)
    # Give it some structure
    H, W = 64, 64
    y = torch.linspace(-1, 1, H)
    x = torch.linspace(-1, 1, W)
    Y, X = torch.meshgrid(y, x, indexing="ij")
    R = torch.sqrt(X**2 + Y**2)

    # Create a simple bright circle (phantom)
    gt_spatial = torch.zeros(H, W)
    gt_spatial[R < 0.3] = 1.0

    # GT k-space
    gt_kspace = torch.fft.fft2(gt_spatial)

    # 2. Simulate Repetitions with Noise and Spatial Shift
    noise_std = 0.5  # Add significant noise to k-space
    noise1 = torch.complex(
        torch.randn_like(gt_kspace.real) * noise_std,
        torch.randn_like(gt_kspace.imag) * noise_std,
    )
    noise2 = torch.complex(
        torch.randn_like(gt_kspace.real) * noise_std,
        torch.randn_like(gt_kspace.imag) * noise_std,
    )

    rep1_kspace = gt_kspace + noise1

    # Apply a spatial shift to rep2 (e.g., dy=3, dx=-2 pixels)
    shift_y, shift_x = 3, -2
    shifted_spatial = torch.roll(gt_spatial, shifts=(shift_y, shift_x), dims=(0, 1))
    rep2_drifted_kspace = torch.fft.fft2(shifted_spatial) + noise2

    # 3. Arithmetic Mean (Old Method - Destructive Interference)
    uncorrected_mean_kspace = torch.stack(
        [rep1_kspace, rep2_drifted_kspace], dim=0
    ).mean(dim=0)
    uncorrected_mean_spatial = torch.fft.ifft2(uncorrected_mean_kspace).real
    uncorrected_error = torch.norm(uncorrected_mean_spatial - gt_spatial)

    # 4. Phase Correlation + Fourier Shift Theorem (New Method)
    k_ref = rep1_kspace.unsqueeze(0)  # [1, H, W]
    k_curr = rep2_drifted_kspace.unsqueeze(0)  # [1, H, W]

    ky = torch.fft.fftfreq(H, device=k_ref.device)
    kx = torch.fft.fftfreq(W, device=k_ref.device)
    grid_ky, grid_kx = torch.meshgrid(ky, kx, indexing="ij")

    R_corr = torch.sum(k_ref * torch.conj(k_curr), dim=0)
    R_norm = R_corr / (torch.abs(R_corr) + 1e-8)

    r_spatial = torch.fft.ifftn(R_norm).real

    flat_idx = torch.argmax(r_spatial)
    dy, dx = torch.unravel_index(flat_idx, (H, W))

    # Phase correlation estimates the shift to match ref
    est_shift_y = dy.item() if dy < H // 2 else dy.item() - H
    est_shift_x = dx.item() if dx < W // 2 else dx.item() - W

    # Check that it correctly identified the mathematical shift
    # est_shift should be the translation needed to bring curr back to ref
    # Since we shifted curr by (shift_y, shift_x) relative to gt (and ref is ~ gt),
    # the needed translation to bring curr to ref is exactly (-shift_y, -shift_x).
    assert est_shift_y == -shift_y
    assert est_shift_x == -shift_x

    phase_ramp = torch.exp(
        -1j * 2 * torch.pi * (grid_kx * est_shift_x + grid_ky * est_shift_y)
    )

    k_curr_aligned = k_curr * phase_ramp.unsqueeze(0)

    corrected_mean_kspace = (
        torch.stack([k_ref, k_curr_aligned], dim=0).mean(dim=0).squeeze(0)
    )
    corrected_mean_spatial = torch.fft.ifft2(corrected_mean_kspace).real
    corrected_error = torch.norm(corrected_mean_spatial - gt_spatial)

    # 5. Assertions
    # Corrected should be significantly closer to ground truth than uncorrected
    assert (
        corrected_error < uncorrected_error
    ), f"Corrected Error: {corrected_error} > Uncorrected Error: {uncorrected_error}"

    # Prove SNR increases (noise averages out constructively)
    rep1_spatial = torch.fft.ifft2(rep1_kspace).real
    rep1_error = torch.norm(rep1_spatial - gt_spatial)
    assert corrected_error < rep1_error
