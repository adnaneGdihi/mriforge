import pytest
import torch

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c


def test_parseval_theorem_ortho():
    """
    Verify Parseval's theorem for orthogonal FFT.
    With norm="ortho", sum(|k|^2) == sum(|x|^2).
    """
    # Create random complex image
    x = torch.randn(2, 4, 64, 64, dtype=torch.complex64)

    # Compute k-space
    k = fft2c(x)

    # Compute energy
    energy_x = torch.sum(torch.abs(x)**2)
    energy_k = torch.sum(torch.abs(k)**2)

    # Check relative difference
    scale = max(energy_x.item(), energy_k.item(), 1e-8)
    rel_diff = abs(energy_x.item() - energy_k.item()) / scale

    assert rel_diff < 1e-5, f"Parseval's theorem failed for fft2c! Rel diff: {rel_diff}, Ex: {energy_x}, Ek: {energy_k}"

def test_parseval_theorem_ifft_ortho():
    """
    Verify Parseval's theorem for orthogonal IFFT.
    """
    # Create random complex k-space
    k = torch.randn(2, 4, 64, 64, dtype=torch.complex64)

    # Compute image
    x = ifft2c(k)

    # Compute energy
    energy_x = torch.sum(torch.abs(x)**2)
    energy_k = torch.sum(torch.abs(k)**2)

    # Check relative difference
    scale = max(energy_x.item(), energy_k.item(), 1e-8)
    rel_diff = abs(energy_x.item() - energy_k.item()) / scale

    assert rel_diff < 1e-5, f"Parseval's theorem failed for ifft2c! Rel diff: {rel_diff}, Ex: {energy_x}, Ek: {energy_k}"

def test_parseval_bridge():
    """Bridges ``Sim2Rank.perm_preserves_sq_sum`` (+ ortho-DFT unitarity) <-> ``fft2c``.

    ``fft2c = ifftshift . fft2(norm='ortho') . fftshift``. The shifts are index
    permutations (``perm_preserves_sq_sum``: they preserve ``sum|x|^2``) and the
    ortho DFT is unitary, so ``fft2c`` preserves total L2 energy (Parseval).
    """
    torch.manual_seed(0)
    x = torch.randn(2, 3, 32, 32, dtype=torch.complex64)
    e_x = (x.abs() ** 2).sum()
    e_k = (fft2c(x).abs() ** 2).sum()
    assert torch.allclose(e_x, e_k, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
