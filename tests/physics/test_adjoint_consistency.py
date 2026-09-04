import pytest
import torch

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c


def check_adjoint(A, A_adjoint, x_shape, y_shape, dtype=torch.complex64, device="cpu", tol=1e-5):
    """
    Check if A_adjoint is the true adjoint of A.
    <Ax, y> = <x, A^H y>
    """
    x = torch.randn(x_shape, dtype=dtype, device=device)
    y = torch.randn(y_shape, dtype=dtype, device=device)

    Ax = A(x)
    A_adj_y = A_adjoint(y)

    # <Ax, y> = sum(Ax * conj(y))
    # <x, A^H y> = sum(x * conj(A^H y))

    # Check inner products via real dot products of flattened arrays
    ip1 = torch.sum(Ax * y.conj())
    ip2 = torch.sum(x * A_adj_y.conj())

    diff = torch.abs(ip1 - ip2)

    # Relative difference
    scale = max(torch.abs(ip1).item(), torch.abs(ip2).item(), 1e-8)
    rel_diff = diff.item() / scale

    return rel_diff < tol, diff.item(), ip1.item(), ip2.item()

def test_fft2c_adjoint():
    """Verify that ifft2c is the exact adjoint of fft2c."""
    x_shape = (2, 8, 32, 32)
    y_shape = (2, 8, 32, 32)

    A = lambda x: fft2c(x)
    A_adjoint = lambda y: ifft2c(y)

    is_adjoint, diff, ip1, ip2 = check_adjoint(A, A_adjoint, x_shape, y_shape)

    assert is_adjoint, f"fft2c adjoint failed! Diff: {diff}, <Ax,y>: {ip1}, <x,A^Hy>: {ip2}"

def test_sense_adjoint_bridge():
    """Bridges ``Sim2Rank.diag_adjoint`` <-> ``sense_forward`` / ``sense_adjoint``.

    The composite forward ``A = M F S`` (mask, centered FFT, coil maps) has adjoint
    ``A^H = conj(S) F^-1 M``. ``diag_adjoint`` certifies the diagonal factors' adjoint
    is conjugation; here we check the full SENSE operator numerically:
    ``<A x, y> = <x, A^H y>``.
    """
    from spectramr.infrastructure.physics.fft_ops import sense_adjoint, sense_forward

    torch.manual_seed(0)
    B, C, H, W = 1, 4, 16, 16
    smaps = torch.randn(B, C, H, W, dtype=torch.complex64)
    mask = (torch.rand(B, 1, H, W) > 0.4).float()
    x = torch.randn(B, 1, H, W, dtype=torch.complex64)
    y = torch.randn(B, C, H, W, dtype=torch.complex64)
    ax = sense_forward(x, smaps=smaps, mask=mask)
    ahy = sense_adjoint(y, smaps=smaps, mask=mask)
    ip1 = torch.sum(ax * y.conj())
    ip2 = torch.sum(x * ahy.conj())
    assert torch.allclose(ip1, ip2, rtol=1e-4, atol=1e-4), f"{ip1} vs {ip2}"


if __name__ == "__main__":
    pytest.main([__file__])
