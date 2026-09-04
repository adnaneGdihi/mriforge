import torch

from spectramr.models.losses.pde_losses import HelmholtzPDELoss


def test_helmholtz_pde_loss():
    """Test Helmholtz PDE loss with k_sq=0 (Laplace equation).

    Uses S_real(x,y) = x^2 and S_imag(x,y) = y^2.
    Laplacian(x^2) = d^2(x^2)/dx^2 + d^2(x^2)/dy^2 = 2 + 0 = 2
    Laplacian(y^2) = d^2(y^2)/dx^2 + d^2(y^2)/dy^2 = 0 + 2 = 2
    Helmholtz residual (k_sq=0): mean(|2|^2 + |2|^2) = 8 per coil, mean over coils = 8
    """
    loss_fn = HelmholtzPDELoss(k_sq=0.0)

    batch_size = 5
    num_coils = 4
    coords = torch.rand(batch_size, 2, requires_grad=True)

    # S_real(x,y) = x^2 -> ∇²S_real = 2
    # S_imag(x,y) = y^2 -> ∇²S_imag = 2
    s_real = (coords[:, 0:1] ** 2).expand(-1, num_coils)
    s_imag = (coords[:, 1:2] ** 2).expand(-1, num_coils)

    loss = loss_fn(s_real, s_imag, coords)

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad is True
    # Residual = mean(2^2 + 2^2) = mean(4 + 4) = 8
    assert torch.allclose(loss, torch.tensor(8.0), atol=1e-3)


def test_helmholtz_pde_loss_k_sq():
    """Test Helmholtz PDE loss with non-zero k_sq.

    Uses S_real = x^2, S_imag = y^2 (quadratic, so autograd keeps full graph).
    Laplacian(x^2) = 2, Laplacian(y^2) = 2.
    Residual_real = laplacian_real + k_sq * S_real = 2 + k_sq * x^2
    Residual_imag = laplacian_imag + k_sq * S_imag = 2 + k_sq * y^2
    Loss = mean(res_r^2 + res_i^2) = mean((2 + k_sq*x^2)^2 + (2 + k_sq*y^2)^2)
    """
    k_sq = 1.5
    loss_fn = HelmholtzPDELoss(k_sq=k_sq)

    batch_size = 100  # Larger batch for more stable mean
    num_coils = 1
    coords = torch.rand(batch_size, 2, requires_grad=True)

    s_real = coords[:, 0:1] ** 2  # x^2
    s_imag = coords[:, 1:2] ** 2  # y^2

    loss = loss_fn(s_real, s_imag, coords)

    # Expected:
    # res_r = 2 + k_sq * x^2
    # res_i = 2 + k_sq * y^2
    expected_loss = torch.mean(
        (2.0 + k_sq * coords[:, 0] ** 2) ** 2 + (2.0 + k_sq * coords[:, 1] ** 2) ** 2
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad is True
    assert torch.allclose(loss, expected_loss, atol=1e-2)
