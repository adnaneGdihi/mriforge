import torch

from spectramr.infrastructure.physics.fft_ops import (
    fft2c,
    ifft2c,
    sense_adjoint,
    sense_forward,
)


def check_adjoint(forward_op, adjoint_op, x_shape, y_shape, name="Operator"):
    print(f"--- Checking Adjointness for {name} ---")

    # Generate random complex input x (Image Domain)
    x = torch.randn(*x_shape, dtype=torch.complex64)
    # Generate random complex output y (Measurement Domain)
    y = torch.randn(*y_shape, dtype=torch.complex64)

    # Ax
    Ax = forward_op(x)

    # A^H y
    AHz = adjoint_op(y)

    # Inner products
    # <Ax, y>
    inner1 = torch.sum(Ax * torch.conj(y))

    # <x, A^H y>
    inner2 = torch.sum(x * torch.conj(AHz))

    print(f"<Ax, y>   = {inner1.item()}")
    print(f"<x, A^H y> = {inner2.item()}")

    diff = torch.abs(inner1 - inner2)
    rel_error = diff / torch.abs(inner1)

    print(f"Diff: {diff.item():.2e}")
    print(f"Rel Error: {rel_error.item():.2e}")

    if rel_error < 1e-5:
        print("PASS")
        return True
    else:
        print("FAIL")
        return False


def main():
    torch.manual_seed(42)

    # 1. FFT2c / IFFT2c
    # Forward: Image -> Kspace
    # Adjoint: Kspace -> Image
    # Note: fft2c/ifft2c with norm='ortho' are unitary, so inverse is adjoint.
    # But usually <Fx, y> = <x, F^H y>. F^H is Inverse (for unitary).
    input_shape = (1, 1, 32, 32)
    check_adjoint(
        lambda x: fft2c(x),
        lambda y: ifft2c(y),
        input_shape,
        input_shape,
        name="FFT2c (Ortho)",
    )

    # 2. SENSE Operator
    # Forward: Image (1ch) -> Kspace (Coils)
    # Adjoint: Kspace (Coils) -> Image (1ch)
    B, C, H, W = 1, 1, 32, 32
    n_coils = 4

    # Generate random sensitivity maps
    smaps = torch.randn(B, n_coils, H, W, dtype=torch.complex64)
    # Normalize smaps usually expected but not strictly required for adjointness property itself
    # as long as operators use same smaps.

    # Define wrappers closing over smaps
    def A_sense(img):
        return sense_forward(img, smaps=smaps)

    def AH_sense(kspace):
        return sense_adjoint(kspace, smaps=smaps)

    check_adjoint(
        A_sense,
        AH_sense,
        (B, 1, H, W),  # Input image
        (B, n_coils, H, W),  # Measurement k-space
        name="SENSE",
    )

    # 3. Masked FFT
    mask = torch.rand(1, 1, 32, 32) > 0.5

    def A_masked(x):
        # Image -> Kspace -> Mask
        k = fft2c(x)
        return k * mask

    def AH_masked(y):
        # Mask -> Kspace -> Image (IFFT)
        # Adjoint of Sampling Operator S is S^T (fill zeros).
        # Adjoint of FFT is IFFT.
        # So A^H = IFFT( Mask * y )
        # Since mask is real diagonal, M^H = M.
        return ifft2c(y * mask)

    check_adjoint(A_masked, AH_masked, input_shape, input_shape, name="Masked FFT")


if __name__ == "__main__":
    main()
