import math

import torch

from mriforge.infrastructure.training.utils.transform_ops import ComplexTensorHandler


def test_real_imag_to_complex():
    real = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    imag = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])

    comp = ComplexTensorHandler.real_imag_to_complex(real, imag)
    assert comp.is_complex()
    assert torch.allclose(comp.real, real)
    assert torch.allclose(comp.imag, imag)

def test_complex_to_real_imag():
    comp = torch.tensor([[1.0 - 1.0j, 2.0 - 2.0j]])
    real, imag = ComplexTensorHandler.complex_to_real_imag(comp)
    assert not real.is_complex()
    assert not imag.is_complex()
    assert torch.allclose(real, torch.tensor([[1.0, 2.0]]))
    assert torch.allclose(imag, torch.tensor([[-1.0, -2.0]]))

def test_magnitude_phase():
    real = torch.tensor([3.0, 0.0, -4.0])
    imag = torch.tensor([4.0, 5.0, 0.0])
    comp = torch.complex(real, imag)

    mag = ComplexTensorHandler.magnitude(comp)
    assert torch.allclose(mag, torch.tensor([5.0, 5.0, 4.0]))

    phase = ComplexTensorHandler.phase(comp)
    # math.atan2(y, x)
    expected_phase = torch.tensor([
        math.atan2(4.0, 3.0),
        math.atan2(5.0, 0.0),
        math.atan2(0.0, -4.0)
    ])
    assert torch.allclose(phase, expected_phase)

def test_tensor_to_complex():
    handler = ComplexTensorHandler()

    # [B, coils, H, W, 2]
    t1 = torch.randn(2, 3, 4, 4, 2)
    c1 = handler.tensor_to_complex(t1)
    assert c1.is_complex()
    assert c1.shape == (2, 3, 4, 4)

    # [B, 2*C, H, W]
    t2 = torch.randn(2, 6, 4, 4)
    c2 = handler.tensor_to_complex(t2)
    assert c2.is_complex()
    assert c2.shape == (2, 3, 4, 4)
    # check that channels were split properly (even=real, odd=imag)
    assert torch.allclose(c2.real, t2[:, ::2])
    assert torch.allclose(c2.imag, t2[:, 1::2])
