r"""Unit tests for the LCAH acquisition hypernetwork encoder.

Targets ``spectramr.models.encoders.lcah_encoder``.

LCAH conditions a target encoder on the acquisition vector through a
hypernetwork, with both spectral-normalised so the extrapolation certificate
:math:`L_w L_h\lVert\boldsymbol\varphi-\boldsymbol\varphi^\ast\rVert` holds by
construction. The certificate is zero at a training acquisition (consistency) and
grows with distance.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit


def _encoder(**kw):
    from spectramr.models.encoders.lcah_encoder import LCAHEncoder

    torch.manual_seed(0)
    return LCAHEncoder(
        in_channels=1, acq_dim=5, hidden_channels=8, out_channels=2, **kw
    ).eval()


def test_forward_consumes_acquisition_vector() -> None:
    enc = _encoder()
    x = torch.randn(2, 1, 8, 8)
    acq = torch.tensor([[80.0, 3000.0, 0.0, 90.0, 3.0], [60.0, 2000.0, 0.0, 90.0, 0.064]])
    out = enc(x, acquisition=acq)
    assert out.shape == (2, 2, 8, 8)


def test_certified_radius_zero_at_training_acquisition() -> None:
    enc = _encoder(spectral_norm=True)
    phi_train = torch.tensor([[80.0, 3000.0, 0.0, 90.0, 3.0], [60.0, 2000.0, 0.0, 90.0, 1.5]])
    r = enc.certified_radius(torch.tensor([80.0, 3000.0, 0.0, 90.0, 3.0]), phi_train)
    assert float(r) == pytest.approx(0.0, abs=1e-5)


def test_certified_radius_positive_for_unseen_acquisition() -> None:
    enc = _encoder(spectral_norm=True)
    phi_train = torch.tensor([[80.0, 3000.0, 0.0, 90.0, 3.0]])
    r = enc.certified_radius(torch.tensor([80.0, 3000.0, 0.0, 90.0, 7.0]), phi_train)
    assert float(r) > 0.0


def test_spectral_norm_bounds_hypernetwork_lipschitz() -> None:
    """With spectral norm on, the hypernetwork's Lipschitz product is finite/small."""
    from spectramr.models.blocks.spectral_lipschitz import spectral_norm_product

    enc = _encoder(spectral_norm=True)
    lip_h = spectral_norm_product(enc.hypernet)
    assert lip_h < 5.0  # each spectral-normed layer is ~1-Lipschitz


def test_model_is_registered() -> None:
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY

    populate_model_registry()
    assert "lcah_encoder" in MODEL_REGISTRY
