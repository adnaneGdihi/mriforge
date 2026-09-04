"""Unit tests for ``BaseTrainingStrategy._ensure_image_domain_target``.

This is the SSOT seam that brings a *k-space-delivered* dataloader target into
image domain for image-domain strategies (loss / val_psnr / cached visuals are
defined on images). It was added to kill the "k-space-as-the-real-image + black
fake" failure on the svd-coil image-output VF arms exp_p3 / hyper_mamba_meta /
method_c (smoke audit 2026-06-13).

The decision is delegated to ``needs_ifft_for_visualization`` so that arms whose
``coil_processing_mode`` already emits an image (``rss_image`` / ``magnitude``)
are NOT re-FFT'd — the mirror-image regression. These tests pin both the domain
decision and the representation-preserving IFFT.

The strategy ``__init__`` builds the DI container; we bypass it via ``__new__``
(BaseTrainingStrategy has no abstractmethods) and exercise the pure helper on CPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from spectramr.infrastructure.physics.fft_ops import ifft2c
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from tests.utils.data_config_stub import DataConfigStub


def _cfg(*, dataset_type="kspace", coil_processing_mode=None, normalize_kspace=False):
    """Minimal config covering ``needs_ifft_for_visualization``'s cascade.

    ``data`` is the SHARED ``DataConfigStub``, not a hand-rolled namespace. The
    previous version spelled the coil mode FLAT (``data.coil_processing_mode``)
    while the schema moved it to ``data.coils.processing_mode``; the stub routes
    a flat legacy kwarg to its canonical home, so it cannot become a second
    spelling that silently disagrees with the one the reader walks.

    That drift is what left ``test_decision_is_cached`` red on ``dev`` (#826) --
    it mutates ``config.data.coils.processing_mode``, which the flat stub did not
    have. Same class as the ``test_simulator_builder.py`` fixture fixed in #825.
    """
    return SimpleNamespace(
        model=SimpleNamespace(model_type="standard_unet", target_domain="image"),
        data=DataConfigStub(
            dataset_type=dataset_type,
            normalize_kspace=normalize_kspace,
            output_domain="image",
            coil_processing_mode=coil_processing_mode,
        ),
        physics=SimpleNamespace(kspace=SimpleNamespace(enable_kspace_recon=False)),
    )


def _bare(cfg) -> BaseTrainingStrategy:
    s = BaseTrainingStrategy.__new__(BaseTrainingStrategy)
    s.config = cfg
    return s


def test_svd_kspace_complex_target_is_ifft_to_image() -> None:
    """svd coil-processing delivers k-space → the complex target must be IFFT'd."""
    s = _bare(_cfg(dataset_type="kspace", coil_processing_mode="svd"))
    kspace = torch.randn(2, 1, 16, 16, dtype=torch.complex64)
    out = s._ensure_image_domain_target(kspace)
    assert out.is_complex()
    assert torch.allclose(out, ifft2c(kspace), atol=1e-6)
    # And it is genuinely different from the input (a real IFFT happened).
    assert not torch.allclose(out, kspace)


def test_rss_image_target_is_passthrough() -> None:
    """coil_processing_mode=rss_image already emits an image → must NOT IFFT.

    Regression guard: a naive ``dataset_type == 'kspace'`` check would re-FFT the
    image into k-space (the mirror of the original bug). The SSOT
    needs_ifft_for_visualization folds in the coil-mode override.
    """
    s = _bare(_cfg(dataset_type="kspace", coil_processing_mode="rss_image"))
    img = torch.randn(2, 1, 16, 16, dtype=torch.complex64)
    out = s._ensure_image_domain_target(img)
    assert out is img  # untouched


def test_image_dataset_is_passthrough() -> None:
    s = _bare(_cfg(dataset_type="image", coil_processing_mode=None))
    img = torch.randn(2, 2, 8, 8)
    out = s._ensure_image_domain_target(img)
    assert out is img


def test_real_interleaved_kspace_preserves_representation() -> None:
    """A [B, 2C, H, W] real-interleaved k-space target → real-interleaved image."""
    s = _bare(_cfg(dataset_type="kspace", coil_processing_mode="svd"))
    real = torch.randn(2, 2, 8, 8)  # [R, I] of one complex coil
    out = s._ensure_image_domain_target(real)
    assert out.shape == real.shape
    assert not out.is_complex()
    expected = ifft2c(torch.complex(real[:, 0::2], real[:, 1::2]))
    assert torch.allclose(out[:, 0::2], expected.real, atol=1e-6)
    assert torch.allclose(out[:, 1::2], expected.imag, atol=1e-6)


def test_single_channel_real_passthrough_no_pair() -> None:
    """target_channels=1 strips the imaginary half → no complex pair to invert.

    The helper refuses to fabricate phase and passes the tensor through; such an
    arm must set data.target_channels: 2 (the config-side half of the fix).
    """
    s = _bare(_cfg(dataset_type="kspace", coil_processing_mode="svd"))
    one = torch.randn(2, 1, 8, 8)  # odd channel count, real
    out = s._ensure_image_domain_target(one)
    assert out is one


def test_decision_is_cached() -> None:
    """The (frozen-config) domain decision is computed once and cached.

    The config block is frozen, so the old form of this test -- assigning
    ``config.data.coils.processing_mode`` in place -- could only ever have run
    against a stand-in that was not the real schema. Swap the whole ``data``
    block for one that answers differently instead; if the cache were not
    consulted the second call would take the image-domain branch and pass the
    target through untouched.
    """
    s = _bare(_cfg(dataset_type="kspace", coil_processing_mode="svd"))
    assert getattr(s, "_target_needs_image_ifft", None) is None
    _ = s._ensure_image_domain_target(torch.randn(1, 1, 4, 4, dtype=torch.complex64))
    assert s._target_needs_image_ifft is True

    s.config = _cfg(dataset_type="kspace", coil_processing_mode="rss_image")

    # An uncached strategy on the new config takes the image-domain branch...
    passthrough = torch.randn(1, 1, 4, 4, dtype=torch.complex64)
    assert _bare(s.config)._ensure_image_domain_target(passthrough) is passthrough

    # ...while the cached one still IFFTs, which is the property under test.
    # Asserting only that the output is complex would not discriminate: the
    # input is already complex64, so BOTH branches satisfy it.
    target = torch.randn(1, 1, 4, 4, dtype=torch.complex64)
    assert s._ensure_image_domain_target(target) is not target
