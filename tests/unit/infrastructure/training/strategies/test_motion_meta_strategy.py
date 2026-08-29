"""Regression pins for ConcreteMotionMetaTrainingStrategy.

F-kspace-real (smoke audit 2026-06-13): the svd-coil ``experiment_vf_hyper_mamba_
meta`` arm delivers a *k-space* target (``dataset_type: kspace`` + svd coil-
processing → 2-channel real-interleaved k-space). The strategy applies the
kinematic motion operator + the reconstruction loss + cached visuals in image
domain, so the target MUST be IFFT'd to image domain first. The pre-fix code
asserted "target_complex is already an image" and dropped the IFFT — false for
k-space data: the REAL reference then rendered as raw |k-space| and the FAKE
collapsed to black.

Instantiating the strategy needs a full TrainingEnvironment + a CUDA NUFFT
kinematic operator, so the contract is pinned via source-text assertions plus a
behavioural check of the inherited ``_ensure_image_domain_target`` seam on CPU.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace

import torch

from mriforge.infrastructure.training.strategies.motion_meta_strategy import (
    ConcreteMotionMetaTrainingStrategy,
)

# Anchored to this file, not to the CWD. A bare ``pathlib.Path("src/...")``
# resolves against the process working directory, and this read happens at
# MODULE level -- so launching pytest from anywhere but the repo root raises
# FileNotFoundError during collection, and a collection error is not a test
# failure: pytest discards the whole session. Running from ``tests/`` was
# measured to abort the run with "Interrupted: 2 errors during collection",
# taking every unrelated test with it.
_SRC_PATH = (
    pathlib.Path(__file__).resolve().parents[5]
    / "src/mriforge/infrastructure/training/strategies/motion_meta_strategy.py"
)
_SRC = _SRC_PATH.read_text(encoding="utf-8")


def test_both_paths_convert_target_to_image_domain() -> None:
    """_compute_losses_impl AND validation_step must route the target through the
    k-space->image seam before the kinematic operator / loss / visuals."""
    assert _SRC.count("self._ensure_image_domain_target(target_complex)") >= 2, (
        "both _compute_losses_impl and validation_step must IFFT a k-space target"
    )


def test_stale_already_an_image_claim_is_gone() -> None:
    """The misleading comment that justified dropping the IFFT must not return."""
    assert "target_complex is already an\n        # an image" not in _SRC
    assert "already\n        # an image" not in _SRC
    # The specific false justification string must be absent.
    assert "the prior ifft2c(fft2c(.)) was an identity round-trip" not in _SRC


def test_conversion_precedes_kinematic_op() -> None:
    """In the training path the conversion happens before the corruption."""
    conv = _SRC.index("self._ensure_image_domain_target(target_complex)")
    kin = _SRC.index("self.kinematic_op(target_complex, theta)")
    assert conv < kin, "must convert the target to image domain BEFORE corrupting it"


def test_inherited_seam_iffts_svd_kspace_target() -> None:
    """Behavioural check of the inherited helper under a svd/kspace config."""
    s = ConcreteMotionMetaTrainingStrategy.__new__(ConcreteMotionMetaTrainingStrategy)
    s.config = SimpleNamespace(
        model=SimpleNamespace(model_type="hyper_mamba_unet", target_domain="image"),
        data=SimpleNamespace(
            dataset_type="kspace",
            normalize_kspace=False,
            output_domain="image",
            coil_processing_mode="svd",
        ),
        physics=SimpleNamespace(kspace=SimpleNamespace(enable_kspace_recon=False)),
    )
    from mriforge.infrastructure.physics.fft_ops import ifft2c

    kspace = torch.randn(2, 1, 16, 16, dtype=torch.complex64)
    out = s._ensure_image_domain_target(kspace)
    assert torch.allclose(out, ifft2c(kspace), atol=1e-6)
