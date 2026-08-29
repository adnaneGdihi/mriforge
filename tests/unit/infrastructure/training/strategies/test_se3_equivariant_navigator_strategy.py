"""Unit tests for SE3EquivariantNavigatorStrategy (B-3 of the VF campaign).

The full strategy ``__init__`` builds the DI container + AMP + a NUFFT-backed
KinematicForwardOperator (CUDA-only), so these tests exercise the strategy's
pure, device-agnostic helpers via ``__new__`` (bypassing the heavy base init).
Direct-import per the source<->test pairing rule.
"""

from __future__ import annotations

import torch

from mriforge.infrastructure.training.strategies.se3_equivariant_navigator_strategy import (
    SE3EquivariantNavigatorStrategy,
)


def _bare_strategy() -> SE3EquivariantNavigatorStrategy:
    """Construct without running BaseTrainingStrategy.__init__ (no DI/CUDA)."""
    s = SE3EquivariantNavigatorStrategy.__new__(SE3EquivariantNavigatorStrategy)
    s.device = torch.device("cpu")
    return s


def test_to_complex_from_real_interleaved() -> None:
    s = _bare_strategy()
    real = torch.arange(2 * 4 * 2 * 2, dtype=torch.float32).reshape(2, 4, 2, 2)
    cplx = s._to_complex(real)
    assert cplx.is_complex()
    assert cplx.shape == (2, 2, 2, 2)
    # interleaved layout: channel 0 -> real of coil 0, channel 1 -> imag of coil 0.
    assert torch.equal(cplx[:, 0].real, real[:, 0])
    assert torch.equal(cplx[:, 0].imag, real[:, 1])


def test_to_complex_passthrough_on_complex() -> None:
    s = _bare_strategy()
    c = torch.randn(2, 2, 4, 4, dtype=torch.complex64)
    assert s._to_complex(c) is c


def test_complex_to_real_round_trip() -> None:
    s = _bare_strategy()
    c = torch.randn(2, 3, 5, 5, dtype=torch.complex64)
    real = s._complex_to_real(c)
    assert real.shape == (2, 6, 5, 5)
    back = s._to_complex(real)
    assert torch.allclose(back.real, c.real, atol=1e-6)
    assert torch.allclose(back.imag, c.imag, atol=1e-6)


def test_apply_domain_losses_routes_by_domain() -> None:
    """Each loss is evaluated on the tensor domain it was declared under."""
    s = _bare_strategy()

    seen_domains: dict[str, str] = {}

    def make_probe(name: str):
        def probe(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            # complex domain => complex tensor; image => real magnitude.
            seen_domains[name] = "complex" if p.is_complex() else "real"
            return (p.abs() if p.is_complex() else p).mean() - (
                t.abs() if t.is_complex() else t
            ).mean()

        return probe

    s._loss_specs = [
        ("img_l1", make_probe("img_l1"), 1.0, "image"),
        ("cplx", make_probe("cplx"), 0.5, "complex"),
        ("ksp", make_probe("ksp"), 0.25, "kspace"),
    ]

    pred = torch.randn(2, 2, 8, 8)
    target = torch.randn(2, 2, 8, 8)
    g_total, loss_dict = s._apply_domain_losses(pred, target)

    assert seen_domains["img_l1"] == "real"
    assert seen_domains["cplx"] == "complex"
    assert seen_domains["ksp"] == "complex"  # fft2c output is complex
    assert "loss_img_l1" in loss_dict
    assert "loss_cplx" in loss_dict
    assert "loss_ksp" in loss_dict
    assert torch.is_tensor(g_total)


def test_apply_domain_losses_weighting() -> None:
    s = _bare_strategy()

    def const_loss(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.tensor(2.0)

    s._loss_specs = [
        ("a", const_loss, 1.0, "image"),
        ("b", const_loss, 3.0, "image"),
    ]
    pred = torch.randn(1, 2, 4, 4)
    target = torch.randn(1, 2, 4, 4)
    g_total, _ = s._apply_domain_losses(pred, target)
    # 1.0*2 + 3.0*2 == 8
    assert float(g_total) == 8.0


def test_strategy_inherits_required_mixins() -> None:
    from mriforge.infrastructure.training.strategies.base import BaseTrainingStrategy
    from mriforge.infrastructure.training.strategies.mixins.adversarial import (
        AdversarialMixin,
    )
    from mriforge.infrastructure.training.strategies.mixins.reconstruction import (
        ReconstructionMixin,
    )

    mro = SE3EquivariantNavigatorStrategy.__mro__
    assert BaseTrainingStrategy in mro
    assert ReconstructionMixin in mro
    assert AdversarialMixin in mro


def test_eq_defect_fed_latent_not_kspace_F10() -> None:
    """F10 (smoke audit 2026-06-04): the strategy fed the SE3 equivariance loss
    ``lie_input=corrupted_ksp_ri`` (k-space [B,2C,H,W]), which overrode the loss's
    ``u`` and tripped its [B,L,6] guard ("expects [B, L, 6]; got (4, 2, 256, 256)",
    exp_p3 dispatch task 21). The loss tests a latent-space map; it must be scored
    on the [B,L,6] trajectory latent ``xi``."""
    import pathlib

    src = pathlib.Path(
        "src/mriforge/infrastructure/training/strategies/se3_equivariant_navigator_strategy.py"
    ).read_text(encoding="utf-8")
    # The buggy call (now only quoted in an explanatory comment) passed the
    # k-space as the loss's equivariant_fn/lie_input — pin the distinctive
    # active fragment, which must no longer appear as live code.
    assert "equivariant_fn=gen.estimate_motion_latent" not in src, (
        "must not pass the k-space->latent fn / k-space lie_input to the SE3 "
        "loss — that crashed exp_p3 (dispatch task 21)"
    )
    assert "self._eq_defect(xi)" in src


def test_corrupt_and_forward_brings_target_to_image_domain() -> None:
    """F-kspace-real (smoke 2026-06-13): the svd-coil exp_p3 target is delivered
    as k-space; ``_corrupt_and_forward`` must IFFT it to image domain before the
    kinematic op / loss / cached visuals, or the REAL reference renders as raw
    |k-space| and the FAKE collapses to black. Pin the call site + the inherited
    seam's behaviour on a svd config."""
    import pathlib
    from types import SimpleNamespace

    src = pathlib.Path(
        "src/mriforge/infrastructure/training/strategies/se3_equivariant_navigator_strategy.py"
    ).read_text(encoding="utf-8")
    assert "self._ensure_image_domain_target(target_complex)" in src, (
        "_corrupt_and_forward must route the target through the k-space->image seam"
    )

    s = _bare_strategy()
    s.config = SimpleNamespace(
        model=SimpleNamespace(model_type="se3_equivariant_dc_navigator", target_domain="image"),
        data=SimpleNamespace(
            dataset_type="kspace",
            normalize_kspace=True,
            output_domain="image",
            coil_processing_mode="svd",
        ),
        physics=SimpleNamespace(kspace=SimpleNamespace(enable_kspace_recon=False)),
    )
    from mriforge.infrastructure.physics.fft_ops import ifft2c

    kspace = torch.randn(2, 1, 16, 16, dtype=torch.complex64)
    assert torch.allclose(s._ensure_image_domain_target(kspace), ifft2c(kspace), atol=1e-6)


def test_se3_loss_rejects_image_accepts_latent_F10() -> None:
    """Pin the loss contract the fix relies on: [B,L,6] latent is valid; a 4D
    image (the exp_p3 crash input) is rejected with a clear shape error."""
    import pytest

    from mriforge.models.losses.se3_equivariance_defect import (
        SE3EquivarianceDefectLoss,
    )

    loss = SE3EquivarianceDefectLoss(group="SO3", num_rotations=2, seed=0)
    out = loss(torch.randn(2, 4, 6))  # [B, L, 6] trajectory latent
    assert out.ndim == 0 and torch.isfinite(out)
    with pytest.raises(ValueError, match=r"\[B, L, 6\]"):
        loss(torch.randn(4, 2, 256, 256))  # the exp_p3 image that crashed
