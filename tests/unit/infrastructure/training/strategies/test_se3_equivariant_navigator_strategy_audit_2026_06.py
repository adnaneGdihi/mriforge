"""Regression tests for the 2026-06/07 audit fixes on SE3EquivariantNavigatorStrategy.

History of the seeded val-motion draw in ``_corrupt_and_forward``:

* 2026-06 [low]: the RNG restore covered only the CPU global generator,
  leaking the advanced CUDA RNG stream into training → fixed by bracketing
  BOTH global streams.
* 2026-07-01: the bracket was replaced by a SCOPED ``torch.Generator``
  threaded into ``generate_random_motion`` (the repo convention) — the
  bracket copied the full CUDA RNG state per seeded val call and still
  mutated global state mid-call. The invariant under test is unchanged:
  a seeded draw must leave the caller's global CPU/CUDA streams untouched.

These mirror the lightweight ``__new__`` construction used by the sibling test file
(``test_se3_equivariant_navigator_strategy.py``): the full ``__init__`` builds a
DI container + AMP + a NUFFT-backed operator (CUDA-only), so the device-agnostic
helpers are exercised on a bare instance.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from spectramr.infrastructure.training.strategies.se3_equivariant_navigator_strategy import (
    SE3EquivariantNavigatorStrategy,
)


def _image_domain_config() -> SimpleNamespace:
    """Minimal config so ``_ensure_image_domain_target`` resolves to a no-op.

    ``_corrupt_and_forward`` now routes the target through the base
    k-space->image seam, which reads ``self.config`` via
    ``needs_ifft_for_visualization``. These RNG-bracket tests are domain-agnostic,
    so an image-domain config makes the seam a passthrough and leaves the draw
    sequence under test untouched.
    """
    return SimpleNamespace(
        model=SimpleNamespace(
            model_type="se3_equivariant_dc_navigator", target_domain="image"
        ),
        data=SimpleNamespace(
            dataset_type="image",
            normalize_kspace=False,
            output_domain="image",
            coil_processing_mode=None,
        ),
        physics=SimpleNamespace(kspace=SimpleNamespace(enable_kspace_recon=False)),
    )


class _FakeKinematicOp:
    """Tiny stand-in for KinematicForwardOperator.

    Mirrors the real operator's RNG contract: when ``generator`` is given the
    draw uses it (global streams untouched); otherwise it consumes the global
    RNG via ``torch.rand`` on the requested device — so a strategy that fails
    to pass the scoped generator would be observable downstream.
    """

    def generate_random_motion(
        self,
        *,
        batch_size: int,
        max_translation: float,
        max_rotation: float,
        motion_type: str,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if generator is not None:
            return torch.rand(
                batch_size, 3, 8, generator=generator, device=generator.device
            ).to(device)
        return torch.rand(batch_size, 3, 8, device=device)

    def __call__(self, image: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        # Return something complex of the same shape; value is irrelevant.
        return image


class _FakeEnv:
    """Minimal env exposing only ``.generator`` (what ``generator_model`` reads)."""

    def __init__(self) -> None:
        # ``generator_model`` returns its first arg so the helper completes.
        self.generator = lambda corrupted_ri, kspace_measured=None: corrupted_ri


def _bare_strategy(device: torch.device) -> SE3EquivariantNavigatorStrategy:
    """Construct without running BaseTrainingStrategy.__init__ (no DI/CUDA init)."""
    s = SE3EquivariantNavigatorStrategy.__new__(SE3EquivariantNavigatorStrategy)
    s.device = device
    s.kinematic_op = _FakeKinematicOp()
    # ``generator_model`` is a read-only property backed by ``self.env.generator``.
    s.env = _FakeEnv()
    # ``_corrupt_and_forward`` routes the target through the base k-space->image
    # seam, which reads ``self.config``; an image-domain config makes it a no-op.
    s.config = _image_domain_config()
    return s


def test_seeded_corrupt_and_forward_restores_cpu_rng_state() -> None:
    """A seeded draw must not perturb the CPU global RNG stream of the caller."""
    s = _bare_strategy(torch.device("cpu"))

    target = torch.randn(2, 2, 8, 8, dtype=torch.complex64)

    # Snapshot the CPU RNG, run a seeded corrupt-and-forward, then draw a value.
    torch.manual_seed(99)
    before_state = torch.random.get_rng_state()
    expected_next = torch.rand(4)

    torch.manual_seed(99)
    s._corrupt_and_forward(target, seed=1234)
    after_seeded = torch.rand(4)

    # The seeded internal draw was fully bracketed, so the caller's stream is
    # exactly where it would have been with no intervening draw.
    assert torch.random.get_rng_state().shape == before_state.shape
    assert torch.equal(expected_next, after_seeded)


def test_unseeded_corrupt_and_forward_does_not_save_restore() -> None:
    """Without a seed there is no bracket; the helper must still run cleanly."""
    s = _bare_strategy(torch.device("cpu"))
    target = torch.randn(1, 2, 8, 8, dtype=torch.complex64)
    # Should not raise (no cpu_rng_state / cuda_rng_state referenced when seed=None).
    pred, target_ri, ksp_ri = s._corrupt_and_forward(target, seed=None)
    assert pred is not None
    assert target_ri.shape[1] == 2 * target.shape[1]


def test_corrupt_and_forward_uses_scoped_generator_not_global_bracket() -> None:
    """The seeded branch must use a scoped ``torch.Generator`` and never touch
    the global RNG APIs.

    Verified by bytecode-name inspection so the test passes on CPU-only
    machines: the old save/restore bracket (or a raw global ``manual_seed``
    without a scoped generator) reappearing would re-introduce either the
    per-call CUDA-state copy cost or the global-stream mutation.
    """
    src = SE3EquivariantNavigatorStrategy._corrupt_and_forward.__code__
    names = set(src.co_names)
    assert "Generator" in names, "seeded draw must build a scoped torch.Generator"
    for banned in (
        "get_rng_state",
        "set_rng_state",
        "get_rng_state_all",
        "set_rng_state_all",
    ):
        assert banned not in names, f"global RNG bracket API '{banned}' reappeared"


def test_seeded_corrupt_and_forward_is_reproducible() -> None:
    """Same seed → identical synthesized motion (and thus identical outputs)."""
    s = _bare_strategy(torch.device("cpu"))
    target = torch.randn(2, 2, 8, 8, dtype=torch.complex64)

    pred_a, _, ksp_a = s._corrupt_and_forward(target, seed=1234)
    pred_b, _, ksp_b = s._corrupt_and_forward(target, seed=1234)

    assert torch.equal(ksp_a, ksp_b)
    assert torch.equal(pred_a, pred_b)


@pytest.mark.gpu
def test_seeded_corrupt_and_forward_restores_cuda_rng_state() -> None:
    """On CUDA, the seeded draw must leave the device RNG stream unadvanced."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    s = _bare_strategy(device)
    target = torch.randn(2, 2, 8, 8, dtype=torch.complex64, device=device)

    # Pin the CUDA stream, then run a seeded corrupt-and-forward (which seeds + draws
    # on CUDA internally), then draw. Restored state => identical follow-on draw.
    torch.cuda.manual_seed_all(7)
    expected_next = torch.rand(4, device=device)

    torch.cuda.manual_seed_all(7)
    s._corrupt_and_forward(target, seed=1234)
    after_seeded = torch.rand(4, device=device)

    assert torch.equal(expected_next, after_seeded)
