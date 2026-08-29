"""Regression tests for :mod:`synthetic_pathology_aug_strategy`.

Focus: the lesion-region-weighted L1 term must be live.

Pre-fix bug (finding 293): the child read ``patched.get("prediction")`` — a key
that is *never* written into the batch — so ``pred`` was always ``None``, the
guard ``if lesion_mask is not None and pred is not None and target is not None``
was always False, and ``loss_lesion_weighted`` (the strategy's only
paradigm-specific loss) was never added to ``g_total_loss``. That made the
strategy a silent no-op (pitfalls #9 / #10).

The fix introduces a prediction seam on the parent
(``ReconstructionTrainingStrategy._last_prediction`` / ``_last_target``, set in
``_compute_losses_impl``) and has the child read it. These tests pin that the
term is (a) present, (b) non-zero, and (c) carries grad back to the generator.
They are written so they FAIL on the pre-fix code and PASS now.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.infrastructure.training.strategies.synthetic_pathology_aug_strategy import (
    LesionSampler,
    SyntheticPathologyAugStrategy,
)


class _TinyGenerator(nn.Module):
    """Identity-ish conv so the produced prediction carries grad to params."""

    def __init__(self) -> None:
        super().__init__()
        # 1->1 channel 1x1 conv: differentiable, weights reachable from a loss.
        self.conv = nn.Conv2d(1, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.conv(x)


def _make_strategy() -> tuple[SyntheticPathologyAugStrategy, _TinyGenerator]:
    """Build a SyntheticPathologyAugStrategy without the heavy parent __init__.

    We bypass ``__init__`` (which needs a full TrainingSettings + environment)
    and instead stub the parent ``_compute_losses_impl`` to do exactly what the
    real one does w.r.t. the seam: run the (tiny) generator on the COMPOSITED
    input, set ``self._last_prediction`` / ``self._last_target`` to the
    grad-carrying output / target, and return a loss dict with ``g_total_loss``.
    This isolates the child's loss-folding logic (the bug locus, line 117).
    """
    strat = SyntheticPathologyAugStrategy.__new__(SyntheticPathologyAugStrategy)
    # Plain class-attr defaults the child declares.
    strat.p_augment = 1.0  # force augmentation so a lesion mask always exists
    strat.lambda_lesion_l1 = 0.5
    strat.lambda_lesion_cls = 0.1
    strat._sampler = None

    gen = _TinyGenerator()

    def _resolve_legacy_batch(input_batch, kwargs):  # noqa: ANN001
        # Mirror the real helper: the dict batch is forwarded under kwargs["batch"].
        if isinstance(input_batch, dict):
            return input_batch
        return kwargs.get("batch")

    def _parent_impl(*, input_batch, target_batch, epoch, **kwargs):  # noqa: ANN001
        # Run the generator on the composited input (grad-carrying).
        pred = gen(input_batch)
        total = (pred - target_batch).abs().mean()
        strat._last_prediction = pred  # the seam (NOT detached)
        strat._last_target = target_batch
        return {"g_total_loss": total, "loss": total}

    # Bind the parent's _compute_losses_impl via super(); easiest is to monkeypatch
    # the bound method the child calls through super().
    strat._resolve_legacy_batch = _resolve_legacy_batch  # type: ignore[method-assign]

    # The child calls super()._compute_losses_impl(...). Patch on the parent class
    # for the duration of the test by attaching to the instance is not enough
    # (super() resolves on the class). Instead we patch the parent method to the
    # instance-aware stub via the class MRO entry the child uses.
    from mriforge.infrastructure.training.strategies import reconstruction

    strat._test_parent_impl = _parent_impl  # type: ignore[attr-defined]
    return strat, gen


def _run_one_step(strat, gen, monkeypatch_parent):  # noqa: ANN001
    """Drive a single ``_compute_losses_impl`` call with a dict batch."""
    B, C, H, W = 2, 1, 16, 16
    batch = {
        "input": torch.randn(B, C, H, W),
        "target": torch.randn(B, C, H, W),
    }
    monkeypatch_parent(strat._test_parent_impl)
    out = strat._compute_losses_impl(input_batch=batch, batch=batch, epoch=0)
    return out


def test_loss_lesion_weighted_is_present_and_nonzero(monkeypatch) -> None:
    """The region-weighted term must appear and be > 0 (pre-fix it was absent)."""
    strat, gen = _make_strategy()
    from mriforge.infrastructure.training.strategies.reconstruction import (
        ReconstructionTrainingStrategy,
    )

    def _patch(impl):  # noqa: ANN001
        monkeypatch.setattr(
            ReconstructionTrainingStrategy,
            "_compute_losses_impl",
            lambda self, **kw: impl(**kw),
        )

    out = _run_one_step(strat, gen, _patch)

    assert "loss_lesion_weighted" in out, (
        "loss_lesion_weighted missing — the lesion-region-weighted term is dead "
        "(pred was read from a nonexistent batch['prediction'] key)."
    )
    assert out["loss_lesion_weighted"].item() > 0.0


def test_loss_lesion_weighted_folded_into_total(monkeypatch) -> None:
    """g_total_loss must include the region-weighted term."""
    strat, gen = _make_strategy()
    from mriforge.infrastructure.training.strategies.reconstruction import (
        ReconstructionTrainingStrategy,
    )

    def _patch(impl):  # noqa: ANN001
        monkeypatch.setattr(
            ReconstructionTrainingStrategy,
            "_compute_losses_impl",
            lambda self, **kw: impl(**kw),
        )

    out = _run_one_step(strat, gen, _patch)
    # Total must be at least the weighted term (it was folded in).
    assert out["g_total_loss"].item() >= out["loss_lesion_weighted"].item() - 1e-6


def test_loss_lesion_weighted_carries_grad_to_generator(monkeypatch) -> None:
    """Backprop from the weighted term must reach generator parameters.

    This is the key correctness assertion: the prediction operand must NOT be
    detached, otherwise the paradigm-specific loss trains nothing.
    """
    strat, gen = _make_strategy()
    from mriforge.infrastructure.training.strategies.reconstruction import (
        ReconstructionTrainingStrategy,
    )

    def _patch(impl):  # noqa: ANN001
        monkeypatch.setattr(
            ReconstructionTrainingStrategy,
            "_compute_losses_impl",
            lambda self, **kw: impl(**kw),
        )

    out = _run_one_step(strat, gen, _patch)

    for p in gen.parameters():
        if p.grad is not None:
            p.grad = None
    out["loss_lesion_weighted"].backward(retain_graph=True)
    grads = [p.grad for p in gen.parameters()]
    assert any(g is not None and torch.any(g != 0) for g in grads), (
        "loss_lesion_weighted produced no gradient on the generator — the "
        "prediction operand was detached (or the term is dead)."
    )


def test_lesion_sampler_shapes() -> None:
    """LesionSampler returns a composite image and a (B,1,H,W) binary mask."""
    sampler = LesionSampler()
    img = torch.zeros(3, 1, 8, 8)
    composite, mask = sampler(img)
    assert composite.shape == img.shape
    assert mask.shape == img.shape
    assert torch.all((mask == 0.0) | (mask == 1.0))
