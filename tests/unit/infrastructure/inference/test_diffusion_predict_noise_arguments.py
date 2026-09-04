"""Regression: ``_predict_noise`` must hand the model the evolving sample (#1030).

Two defects lived in one function, both instances of the caller and the callee
disagreeing about **who concatenates**. ``efficient_predict_noise`` owns the
concat -- it does ``torch.cat([x_t, conditioning], dim=1)`` itself -- so a caller
that pre-concatenates, or that substitutes one argument for the other, corrupts
the model input while raising nothing:

1. The plain conditional branch passed ``conditioning`` in the ``x_t`` slot and
   ``None`` as conditioning. The denoiser then never saw the evolving sample: on
   every step of the reverse loop it received the same fixed tensor, so the DDPM
   recursion degenerated into a map of a constant. Output stayed plausible --
   the conditioning *is* real data -- which is why no one caught it by eye.
2. The classifier-free-guidance branch pre-concatenated ``[x_t, conditioning]``
   and passed **that** as conditioning, so the callee concatenated a second time
   and the model received ``[x_t, x_t, conditioning]`` -- 3C channels.

Both are fixed by threading ``x_t`` and ``conditioning`` through their own
parameters and letting the callee do the single concat it already does.
"""

from __future__ import annotations

import torch

from spectramr.infrastructure.inference.diffusion_inference_strategy import (
    DiffusionInferenceStrategy,
)

C = 2  # conditioning/sample channel count; 2C and 3C must stay distinguishable


class _Recorder:
    """Stands in for the performance optimizer and records every call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def run_optimized_inference(self, model, x_t, t, conditioning=None):
        self.calls.append({"x_t": x_t, "t": t, "conditioning": conditioning})
        return torch.zeros_like(x_t)


def _strategy(*, cfg: bool, guidance: float = 1.0) -> tuple:
    strat = DiffusionInferenceStrategy.__new__(DiffusionInferenceStrategy)
    strat.model = object()
    strat.use_classifier_free_guidance = cfg
    strat.guidance_scale = guidance
    rec = _Recorder()
    strat.performance_optimizer = rec
    return strat, rec


def test_evolving_sample_reaches_the_denoiser() -> None:
    """The sample slot must carry x_t, and must differ as x_t does.

    This is the assertion the defect could not pass: it fed the same
    conditioning tensor at every timestep.
    """
    strat, rec = _strategy(cfg=False)
    cond = torch.randn(1, C, 8, 8)
    steps = [torch.full((1, C, 8, 8), float(i)) for i in range(3)]

    for i, x_t in enumerate(steps):
        strat._predict_noise(x_t, i, cond)

    seen = [c["x_t"] for c in rec.calls]
    assert len(seen) == 3
    for i, (got, expected) in enumerate(zip(seen, steps, strict=True)):
        assert torch.equal(got, expected), (
            f"step {i}: the denoiser was handed a tensor that is not x_t -- "
            "the reverse loop's state never reaches the model"
        )
    assert not torch.equal(seen[0], seen[1]), (
        "the sample handed to the denoiser did not change between steps, so the "
        "recursion is driven by a constant"
    )


def test_conditioning_is_passed_once_not_pre_concatenated() -> None:
    """The callee owns the concat, so the caller passes C channels, never 2C."""
    strat, rec = _strategy(cfg=False)
    cond = torch.randn(1, C, 8, 8)
    strat._predict_noise(torch.randn(1, C, 8, 8), 0, cond)

    passed = rec.calls[0]["conditioning"]
    assert passed is not None, "conditioning was dropped entirely"
    assert passed.shape[1] == C, (
        f"caller pre-concatenated: passed {passed.shape[1]} channels where the "
        f"callee expects {C} and concatenates to {2 * C} itself"
    )
    assert torch.equal(passed, cond)


def test_classifier_free_guidance_does_not_triple_the_channels() -> None:
    """CFG passed ``cat([x_t, conditioning])`` AS conditioning -> 3C at the model."""
    strat, rec = _strategy(cfg=True, guidance=2.0)
    cond = torch.randn(1, C, 8, 8)
    x_t = torch.randn(1, C, 8, 8)
    strat._predict_noise(x_t, 0, cond)

    assert len(rec.calls) == 2, "CFG must make a conditional and an unconditional call"
    for call in rec.calls:
        assert torch.equal(call["x_t"], x_t), "CFG lost the evolving sample too"
        assert call["conditioning"].shape[1] == C, (
            f"CFG passed {call['conditioning'].shape[1]} channels; the callee "
            f"concatenates, so anything but {C} reaches the model as 3C"
        )
    # The unconditional leg is the zero-conditioning one, and it is still C-wide.
    assert torch.count_nonzero(rec.calls[1]["conditioning"]) == 0


def test_unconditional_path_passes_no_conditioning() -> None:
    """Absence of conditioning is unchanged -- x_t through, conditioning None."""
    strat, rec = _strategy(cfg=False)
    x_t = torch.randn(1, C, 8, 8)
    strat._predict_noise(x_t, 0, None)
    assert torch.equal(rec.calls[0]["x_t"], x_t)
    assert rec.calls[0]["conditioning"] is None
