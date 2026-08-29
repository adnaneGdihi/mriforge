"""Tests for BlochSynthesisStrategy (MICCAI MRIxFields2026, idea 2.1).

White-box: the full pipeline is exercised by the Tier-2 audit probe; here we pin the
inline objective (source-consistency + dispersion-prior + seg-consistency) and the
load-bearing claim that its weights are driven by the loss_schedule curriculum via
``loop_state.loss_weight_overrides``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mriforge.infrastructure.training.loop_state import LoopState
from mriforge.infrastructure.training.strategies.bloch_synth_strategy import (
    BlochSynthesisStrategy,
    _IntensitySoftSegmenter,
)
from mriforge.models.generators.relaxometry_encoder import RelaxometryEncoder
from mriforge.models.losses.bloch_synth_losses import (
    BlochSourceConsistencyLoss,
    DispersionPriorLoss,
)
from mriforge.models.losses.dice_anatomy_loss import SegmentationDiceLoss


def test_intensity_soft_segmenter_shape() -> None:
    seg = _IntensitySoftSegmenter(n_classes=6)
    logits = seg(torch.rand(2, 1, 8, 8))
    assert logits.shape == (2, 6, 8, 8)


def _make_strategy(seg_weight: float = 0.05) -> BlochSynthesisStrategy:
    s = object.__new__(BlochSynthesisStrategy)
    s.device = torch.device("cpu")
    s._source_consistency_weight = 1.0
    s._seg_consistency_weight = seg_weight
    s._dispersion_prior_weight = 0.1
    s._residual_weight = 0.25
    s._segmenter_backend = "label_dice"
    s._segmenter = _IntensitySoftSegmenter()
    s._source_loss = BlochSourceConsistencyLoss(weight=1.0)
    s._dispersion_prior = DispersionPriorLoss(weight=1.0)
    s._seg_loss = SegmentationDiceLoss(require_segmenter=True)
    s.last_dispersion_beta = None
    s.loop_state = LoopState()
    gen = RelaxometryEncoder(in_channels=3)
    s.env = SimpleNamespace(generator=gen, losses={})
    s.config = SimpleNamespace(
        training=SimpleNamespace(enforce_output_range=False),
        losses=SimpleNamespace(image_losses=[], kspace_losses=[], complex_losses=[]),
    )
    return s


def _batch():
    return {
        "input": torch.rand(2, 3, 24, 24),
        "target": torch.rand(2, 1, 24, 24),
        "field_strength": torch.tensor([0.1, 1.5]),
        "field_strength_target": torch.tensor([7.0, 7.0]),
    }


def test_inline_terms_present_and_finite() -> None:
    s = _make_strategy()
    out = s._compute_losses_impl(input_batch=_batch())
    assert torch.isfinite(out["loss_total"])
    for k in (
        "loss_bloch_source_consistency",
        "loss_dispersion_prior",
        "loss_seg_consistency",
    ):
        assert k in out
    assert s.last_dispersion_beta is not None


def test_seg_omitted_when_weight_zero() -> None:
    s = _make_strategy(seg_weight=0.0)
    out = s._compute_losses_impl(input_batch=_batch())
    assert "loss_seg_consistency" not in out


def test_curriculum_override_drives_seg_weight() -> None:
    """A loss_schedule override on 'seg_consistency' changes the objective (the model
    is unchanged between calls, so only the weight differs)."""
    s = _make_strategy()
    batch = _batch()

    s.loop_state.loss_weight_overrides = {}
    total_base = float(s._compute_losses_impl(input_batch=batch)["loss_total"])

    s.loop_state.loss_weight_overrides = {"seg_consistency": 5.0}
    total_hi = float(s._compute_losses_impl(input_batch=batch)["loss_total"])

    assert total_hi > total_base


def test_missing_target_field_raises() -> None:
    s = _make_strategy()
    bad = _batch()
    del bad["field_strength_target"]
    with pytest.raises(ValueError, match="field_strength_target"):
        s._compute_losses_impl(input_batch=bad)


def test_validation_forward_uses_training_residual_weight() -> None:
    # Regression: validation must blend the opaque residual at residual_weight (0.25),
    # matching training, NOT the model.forward default of 1.0 (else checkpoints misrank).
    s = _make_strategy()
    assert s._residual_weight == 0.25
    x = torch.rand(2, 3, 24, 24)
    bc = {"input": x, "field_strength_target": torch.tensor([7.0, 7.0])}
    y = s._validation_forward(x, bc)

    gen = s.env.generator
    params = gen.predict_parameters(x)
    y_det = gen.render(params, torch.tensor([7.0, 7.0]))
    resid = gen.opaque_residual(y_det, x)
    expected = (y_det + 0.25 * resid).clamp(0.0, 1.0)
    assert torch.allclose(y, expected, atol=1e-6)
    # ... and it differs from the weight-1.0 model.forward()
    y_full = gen(x, field_strength=torch.tensor([7.0, 7.0]))
    assert not torch.allclose(y, y_full)


def test_validation_forward_requires_target_field() -> None:
    s = _make_strategy()
    with pytest.raises(ValueError, match="field_strength_target"):
        s._validation_forward(torch.rand(2, 3, 8, 8), {"input": torch.rand(2, 3, 8, 8)})


# --- curriculum resolvability (#13b) -----------------------------------------
# The `ramp_seg_consistency` rule used to target `seg_consistency`, a strategy-local
# alias the loss-weight SSOT cannot resolve — the same defect that killed
# field_cocycle_anyfield, latent here because both bloch arms were NOT-IN-RUN when it
# was found. `segmentation_dice` is the registered name the strategy also accepts.


def test_segmentation_dice_declared_as_inline_managed() -> None:
    from mriforge.infrastructure.training.strategies.bloch_synth_strategy import (
        BlochSynthesisStrategy,
    )
    from mriforge.infrastructure.training.strategies.loss_folding import (
        inline_managed_with,
    )

    extra = BlochSynthesisStrategy._INLINE_MANAGED_EXTRA
    assert "segmentation_dice" in extra
    # Declaring it on image_losses must not make the fold apply it a second time.
    assert "segmentation_dice" in inline_managed_with(*extra)


def test_seg_consistency_alias_does_not_canonicalise_to_the_registered_name() -> None:
    """Why the rule had to be retargeted rather than the alias declared."""
    from mriforge.models.losses.weights import canonical_loss_name

    assert canonical_loss_name("seg_consistency") != canonical_loss_name(
        "segmentation_dice"
    )
