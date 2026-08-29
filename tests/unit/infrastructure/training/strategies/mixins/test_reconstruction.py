"""Field-conditioning injection on the LIVE reconstruction forward (regressions).

A field-conditioned generator (``AnatomyFieldRenderer`` / ``CrossFieldRenderer`` &
friends) takes ``field_strength`` as a *keyword-only, required* argument. When such
a model is routed through the GENERIC reconstruction validation path — e.g. the
calibration strategy (``ConformalCalibrationStrategy``), which inherits
``_validation_forward`` from ``ReconstructionTrainingStrategy`` and has no
field-aware override — the forward kwargs MUST carry ``field_strength`` or the
generator raises ``missing 1 required keyword-only argument: 'field_strength'`` and
the run produces zero successful validation batches (CLAUDE.md #10; the mrixfields
``b17_dice_risk_calibration`` crash, re-reproduced 2026-06-24).

History (2026-06-25): the original 2026-06-22 fix injected the kwarg into
``ReconstructionMixin._prepare_generator_inputs_reconstruction`` — but that method
has **no callers**; the live path is
``ReconstructionTrainingStrategy._prepare_generator_inputs`` (used by
``_validation_forward`` and ``_compute_losses``). So the green mixin test guarded
dead code while b17 kept crashing. These tests therefore exercise the LIVE method
and the real ``_validation_forward`` seam.

The injection is signature-gated (``_callable_accepts_kwarg``) so plain models are
untouched, prefers the TARGET field over the source, and is never defaulted (a
default field would be a silent fallback, pitfall #9).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from mriforge.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)


class _FieldGen(nn.Module):
    """Cross-field renderer stub: field_strength is keyword-only + required."""

    def forward(self, x, *, field_strength, contrast_id=None):  # noqa: D401
        # Echo the field so a test can assert it actually flowed through.
        self.last_field_strength = field_strength
        self.last_contrast_id = contrast_id
        return x


class _PlainGen(nn.Module):
    def forward(self, x):  # noqa: D401
        return x


def _host(generator: nn.Module) -> ReconstructionTrainingStrategy:
    """Minimal LIVE strategy host (no full DI build) for the forward-prep seam."""
    host = ReconstructionTrainingStrategy.__new__(ReconstructionTrainingStrategy)
    host.env = SimpleNamespace(
        generator=generator,
        config=SimpleNamespace(
            model=SimpleNamespace(model_type="anatomy_field_renderer", input_type="image"),
            # `multislice_enabled` is a declared DataConfigSchema field with a
            # default, so a real config always carries it. The stub omitted it,
            # which only passed while the reader used a hasattr fallback.
            data=SimpleNamespace(multislice_enabled=False),
            training=SimpleNamespace(),
        ),
    )
    host.state = SimpleNamespace(config=host.env.config)
    host.config = host.env.config
    host.logging_service = None
    host.dc_layer = None
    return host


def _ctx(**extra):
    base = {"use_dc": False, "multimodal_inputs": None, "measured_kspace": None}
    base.update(extra)
    return base


def test_field_strength_injected_when_generator_requires_it():
    host = _host(_FieldGen())
    x = torch.randn(1, 1, 8, 8)
    ctx = _ctx(field_strength=torch.tensor([0.1]), field_strength_target=torch.tensor([3.0]))
    _lr, fwd_kwargs = host._prepare_generator_inputs(ctx, x, validation=True)
    assert "field_strength" in fwd_kwargs
    # prefers the TARGET field (render at HF), not the 0.1 T source
    assert torch.allclose(fwd_kwargs["field_strength"], torch.tensor([3.0]))


def test_field_strength_falls_back_to_source_when_no_target():
    host = _host(_FieldGen())
    x = torch.randn(1, 1, 8, 8)
    ctx = _ctx(field_strength=torch.tensor([0.064]))
    _lr, fwd_kwargs = host._prepare_generator_inputs(ctx, x, validation=True)
    assert torch.allclose(fwd_kwargs["field_strength"], torch.tensor([0.064]))


def test_field_strength_not_injected_for_plain_generator():
    host = _host(_PlainGen())
    x = torch.randn(1, 1, 8, 8)
    ctx = _ctx(field_strength_target=torch.tensor([3.0]))
    _lr, fwd_kwargs = host._prepare_generator_inputs(ctx, x, validation=True)
    assert "field_strength" not in fwd_kwargs


def test_contrast_id_injected_when_generator_accepts_it():
    host = _host(_FieldGen())
    x = torch.randn(1, 1, 8, 8)
    ctx = _ctx(field_strength_target=torch.tensor([3.0]), contrast_id=torch.tensor([1]))
    _lr, fwd_kwargs = host._prepare_generator_inputs(ctx, x, validation=True)
    assert "contrast_id" in fwd_kwargs
    assert torch.equal(fwd_kwargs["contrast_id"], torch.tensor([1]))


def test_validation_forward_does_not_raise_for_field_renderer():
    """The real crash seam: ``_validation_forward`` -> generator(**kwargs).

    Before the fix this raised ``missing 1 required keyword-only argument:
    'field_strength'`` (b17). Now it must complete and the field must reach the
    generator's forward.
    """
    gen = _FieldGen()
    host = _host(gen)
    x = torch.randn(1, 1, 8, 8)
    ctx = _ctx(field_strength_target=torch.tensor([3.0]))
    out = host._validation_forward(x, ctx)
    assert out.shape == x.shape
    assert torch.allclose(gen.last_field_strength, torch.tensor([3.0]))


def test_mixin_multislice_flag_is_read_directly() -> None:
    """Same dead-guard as the strategy: the mixin carries a byte-identical copy
    of the block, so both had to be fixed together."""
    import inspect

    from mriforge.config.schemas.data import DataConfigSchema
    from mriforge.infrastructure.training.strategies.mixins import reconstruction

    assert "multislice_enabled" in DataConfigSchema.model_fields
    src = inspect.getsource(reconstruction)
    assert 'hasattr(config.data, "multislice_enabled")' not in src
    assert "multislice_enabled = config.data.multislice_enabled" in src
