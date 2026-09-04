"""Config schema for the field-dependent spectral Wiener strategy (B-2.7)."""

from __future__ import annotations

from spectramr.config.schemas.strictness import StrictSchema


class FieldWienerConfig(StrictSchema):
    """Knobs for the field-dependent spectral Wiener restorer (B-2.7).

    The field-dependent noise PSD is structural on the model (``model_kwargs.use_field_noise`` —
    the one-knob ablation); this block carries the loss weight only.
    """
