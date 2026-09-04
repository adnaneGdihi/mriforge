"""Phase 5 tests — sampler-variant factory + schema validation.

Coverage:

1. Schema validation: ``label`` / ``weighted`` types require their
   companion fields; missing field → ValidationError at config-load.
2. Factory dispatch: each sampler type produces the correct TorchIO
   class with the user-declared parameters threaded through.
3. End-to-end via TorchIOQueueConfig: when ``data.modes.train.sampler``
   declares a non-uniform type, the queue config carries the right
   ``sampler_*`` fields and the helper namespace adapter exposes them
   for the factory.
"""
from __future__ import annotations

import pytest

from spectramr.config.schemas.data import (
    DataConfigSchema,
    DataModesSchema,
    ModeConfigSchema,
    ModeSamplerSchema,
)
from spectramr.data.builders.sampler_factory import (
    PATCH_SAMPLER_TYPES,
    build_patch_sampler,
)
from spectramr.data.builders.torchio_queue_builder import (
    TorchIOQueueConfig,
    config_to_sampler_view,
)


# ── Schema validation ──────────────────────────────────────────────────────


def test_label_sampler_requires_label_name() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="label_name"):
        ModeSamplerSchema(type="label")


def test_weighted_sampler_requires_probability_map() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="probability_map"):
        ModeSamplerSchema(type="weighted")


def test_uniform_sampler_needs_no_extra_fields() -> None:
    """Default ``uniform`` validates without label_name or probability_map."""
    s = ModeSamplerSchema(type="uniform", samples_per_volume=4)
    assert s.type == "uniform"
    assert s.label_name is None
    assert s.probability_map is None


def test_label_sampler_with_label_name_validates() -> None:
    s = ModeSamplerSchema(
        type="label",
        label_name="mask",
        label_probabilities={0: 0.2, 1: 0.8},
    )
    assert s.label_name == "mask"
    assert s.label_probabilities == {0: 0.2, 1: 0.8}


def test_weighted_sampler_with_probability_map_validates() -> None:
    s = ModeSamplerSchema(type="weighted", probability_map="brain_mask")
    assert s.probability_map == "brain_mask"


# ── Factory dispatch ────────────────────────────────────────────────────────


def test_factory_builds_uniform_sampler() -> None:
    import torchio as tio

    from types import SimpleNamespace

    ns = SimpleNamespace(
        type="uniform",
        label_name=None,
        label_probabilities=None,
        probability_map=None,
    )
    s = build_patch_sampler(ns, (32, 32, 1))
    assert isinstance(s, tio.data.UniformSampler)


def test_factory_builds_label_sampler() -> None:
    import torchio as tio

    from types import SimpleNamespace

    ns = SimpleNamespace(
        type="label",
        label_name="mask",
        label_probabilities={0: 0.2, 1: 0.8},
        probability_map=None,
    )
    s = build_patch_sampler(ns, (32, 32, 1))
    assert isinstance(s, tio.data.LabelSampler)


def test_factory_builds_weighted_sampler() -> None:
    import torchio as tio

    from types import SimpleNamespace

    ns = SimpleNamespace(
        type="weighted",
        label_name=None,
        label_probabilities=None,
        probability_map="brain_mask",
    )
    s = build_patch_sampler(ns, (32, 32, 1))
    assert isinstance(s, tio.data.WeightedSampler)


def test_factory_rejects_non_patch_types() -> None:
    """``full``, ``grid``, and ``temporal_*`` are handled elsewhere —
    the patch-sampler factory must reject them with a clear message."""
    from types import SimpleNamespace

    for t in ("full", "grid", "temporal_uniform", "temporal_grid"):
        ns = SimpleNamespace(
            type=t,
            label_name=None,
            label_probabilities=None,
            probability_map=None,
        )
        with pytest.raises(ValueError, match="not a patch sampler"):
            build_patch_sampler(ns, (32, 32, 1))


def test_factory_defensive_check_for_missing_label_name() -> None:
    """Pydantic should catch this earlier; the factory's defensive
    check is here in case a caller hand-constructs a config."""
    from types import SimpleNamespace

    ns = SimpleNamespace(
        type="label",
        label_name=None,  # missing!
        label_probabilities=None,
        probability_map=None,
    )
    with pytest.raises(ValueError, match="label_name"):
        build_patch_sampler(ns, (32, 32, 1))


# ── End-to-end through TorchIOQueueConfig ──────────────────────────────────


def test_queue_config_threads_sampler_type_from_train_mode() -> None:
    """When ``data.modes.train.sampler.type='label'`` is configured, the
    queue config carries the right ``sampler_*`` fields for the factory."""
    c = DataConfigSchema(
        samples_per_volume=4,
        modes=DataModesSchema(
            train=ModeConfigSchema(
                sampler=ModeSamplerSchema(
                    type="label",
                    label_name="mask",
                    label_probabilities={0: 0.2, 1: 0.8},
                )
            )
        ),
    )
    qc = TorchIOQueueConfig.from_training_config(c)
    assert qc.sampler_type == "label"
    assert qc.sampler_label_name == "mask"
    assert qc.sampler_label_probabilities == {0: 0.2, 1: 0.8}


def test_queue_config_defaults_to_uniform_when_no_modes() -> None:
    """Legacy YAML (no ``data.modes`` block) → sampler_type defaults to
    'uniform', preserving pre-Phase-5 behavior."""
    c = DataConfigSchema(samples_per_volume=4)
    qc = TorchIOQueueConfig.from_training_config(c)
    assert qc.sampler_type == "uniform"
    assert qc.sampler_label_name is None


def test_config_to_sampler_view_round_trip() -> None:
    """The adapter exposes a namespace the factory can consume."""
    c = DataConfigSchema(
        samples_per_volume=4,
        modes=DataModesSchema(
            train=ModeConfigSchema(
                sampler=ModeSamplerSchema(
                    type="weighted", probability_map="brain_mask"
                )
            )
        ),
    )
    qc = TorchIOQueueConfig.from_training_config(c)
    view = config_to_sampler_view(qc)
    assert view.type == "weighted"
    assert view.probability_map == "brain_mask"

    # And the factory accepts the namespace.
    import torchio as tio

    s = build_patch_sampler(view, (16, 16, 1))
    assert isinstance(s, tio.data.WeightedSampler)


def test_patch_sampler_types_constant_lists_all_factory_kinds() -> None:
    """The ``PATCH_SAMPLER_TYPES`` constant must list exactly the kinds
    the factory handles — used by the queue builder to decide whether
    to dispatch."""
    assert PATCH_SAMPLER_TYPES == {"uniform", "label", "weighted"}
