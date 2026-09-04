"""`PhysicsBuilder` reads the schedule length from the canonical path.

Paired with ``src/spectramr/infrastructure/training/builders/physics_builder.py``.

``build_mask_generator`` used to have a second branch reading
``training.num_timesteps`` — a path no schema has ever carried, so it could not
fire even when the canonical block was absent. These tests pin the surviving
read against a REAL ``TrainingSettings`` rather than a namespace stub: a stub
that carries the knob flat is a shape no real config produces, which is exactly
how the equivalent dead read in ``forward_probe`` stayed green.
"""

from __future__ import annotations

import pytest

from spectramr.config.schemas.training.base import UnspecifiedParams
from spectramr.infrastructure.training.builders.physics_builder import PhysicsBuilder
from tests.utils.minimal_settings import minimal_settings_for


def _settings_with_timesteps(n: int | None):
    """A real ``TrainingSettings``, optionally carrying a diffusion block.

    ``gan`` is used because it is the only ``minimal_settings_for`` key that
    still loads — the other 47 resolve to fixtures the loader refuses on
    ``config_version: 6.0`` (see the tracking issue). It carries no diffusion
    block of its own, which makes it a clean base for both cases here.
    """
    settings = minimal_settings_for("gan")
    if n is None:
        return settings
    training = settings.training.model_copy(update={"diffusion": UnspecifiedParams(timesteps=n)})
    return settings.model_copy(update={"training": training})


def _mask_generator(settings):
    builder = PhysicsBuilder(settings, "cpu").build_mask_generator()
    return builder._components.get("mask_generator")


def test_canonical_timesteps_reaches_the_mask_generator() -> None:
    generator = _mask_generator(_settings_with_timesteps(42))
    assert generator is not None, "mask generator was not built at all"
    assert generator.num_timesteps == 42


def test_absent_diffusion_block_falls_back_to_the_documented_default() -> None:
    """Anti-vacuity: if the builder always produced 1000 the test above would
    still pass on a broken read, so pin the no-block case separately."""
    generator = _mask_generator(_settings_with_timesteps(None))
    assert generator is not None, "mask generator was not built at all"
    assert generator.num_timesteps == 1000


@pytest.mark.parametrize("declared", [8, 250, 1000])
def test_the_value_is_carried_through_rather_than_defaulted(declared: int) -> None:
    """A read that silently defaulted would pass only for `declared == 1000`."""
    assert _mask_generator(_settings_with_timesteps(declared)).num_timesteps == declared


def _settings_with_undersampling():
    """A real ``TrainingSettings`` carrying the exp_11 acceleration block."""
    from spectramr.config.schemas.acceleration import AccelerationConfigSchema

    settings = _settings_with_timesteps(28)
    return settings.model_copy(
        update={
            "undersampling": AccelerationConfigSchema(
                acceleration_type="density_nested",
                base_acceleration=2.0,
                max_acceleration=32.0,
                center_fraction=0.08,
                min_center_fraction=0.02,
                mask_direction="phase",
                schedule_type="step",
                mask_seed=42,
                enforce_nested=True,
            )
        }
    )


def test_declared_acceleration_reaches_the_accelerator_intact() -> None:
    """``build_mask_generator`` used to splat the whole ``model_dump()``.

    Every schema field rode into the accelerator constructor — defaults for
    knobs it does not read included — while ``mask_seed`` was never translated
    to ``seed``. Constructing the accelerator is the assertion: the kwarg gate
    raises on any name outside the registered vocabulary.
    """
    generator = _mask_generator(_settings_with_undersampling())
    assert generator is not None
    accelerator = generator._get_accelerator(None)
    assert accelerator is not None
    assert accelerator.seed == 42, "mask_seed must arrive as the accelerator's seed"


def test_unread_schema_defaults_are_not_forwarded() -> None:
    """Names the accelerator does not read must not be in the kwargs at all."""
    generator = _mask_generator(_settings_with_undersampling())
    kwargs = generator._accelerator_kwargs
    for junk in ("mixed_precision", "use_compile", "acceleration_type", "mask_seed"):
        assert junk not in kwargs, f"{junk} would reach the accelerator"
    assert kwargs["max_acceleration"] == 32.0
    assert kwargs["min_center_fraction"] == 0.02


def test_build_coil_sensitivity_is_an_honest_no_op():
    """The step never worked, and could not have been noticed from outside.

    It imported ``ESPIRiTSensitivity`` from ``physics/coil_sensitivity.py``, which
    exports FUNCTIONS and has never defined that class. The import never actually
    raised, though: the two guards above it read
    ``config.physics.parallel_imaging.enabled``, and ``parallel_imaging`` is not a
    field on ANY config schema while ``settings.physics`` defaults to ``None`` — so
    both returned early on every call and the body was unreachable. A dead knob
    kept a dead import invisible.

    Restoring the import would fix nothing: ``_components["coil_sens"]`` was the
    only reference to that key tree-wide. ``estimate_smaps`` (called live from
    ``data_pipeline_director``) is the elected owner (non-negotiable 17).
    """
    import inspect

    from spectramr.infrastructure.training.builders.physics_builder import PhysicsBuilder

    source = inspect.getsource(PhysicsBuilder.build_coil_sensitivity)
    body = source.split('"""')[2]
    assert "ESPIRiTSensitivity" not in body, (
        "the non-existent class is referenced outside the explanatory docstring"
    )
    assert "coil_sens" not in body, "a component nothing reads is being populated again"


def test_build_coil_sensitivity_still_chains():
    """It stays in the fluent chain (``director.py`` calls it) — a step that
    silently vanishes is harder to notice than one that says why it does nothing."""
    import inspect

    from spectramr.infrastructure.training.builders.physics_builder import PhysicsBuilder

    sig = inspect.signature(PhysicsBuilder.build_coil_sensitivity)
    assert "PhysicsBuilder" in str(sig.return_annotation)
