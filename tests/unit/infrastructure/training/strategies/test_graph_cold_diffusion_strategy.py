"""Degradation-pattern resolution for :class:`GraphColdDiffusionStrategy` (#1092).

The strategy used to resolve its k-space mask through code that could not run:
``_setup_accelerator`` imported ``ACCELERATOR_REGISTRY`` (the name is
``_ACCELERATOR_REGISTRY``), so the import raised ``ImportError`` on every run, was
caught, and logged at ``debug``. ``self.accelerator`` was therefore always ``None``, the
NUFFT branch it gated was unreachable, and every run fell through to a mask **hardcoded**
to ``random_cartesian`` — ignoring ``physics.compressed_sensing.sampling_pattern``
entirely, on eleven arms including three literature baselines.

These tests pin the three properties that replacement has to hold:

1. the eight existing arms keep training on exactly the pattern they trained on before
   (a wiring fix must not silently change anyone's science);
2. an unresolvable pattern RAISES instead of falling back (CLAUDE.md #9); and
3. the non-Cartesian patterns stay refused for as long as their schedule is broken —
   which is a property of the mask stack, so it is asserted against the mask stack
   rather than trusted to a comment.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from spectramr.infrastructure.training.strategies.graph_cold_diffusion_strategy import (
    GraphColdDiffusionStrategy as S,
)


def _resolve(pattern: str) -> str:
    """Call the resolver with a minimal stand-in for ``self``.

    Deliberately not a real strategy instance: constructing one needs a
    ``TrainingEnvironment`` (model, optimizers, dataloaders), none of which the resolver
    touches. Binding the plain function to a namespace keeps the test on the logic under
    test instead of on a fixture that could pass for the wrong reason.
    """
    fake_self = SimpleNamespace(
        # the class attributes the resolver reads off `self`
        _PATTERN_ALIASES=S._PATTERN_ALIASES,
        _NON_CARTESIAN_PATTERNS=S._NON_CARTESIAN_PATTERNS,
        config=SimpleNamespace(
            physics=SimpleNamespace(
                compressed_sensing=SimpleNamespace(sampling_pattern=pattern)
            )
        )
    )
    return S._resolve_degradation_pattern(fake_self)


class TestPatternResolution:
    def test_schema_default_cartesian_resolves_and_does_not_raise(self) -> None:
        """``cartesian`` is the schema default and what all eight arms declare — but it
        is NOT a key in the mask registry. It must be mapped, not rejected."""
        assert _resolve("cartesian") == "random_cartesian"

    def test_the_alias_preserves_what_the_hardcoded_fallback_used(self) -> None:
        """The behaviour-preservation contract. The old code hardcoded
        ``pattern="random_cartesian"``; the alias must land on the same string, or this
        'wiring fix' silently changes what eight arms train on."""
        assert S._PATTERN_ALIASES["cartesian"] == "random_cartesian"

    def test_a_registered_cartesian_pattern_passes_through_unchanged(self) -> None:
        assert _resolve("uniform_cartesian") == "uniform_cartesian"

    def test_an_unknown_pattern_raises_and_names_the_vocabulary(self) -> None:
        """`sampling_pattern` is typed `str`, not a Literal, so a typo cannot be caught
        at schema-validation time. It has to be caught here or not at all."""
        with pytest.raises(ValueError, match="not a registered k-space pattern"):
            _resolve("cartesain")  # transposed letters

    def test_the_unknown_pattern_error_lists_what_is_available(self) -> None:
        with pytest.raises(ValueError) as exc:
            _resolve("definitely_not_a_pattern")
        assert "random_cartesian" in str(exc.value)

    @pytest.mark.parametrize("pattern", ["radial", "spiral", "golden_angle"])
    def test_non_cartesian_patterns_are_refused_with_the_reason(self, pattern) -> None:
        """Refused rather than silently accepted: the mask stack DOES produce these, so
        without an explicit guard they would look like they work."""
        with pytest.raises(ValueError, match="non-Cartesian family"):
            _resolve(pattern)


class TestTheScheduleThatJustifiesTheRefusal:
    """The refusal above is only correct while the schedule is actually broken.

    Asserted against the live mask generator rather than taken on faith, so that if
    someone fixes the non-Cartesian schedule this test fails and points at the guard
    that should then be removed — instead of the guard quietly outliving its reason.
    """

    @staticmethod
    def _fractions(pattern: str, timesteps=(0, 100, 200, 400, 999)):
        from spectramr.infrastructure.training.utils.kspace_masks import KSpaceMaskGenerator

        g = KSpaceMaskGenerator(num_timesteps=1000)
        return [
            g.generate_acceleration_mask(
                timestep=t, image_shape=(64, 64), pattern=pattern
            )
            .float()
            .mean()
            .item()
            for t in timesteps
        ]

    @pytest.mark.parametrize("pattern", ["random_cartesian", "uniform_cartesian"])
    def test_cartesian_patterns_start_fully_sampled(self, pattern) -> None:
        """t=0 is the CLEAN end of a cold-diffusion schedule."""
        assert self._fractions(pattern)[0] == pytest.approx(1.0)

    @pytest.mark.parametrize("pattern", ["random_cartesian", "uniform_cartesian"])
    def test_cartesian_degradation_never_decreases_with_t(self, pattern) -> None:
        f = self._fractions(pattern)
        assert all(f[i] >= f[i + 1] - 1e-6 for i in range(len(f) - 1)), f

    @pytest.mark.parametrize("pattern", ["radial", "spiral", "golden_angle"])
    def test_non_cartesian_has_no_clean_end(self, pattern) -> None:
        """The reason for the guard: at t=0 these are already heavily undersampled, so
        the model never sees x_0."""
        assert self._fractions(pattern)[0] < 0.5

    @pytest.mark.parametrize("pattern", ["radial", "spiral", "golden_angle"])
    def test_non_cartesian_schedule_saturates(self, pattern) -> None:
        """The second reason: the schedule goes FLAT, so the timestep conditioning
        carries almost no information over most of its range.

        Measured from t=400 rather than t=200: radial and spiral are already flat at
        200, but golden_angle still moves (0.086 -> 0.057) between 200 and 400. Pinning
        the looser, true bound rather than the tidier, false one — the claim being
        tested is 'saturates', not 'saturates at exactly 200'."""
        f = self._fractions(pattern, timesteps=(400, 600, 999))
        assert max(f) - min(f) < 0.01, f
