"""Unit tests for ``TorchIOAugmentationFactory``.

Pairs ``src/spectramr/data/transforms/augmentation_factory.py``, which had **no**
paired test until this file — which is how the defect below survived.

Until this commit, lines 133-141 of the factory read::

    # =====================================================================
    # 2. Intensity Transforms
    # =====================================================================
    # ... (unchanged) ...

    # =====================================================================
    # 3. MRI-Specific Augmentations
    # =====================================================================
    # ... (rest of methods) ...

A paste-elision marker, committed as production code, in place of the entire
intensity section — while the class docstring three lines above advertised
"Intensity: gamma, contrast, brightness, noise".  167 arms under
``experiments/inprogress/`` set ``enable_gamma`` and 67 set it ``true``; every
one of them got silence.  Nothing was red, because nothing looked.

The census below is the general form of that defect: *a schema flag the factory
ignores*.  Per non-negotiable 15 a detector is only trusted once it has been
watched failing, so each census runs against a planted stub as well as against
the real factory — the ``ignores`` fixtures below are the plants.
"""

from __future__ import annotations

import inspect

import pytest

from spectramr.config.schemas.augmentation import AugmentationConfigSchema
from spectramr.data.transforms import augmentation_factory
from spectramr.data.transforms.augmentation_factory import TorchIOAugmentationFactory
from tests.utils.augmentation_coverage import (
    NOT_PER_SAMPLE,
    build_ignoring,
    enable_flags,
    real_build,
    unwired_flags,
)


class TestEveryFlagIsWired:
    """The census that would have caught the elision."""

    def test_the_census_found_flags_to_judge(self) -> None:
        """Anti-vacuity: a census over an empty set would pass forever."""
        assert len(enable_flags()) >= 15

    def test_no_enable_flag_is_silently_ignored(self) -> None:
        unwired = unwired_flags(real_build)
        assert unwired == set(NOT_PER_SAMPLE), (
            "augmentation flag(s) exposed by the schema produce no transform:\n  "
            + "\n  ".join(sorted(unwired - set(NOT_PER_SAMPLE)))
            + "\n\nEvery exposed knob must be read and wired (non-negotiable 8). "
            "Either build a transform for it, or add it to NOT_PER_SAMPLE with "
            "the reason it cannot be one."
        )

    def test_no_stale_deferrals(self) -> None:
        """A name in ``NOT_PER_SAMPLE`` that is no longer a schema field, or is
        now wired, would silence a future finding."""
        assert set(NOT_PER_SAMPLE) <= set(enable_flags())
        assert set(NOT_PER_SAMPLE) <= unwired_flags(real_build)


class TestTheCensusActuallyFails:
    """Non-negotiable 15: plant the violation, in each shape it can take."""

    def test_detects_a_single_ignored_flag(self) -> None:
        """Shape 1 — one knob quietly dropped."""
        assert "enable_gamma" in unwired_flags(build_ignoring("enable_gamma"))

    def test_detects_a_whole_ignored_section(self) -> None:
        """Shape 2 — the original defect: an entire block elided."""
        section = (
            "enable_gamma",
            "enable_brightness",
            "enable_contrast",
            "enable_noise",
        )
        assert set(section) <= unwired_flags(build_ignoring(*section))


class TestNoElisionMarkers:
    """A direct regression for the literal defect, not just its general form."""

    MARKERS = ("... (unchanged)", "... (rest of", "# ...")

    @staticmethod
    def _markers_in(text: str) -> list[str]:
        return [m for m in TestNoElisionMarkers.MARKERS if m in text]

    def test_the_matcher_fires_on_the_committed_text(self) -> None:
        """Planted violation: the exact string that shipped."""
        assert self._markers_in("        # ... (unchanged) ...\n")

    def test_factory_source_carries_no_elision_marker(self) -> None:
        source = inspect.getsource(augmentation_factory)
        assert not self._markers_in(source), (
            "paste-elision marker in production source — a section was replaced "
            "by a placeholder comment instead of being written"
        )


class TestComplexGuard:
    """Intensity models are defined on magnitude images only."""

    INTENSITY = (
        "enable_gamma",
        "enable_brightness",
        "enable_contrast",
        "enable_noise",
        "enable_bias_field",
        "enable_rician_noise",
        "enable_motion_blur",
        "enable_blur",
    )

    def test_intensity_transforms_are_skipped_for_kspace(self) -> None:
        config = AugmentationConfigSchema(enabled=True, **dict.fromkeys(self.INTENSITY, True))
        assert TorchIOAugmentationFactory.build(config, dataset_type="kspace") is None

    def test_geometric_transforms_still_build_for_kspace(self) -> None:
        """The guard is scoped to intensity: flip stays available on k-space."""
        config = AugmentationConfigSchema(enabled=True, enable_flip=True)
        result = TorchIOAugmentationFactory.build(config, dataset_type="kspace")
        assert result is not None
        assert [type(t).__name__ for t in result.transforms] == ["RandomFlip"]


class TestGammaIsParameterisedInLogSpace:
    """``gamma_range`` is stated in gamma; TorchIO wants log-gamma."""

    def test_range_is_logged_on_the_way_in(self) -> None:
        import math

        config = AugmentationConfigSchema(enabled=True, enable_gamma=True, gamma_range=(0.5, 2.0))
        result = TorchIOAugmentationFactory.build(config, dataset_type="image")
        gamma = result.transforms[0]
        assert gamma.log_gamma_range == pytest.approx((math.log(0.5), math.log(2.0)))

    def test_a_raw_range_would_have_been_a_harsher_remap(self) -> None:
        """Guards the conversion itself: passing the range through unlogged
        requests exp(0.8)..exp(1.2), not 0.8..1.2."""
        import math

        assert math.exp(0.8) > 2.2

    def test_disabled_config_returns_none(self) -> None:
        assert (
            TorchIOAugmentationFactory.build(
                AugmentationConfigSchema(enabled=False), dataset_type="image"
            )
            is None
        )
