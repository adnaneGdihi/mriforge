"""Unit tests for cascading acceleration validation in DiffusionTrainingStrategy.

Tests cover:
- The linear-FALLBACK inverse and its ``[T/10, 9T/10]`` band clamp
- The cascading metrics key format
- The tall cascade record
"""

import unittest

from spectramr.core.cascading_validation import (
    CASCADE_ID_COLUMNS,
    CASCADING_LEVELS,
    build_cascade_row,
    legacy_linear_timestep,
    training_band,
)

# ---------------------------------------------------------------------------
# Pure-function tests (no torch/model needed)
# ---------------------------------------------------------------------------

# Probe inputs for the inversion formula below -- deliberately WIDER than the
# production sweep and NOT a copy of it. It was named `CASCADING_LEVELS` and
# read as production's list while disagreeing with it ([2,4,8,12,16,32] vs
# [2,8,32]); the real set is imported from the SSOT as `PRODUCTION_LEVELS`.
_FORMULA_PROBE_LEVELS = [2, 4, 8, 12, 16, 32]

#: The one true sweep. Imported, never restated (#697).
PRODUCTION_LEVELS = CASCADING_LEVELS

#: The fallback formula's own span, ``max_acceleration - base_acceleration``.
#: These tests exercise the historical R∈[1,8] arm the formula was written for.
_BASE = 1.0
_MAX = 8.0


def _timestep_for_level(accel: int, num_timesteps: int) -> int:
    """Call the REAL fallback inverse -- do not re-derive it here.

    What stood in this slot re-implemented ``t = T*(R-1)/7`` and its clamp, and
    then asserted properties of its own copy: it could not fail when production
    changed, which is the same self-mirroring shape `TestTallCascadeRecord`
    below already condemns. #1295 extracted the formula to
    `core.cascading_validation.legacy_linear_timestep` precisely so this file
    can call it.
    """
    return legacy_linear_timestep(
        acceleration=accel,
        base_acceleration=_BASE,
        max_acceleration=_MAX,
        num_timesteps=num_timesteps,
    )


class TestLinearFallbackInversionFormula(unittest.TestCase):
    """The FALLBACK inverse, reached only when no mask generator is wired.

    The band clamp lives here and nowhere else now. On the schedule-aware path
    it was removed in #1295: `timestep_for_acceleration` already returns a
    timestep in ``[0, T-1]``, and clamping that answer into ``[T/10, 9T/10]``
    moved the mask onto a neighbouring rung while the metric column kept naming
    the requested one.
    """

    def setUp(self):
        self.T = 1000

    def test_all_levels_within_bounds(self):
        """All probe levels must produce t ∈ [T/10, 9T/10]."""
        min_t, max_t = training_band(self.T)
        for accel in _FORMULA_PROBE_LEVELS:
            t = _timestep_for_level(accel, self.T)
            self.assertGreaterEqual(t, min_t, f"R={accel}x: t={t} below min_t={min_t}")
            self.assertLessEqual(t, max_t, f"R={accel}x: t={t} above max_t={max_t}")

    def test_monotone_nondecreasing(self):
        """Higher acceleration should produce equal or higher timestep."""
        prev_t = 0
        for accel in _FORMULA_PROBE_LEVELS:
            t = _timestep_for_level(accel, self.T)
            self.assertGreaterEqual(
                t, prev_t, f"Monotone violated: R={accel}x gives t={t} < prev {prev_t}"
            )
            prev_t = t

    def test_2x_acceleration_formula(self):
        """2x → T*(2-1)/7 ≈ 142 for T=1000, well within the band."""
        self.assertEqual(_timestep_for_level(2, self.T), int(self.T * 1 / 7.0))

    def test_4x_acceleration_formula(self):
        """4x → T*3/7 ≈ 428."""
        self.assertEqual(_timestep_for_level(4, self.T), int(self.T * 3 / 7.0))

    def test_high_levels_clamp_to_max(self):
        """The formula is unbounded above ``max_acceleration``; R=32 would give
        t=4428 for T=1000. The clamp is what keeps the fallback legal, and it
        is the reason the band survives on this path."""
        _, max_t = training_band(self.T)
        for accel in (8, 12, 16, 32):
            self.assertEqual(_timestep_for_level(accel, self.T), max_t)

    def test_different_T_values(self):
        """Formula must stay in-bounds for T=100, 500, 2000."""
        for T in (100, 500, 2000):
            min_t, max_t = training_band(T)
            for accel in _FORMULA_PROBE_LEVELS:
                t = _timestep_for_level(accel, T)
                self.assertGreaterEqual(t, min_t, f"T={T}, R={accel}x: t={t} < min_t")
                self.assertLessEqual(t, max_t, f"T={T}, R={accel}x: t={t} > max_t")

    def test_the_band_is_imported_not_restated(self):
        """The regression this class carried: a hand-copy of production's
        arithmetic that agreed with it only until one of them moved."""
        self.assertEqual(training_band(1000), (100, 900))
        self.assertEqual(training_band(29), (2, 27))
        # T<10 must not collapse to zero -- `max(1, ...)` is load-bearing.
        self.assertEqual(training_band(5)[0], 1)


class TestCascadingMetricsKeyFormat(unittest.TestCase):
    """Verify that metric keys are correctly suffixed per acceleration level."""

    def _apply_level_prefix(self, level_metrics: dict, accel: int) -> dict:
        """Replicate the key transformation from validation_step()."""
        suffix = f"_{accel}x"
        return {f"{k}{suffix}": v for k, v in level_metrics.items()}

    def test_single_level_keys(self):
        base_metrics = {"val_psnr": 40.0, "val_ssim": 0.92, "val_loss": 0.01}
        result = self._apply_level_prefix(base_metrics, 4)
        self.assertIn("val_psnr_4x", result)
        self.assertIn("val_ssim_4x", result)
        self.assertIn("val_loss_4x", result)
        self.assertAlmostEqual(result["val_psnr_4x"], 40.0)

    def test_all_levels_yield_distinct_keys(self):
        base_metrics = {"val_psnr": 40.0}
        combined: dict = {}
        for accel in _FORMULA_PROBE_LEVELS:
            combined.update(self._apply_level_prefix(base_metrics, accel))

        expected_keys = {f"val_psnr_{a}x" for a in _FORMULA_PROBE_LEVELS}
        self.assertEqual(set(combined.keys()), expected_keys)

    def test_no_untagged_base_keys_in_combined(self):
        """The combined dict must NOT contain bare 'val_psnr' (only suffixed versions)."""
        base_metrics = {"val_psnr": 40.0, "val_ssim": 0.92}
        combined: dict = {}
        for accel in _FORMULA_PROBE_LEVELS:
            combined.update(self._apply_level_prefix(base_metrics, accel))
        self.assertNotIn("val_psnr", combined)
        self.assertNotIn("val_ssim", combined)


class TestTallCascadeRecord(unittest.TestCase):
    """The sweep is TALL now: level and timestep are values, not column names.

    What stood here was a self-mirroring test. `_build_cascading_columns`
    announced itself as mirroring train.py's CSV initialization, and then
    reimplemented it -- against a DIFFERENT level list ([2,4,8,12,16,32]) than
    production's [2,8,32], asserting `6 * 15 == 90` columns while production
    built 45 and the row writer's `extrasaction="ignore"` discarded every one.

    It could not fail: it tested its own copy. That is issue #686's
    hand-copied-vocabulary shape, and it is why the dead code survived.
    These tests call the real `build_cascade_row` instead.
    """

    def test_the_severity_point_is_data_not_a_column_name(self):
        row = build_cascade_row(
            acceleration_level=8,
            heldout=False,
            timestep=200,
            metrics={"val_psnr": 31.0},
        )
        self.assertEqual(row["acceleration_level"], 8.0)
        self.assertEqual(row["timestep"], 200.0)
        self.assertIn("val_psnr", row)
        # The retired shape. A suffixed key here means someone reintroduced it.
        self.assertNotIn("val_psnr_8x", row)

    def test_metric_names_are_not_drawn_from_any_list(self):
        """Whatever the arm computed gets a column -- the point of the change.

        The retired 15-name list did not contain `val_hfen`, so a drained
        `kspace_filling` arm emitting `val_hfen_8x` had nowhere to put it.
        """
        row = build_cascade_row(
            acceleration_level=2,
            heldout=False,
            timestep=100,
            metrics={"val_hfen": 0.4, "val_kspace_error": 0.1, "val_phase_mse": 0.02},
        )
        for name in ("val_hfen", "val_kspace_error", "val_phase_mse"):
            self.assertIn(name, row)

    def test_heldout_is_a_column_not_a_third_naming_convention(self):
        """`_heldout_{R}x` appeared in no name list at all."""
        held = build_cascade_row(
            acceleration_level=64,
            heldout=True,
            timestep=900,
            metrics={"val_psnr": 20.0},
        )
        self.assertIs(held["heldout"], True)
        self.assertEqual(held["acceleration_level"], 64.0)
        self.assertNotIn("val_psnr_heldout_64x", held)

    def test_identity_columns_win_over_a_colliding_metric_name(self):
        """A metric literally called `timestep` must not overwrite the severity
        point the row is about."""
        row = build_cascade_row(
            acceleration_level=8,
            heldout=False,
            timestep=200,
            metrics={"timestep": -1.0, "acceleration_level": -1.0},
        )
        self.assertEqual(row["timestep"], 200.0)
        self.assertEqual(row["acceleration_level"], 8.0)

    def test_a_missing_timestep_is_none_not_zero(self):
        """0 is a legal timestep; the absence of one must not read as t=0."""
        row = build_cascade_row(
            acceleration_level=2, heldout=False, timestep=None, metrics={}
        )
        self.assertIsNone(row["timestep"])

    def test_id_columns_cover_what_the_row_declares(self):
        row = build_cascade_row(
            acceleration_level=2, heldout=False, timestep=100, metrics={}
        )
        # iteration/epoch are stamped by the writer, which knows them.
        self.assertEqual(set(row), set(CASCADE_ID_COLUMNS) - {"iteration", "epoch"})

    def test_production_levels_are_the_ssot_not_a_local_copy(self):
        """The regression this file itself carried: two disagreeing lists."""
        self.assertEqual(tuple(PRODUCTION_LEVELS), (2, 8, 32))
        self.assertIsNot(PRODUCTION_LEVELS, _FORMULA_PROBE_LEVELS)


if __name__ == "__main__":
    unittest.main()
