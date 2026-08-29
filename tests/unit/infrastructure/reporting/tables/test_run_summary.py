"""The tables-only report floor must actually have a floor.

``generate_report`` on a plain training run used to emit NOTHING. Its ``default``
preset holds two publication tables — ``tab_2_1_main_results`` (rows = methods)
and ``tab_2_4_dataset_descriptor`` (needs a cohort descriptor) — and a single run
has neither, so both correctly returned ``None``. Verified on a real run dir
carrying 27 rows of perfectly good metrics: zero files written.

That made "fall back to tables" an empty promise, and an empty promise is worse
than the legacy generator it would have replaced.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mriforge.infrastructure.reporting.tables import TABLES, run_summary


def _frame() -> pd.DataFrame:
    """Long-format, matching what ``aggregator.aggregate`` emits."""
    return pd.DataFrame(
        [
            {"split": "val", "metric": "val_psnr", "value": 20.0, "step": 10},
            {"split": "val", "metric": "val_psnr", "value": 25.0, "step": 20},
            {"split": "val", "metric": "val_psnr", "value": 22.0, "step": 30},
            {"split": "train", "metric": "loss", "value": 0.9, "step": 10},
            {"split": "train", "metric": "loss", "value": 0.3, "step": 20},
            {"split": "train", "metric": "loss", "value": 0.5, "step": 30},
        ]
    )


class TestRunSummaryTable:
    def test_it_is_registered(self):
        assert "tab_run_summary" in TABLES

    def test_emits_all_three_formats(self, tmp_path):
        out = run_summary.make(_frame(), tmp_path)
        assert out is not None
        assert set(out) == {"tex", "md", "csv"}
        for p in out.values():
            assert p.exists() and p.stat().st_size > 0

    def test_best_follows_the_direction_ssot_not_the_last_value(self, tmp_path):
        """The whole point. ``val_psnr`` peaks mid-run and ``loss`` bottoms
        mid-run; a naive "final value" summary would report neither."""
        run_summary.make(_frame(), tmp_path)
        df = pd.read_csv(tmp_path / "run_summary.csv").set_index("metric")

        psnr = df.loc["val_psnr"]
        assert psnr["final"] == 22.0 and psnr["best"] == 25.0
        assert psnr["best_step"] == 20 and psnr["direction"] == "higher"

        loss = df.loc["loss"]
        assert loss["final"] == 0.5 and loss["best"] == 0.3
        assert loss["best_step"] == 20 and loss["direction"] == "lower"

    def test_an_unknown_metric_gets_no_best_rather_than_a_guessed_one(self, tmp_path):
        """``resolve_direction`` returns None for an unrecognised key. Defaulting
        to max is exactly what made ``best_metric_name: lpips`` maximise LPIPS
        (#208), so an empty cell is the correct answer here."""
        df = pd.DataFrame(
            [
                {"split": "val", "metric": "zzz_not_a_metric", "value": 1.0, "step": 1},
                {"split": "val", "metric": "zzz_not_a_metric", "value": 9.0, "step": 2},
            ]
        )
        run_summary.make(df, tmp_path)
        row = pd.read_csv(tmp_path / "run_summary.csv").iloc[0]
        assert row["final"] == 9.0
        assert pd.isna(row["best"]) and pd.isna(row["best_step"])
        assert pd.isna(row["direction"]) or row["direction"] == ""

    @pytest.mark.parametrize(
        "df",
        [
            pd.DataFrame(),
            pd.DataFrame([{"unrelated": 1}]),
            pd.DataFrame([{"split": "val", "metric": "val_psnr", "value": None}]),
        ],
        ids=["empty", "wrong-columns", "all-null-values"],
    )
    def test_returns_none_rather_than_a_header_over_nothing(self, df, tmp_path):
        """A header with no rows reads as "measured, came back blank"."""
        assert run_summary.make(df, tmp_path) is None
        assert not list(tmp_path.glob("run_summary.*"))

    def test_works_without_a_step_column(self, tmp_path):
        """Hand-built frames (and older aggregator output) carry no ``step``."""
        df = _frame().drop(columns=["step"])
        assert run_summary.make(df, tmp_path) is not None


class TestThePublicationTablesCannotServeAsTheFloor:
    """Anti-vacuity: if these ever fill on a single run, the floor is redundant."""

    def test_main_results_is_empty_for_a_single_run(self, tmp_path):
        from mriforge.infrastructure.reporting.tables import main_results

        assert main_results.make(_frame(), tmp_path) is None
