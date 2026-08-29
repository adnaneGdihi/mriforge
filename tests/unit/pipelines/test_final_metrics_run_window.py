"""``final_metrics.json`` must summarise THIS run, not the CSV's history (#586).

``logs/training_metrics.csv`` lives in the arm's output directory and is
APPENDED to by every run that writes there. Unwindowed, the ``best`` block
reports whichever of those runs happened to score best. The observed case: an
exp_11 run of 3000 iterations with ``log_interval: 5000`` wrote ZERO rows, and
its ``final_metrics.json`` reported a five-run-blended CSV's minimum as its own.
"""

import pytest

from mriforge.pipelines.train import (
    _select_current_run_rows,
    _summarise_best_metrics_from_csv,
)


def _row(iteration, loss, psnr=None):
    row = {"iteration": str(iteration), "complex_l1": str(loss)}
    if psnr is not None:
        row["train_psnr"] = str(psnr)
    return row


def _write_csv(tmp_path, rows, columns=("iteration", "complex_l1")):
    path = tmp_path / "training_metrics.csv"
    lines = [",".join(columns)]
    lines += [",".join(str(r.get(c, "")) for c in columns) for r in rows]
    path.write_text("\n".join(lines) + "\n")
    return str(path)


# --------------------------------------------------------------------------
# _select_current_run_rows: the iteration column resets when a run restarts
# --------------------------------------------------------------------------


def test_selects_only_the_final_ascending_segment():
    rows = [_row(5000, 0.5), _row(10000, 0.4), _row(5000, 0.9), _row(10000, 0.8)]
    assert _select_current_run_rows(rows) == rows[2:]


def test_five_blended_runs_keep_only_the_last():
    """The shape of the real exp_11 CSV: five appended runs."""
    rows = []
    for _ in range(4):
        rows += [_row(i, 0.1) for i in (5000, 10000, 15000, 20000)]
    tail = [_row(i, 0.9) for i in (5000, 10000)]
    assert _select_current_run_rows(rows + tail) == tail


def test_single_run_is_returned_whole():
    rows = [_row(i, 0.5) for i in (100, 200, 300)]
    assert _select_current_run_rows(rows) == rows


def test_rows_without_a_usable_iteration_are_passed_through():
    """No windowing signal — return everything rather than silently drop data."""
    rows = [{"complex_l1": "0.5"}, {"complex_l1": "0.4"}]
    assert _select_current_run_rows(rows) == rows


def test_empty_input():
    assert _select_current_run_rows([]) == []


# --------------------------------------------------------------------------
# _summarise_best_metrics_from_csv: final_iteration closes the zero-rows hole
# --------------------------------------------------------------------------


def test_run_that_wrote_no_rows_reports_no_bests(tmp_path):
    """THE regression: 3000 iterations, log_interval 5000, so zero rows written.

    Every row in the file belongs to a previous run that reached 20000. Segment
    detection alone cannot see this — the final segment IS that previous run —
    so ``final_iteration`` is what empties the window.
    """
    csv_path = _write_csv(
        tmp_path, [_row(i, 0.117) for i in (5000, 10000, 15000, 20000)]
    )
    assert _summarise_best_metrics_from_csv(csv_path, final_iteration=3000) == {}


def test_unwindowed_call_still_reports_the_stale_best(tmp_path):
    """Without ``final_iteration`` the old (wrong) answer is what you get.

    Pins WHY the caller must pass it, so a future refactor that drops the
    argument fails here instead of silently regressing #586.
    """
    csv_path = _write_csv(
        tmp_path, [_row(i, 0.117) for i in (5000, 10000, 15000, 20000)]
    )
    assert _summarise_best_metrics_from_csv(csv_path)[
        "complex_l1_best"
    ] == pytest.approx(0.117)


def test_current_run_rows_are_summarised(tmp_path):
    csv_path = _write_csv(
        tmp_path,
        [_row(5000, 0.9), _row(10000, 0.8), _row(100, 0.5), _row(200, 0.3)],
    )
    best = _summarise_best_metrics_from_csv(csv_path, final_iteration=200)
    assert best["complex_l1_best"] == pytest.approx(0.3)


def test_higher_is_better_columns_take_the_maximum(tmp_path):
    csv_path = _write_csv(
        tmp_path,
        [_row(100, 0.5, psnr=20.0), _row(200, 0.3, psnr=24.0)],
        columns=("iteration", "complex_l1", "train_psnr"),
    )
    best = _summarise_best_metrics_from_csv(csv_path, final_iteration=200)
    assert best["train_psnr_best"] == pytest.approx(24.0)
    assert best["complex_l1_best"] == pytest.approx(0.3)


def test_missing_csv_is_empty():
    assert _summarise_best_metrics_from_csv("/nonexistent/training_metrics.csv") == {}
    assert _summarise_best_metrics_from_csv(None) == {}
