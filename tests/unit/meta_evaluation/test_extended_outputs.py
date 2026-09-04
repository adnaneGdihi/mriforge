"""Regression tests for the post-2026-05-12 extensions.

Covers:

* The simulator noise-floor fix — at a high theta even a constant
  phantom must receive non-zero residual energy.
* The continuous gamma sampling fix — at fixed theta, two different
  seeds must produce two different gamma values (not just two signs).
* The four new figure functions (metric_trajectories,
  metric_correlation_clustering, sensitivity_scatter,
  degradation_mosaic) — they must each write a non-empty PNG.
* The CSV exporters — every file must be created with the expected
  header columns and the row count we expect.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest
import torch

from spectramr.core.metrics.meta_evaluation import (
    MetaEvaluationPipeline,
    SimulatorConfig,
)
from spectramr.core.metrics.meta_evaluation.simulator import (
    degrade_gamma,
    degrade_gaussian_noise,
    degrade_rician_noise,
)
from spectramr.core.metrics.meta_evaluation.tables import (
    write_all_tables,
    write_correlation_csv,
    write_leaderboard_latex,
    write_per_family_csv,
    write_per_method_scores_csv,
    write_raw_long_csv,
    write_summary_csv,
)

# ---------------------------------------------------------------------------
# Simulator regression — noise floor + gamma continuous sampling
# ---------------------------------------------------------------------------


def test_gaussian_noise_floor_keeps_noise_on_constant_input() -> None:
    """Before the floor fix, a constant image got zero noise at theta=1."""
    flat = torch.full((1, 16, 16), 0.5)
    gen = torch.Generator().manual_seed(0)
    out = degrade_gaussian_noise(flat, theta=1.0, generator=gen)
    residual = (out - flat).abs().max().item()
    assert residual > 0.0, "constant-image gaussian noise produced no residual"


def test_rician_noise_floor_keeps_noise_on_constant_input() -> None:
    flat = torch.full((1, 16, 16), 0.5)
    gen = torch.Generator().manual_seed(0)
    out = degrade_rician_noise(flat, theta=1.0, generator=gen)
    # Rician is biased upward; abs-residual is non-zero regardless of sign.
    assert (out - flat).abs().max().item() > 0.0


def test_gamma_samples_more_than_two_values() -> None:
    """Previous implementation only sampled gamma ∈ {0.6, 1.4} (the two signs)."""
    img = torch.linspace(0.05, 0.95, 256).reshape(1, 16, 16)
    seen = set()
    for seed in range(20):
        gen = torch.Generator().manual_seed(seed)
        out = degrade_gamma(img, theta=1.0, generator=gen)
        # Recover gamma from a known input pixel: out = sign(x) * |x|^gamma.
        # Pick a non-trivial pixel.
        x = img[0, 0, 5].item()
        y = out[0, 0, 5].item()
        if x > 0 and y > 0:
            gamma = torch.log(torch.tensor(y)) / torch.log(torch.tensor(x))
            seen.add(round(float(gamma), 3))
    # Continuous sampling should produce many distinct gammas, not 2.
    assert len(seen) > 3, f"gamma only took {len(seen)} distinct values: {seen}"


# ---------------------------------------------------------------------------
# Pipeline-output fixture — builds a small real MetaEvaluationOutput
# ---------------------------------------------------------------------------


@pytest.fixture
def small_output(clean_volumes, metric_set):
    """A tiny but real MetaEvaluationOutput for the figure / CSV smoke tests."""
    pipeline = MetaEvaluationPipeline.from_defaults()
    pipeline.simulator_config = SimulatorConfig(
        families=["gaussian_noise", "gamma", "kspace_undersample"],
        n_severities_per_family=4,
        seed=7,
    )
    return pipeline.run(metric_set, clean_volumes[:2])


# ---------------------------------------------------------------------------
# Figures — smoke-test that each new figure writes a PNG
# ---------------------------------------------------------------------------


def test_metric_trajectories_writes_png(small_output, tmp_path: Path) -> None:
    from spectramr.core.metrics.meta_evaluation.figures import metric_trajectories

    path = metric_trajectories(small_output, tmp_path)
    assert path is not None
    assert path.exists() and path.stat().st_size > 0


def test_metric_correlation_clustering_writes_png(small_output, tmp_path: Path) -> None:
    from spectramr.core.metrics.meta_evaluation.figures import (
        metric_correlation_clustering,
    )

    path = metric_correlation_clustering(small_output, tmp_path)
    assert path is not None
    assert path.exists() and path.stat().st_size > 0


def test_sensitivity_scatter_writes_png(small_output, tmp_path: Path) -> None:
    from spectramr.core.metrics.meta_evaluation.figures import sensitivity_scatter

    path = sensitivity_scatter(small_output, tmp_path)
    assert path is not None
    assert path.exists() and path.stat().st_size > 0


def test_degradation_mosaic_writes_png(small_output, tmp_path: Path) -> None:
    from spectramr.core.metrics.meta_evaluation.figures import degradation_mosaic

    path = degradation_mosaic(small_output, tmp_path)
    assert path is not None
    assert path.exists() and path.stat().st_size > 0


# ---------------------------------------------------------------------------
# CSV exporters
# ---------------------------------------------------------------------------


def test_write_summary_csv_has_expected_columns(small_output, tmp_path: Path) -> None:
    path = write_summary_csv(small_output, tmp_path / "summary.csv")
    assert path.exists()
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert rows, "summary CSV has no rows"
    expected_columns = {"metric", "rank", "final_z", "defensible"}
    assert expected_columns.issubset(set(rows[0].keys()))
    # Should also have per-method z columns (z_CDSCR etc.).
    z_cols = [c for c in rows[0].keys() if c.startswith("z_")]
    assert z_cols, "summary CSV missing per-method z columns"


def test_write_per_family_csv_one_row_per_metric_family(
    small_output, tmp_path: Path
) -> None:
    path = write_per_family_csv(small_output, tmp_path / "per_family.csv")
    assert path.exists()
    with path.open() as f:
        rows = list(csv.DictReader(f))
    # 4 metrics × 3 families in the small fixture = 12 rows.
    metric_count = len(small_output.dataset.metric_values)
    family_count = len({
        s.degradation_family for s in small_output.dataset.samples
    })
    assert len(rows) == metric_count * family_count
    assert {"metric", "family", "mean", "std", "spearman_theta"}.issubset(
        set(rows[0].keys())
    )


def test_write_correlation_csv_square_matrix(small_output, tmp_path: Path) -> None:
    path = write_correlation_csv(small_output, tmp_path / "corr.csv")
    assert path.exists()
    with path.open() as f:
        rows = list(csv.reader(f))
    # First row is header [metric, m1, m2, ...]; remaining rows are data.
    header = rows[0]
    body = rows[1:]
    assert len(body) == len(header) - 1, "correlation CSV is not square"
    # Diagonal should be 1.0 (a metric's correlation with itself).
    for i, r in enumerate(body):
        assert float(r[i + 1]) == pytest.approx(1.0, abs=1e-6)


def test_write_raw_long_csv_row_count(small_output, tmp_path: Path) -> None:
    path = write_raw_long_csv(small_output, tmp_path / "raw.csv")
    assert path.exists()
    with path.open() as f:
        rows = list(csv.DictReader(f))
    expected = small_output.dataset.n_samples * len(
        small_output.dataset.metric_values
    )
    assert len(rows) == expected


def test_write_per_method_scores_csv_has_method_columns(
    small_output, tmp_path: Path
) -> None:
    path = write_per_method_scores_csv(
        small_output, tmp_path / "per_method.csv"
    )
    assert path.exists()
    with path.open() as f:
        rows = list(csv.DictReader(f))
    expected_methods = {r.method for r in small_output.per_method}
    assert expected_methods.issubset(set(rows[0].keys()))


# ---------------------------------------------------------------------------
# Professor-facing polish: rounding, ordering, display names, LaTeX
# ---------------------------------------------------------------------------


def test_summary_csv_is_rank_ordered_and_has_display_columns(
    small_output, tmp_path: Path
) -> None:
    path = write_summary_csv(small_output, tmp_path / "summary.csv")
    with path.open() as f:
        rows = list(csv.DictReader(f))
    # New professor-facing columns are present alongside the machine key.
    assert {"display_name", "score_norm", "higher_is_better"}.issubset(
        set(rows[0].keys())
    )
    # Rows are ordered by rank (1, 2, 3, ...), blanks last.
    ranks = [int(r["rank"]) for r in rows if r["rank"]]
    assert ranks == sorted(ranks)
    assert ranks[0] == 1


def test_summary_csv_floats_are_rounded(small_output, tmp_path: Path) -> None:
    path = write_summary_csv(small_output, tmp_path / "summary.csv")
    with path.open() as f:
        rows = list(csv.DictReader(f))
    # No 15-digit float dumps: every finite numeric cell is short.
    for r in rows:
        cell = r["final_z"]
        if cell and cell not in ("nan", ""):
            decimals = cell.split(".")[1] if "." in cell else ""
            assert len(decimals) <= 4, f"un-rounded float in summary: {cell}"


def test_score_norm_is_in_unit_interval(small_output, tmp_path: Path) -> None:
    path = write_summary_csv(small_output, tmp_path / "summary.csv")
    with path.open() as f:
        rows = list(csv.DictReader(f))
    vals = [float(r["score_norm"]) for r in rows if r["score_norm"] not in ("", "nan")]
    assert vals, "no normalized scores written"
    assert all(0.0 <= v <= 1.0 for v in vals)
    # The rank-1 metric should carry the top normalized score (==1.0).
    assert max(vals) == pytest.approx(1.0)


def test_write_leaderboard_latex_emits_booktabs(small_output, tmp_path: Path) -> None:
    path = write_leaderboard_latex(small_output, tmp_path / "leaderboard.tex")
    assert path.exists()
    text = path.read_text()
    # Booktabs, ranked, with the normalized + raw columns.
    for token in (r"\begin{tabular}", r"\toprule", r"\midrule",
                  r"\bottomrule", r"\end{tabular}"):
        assert token in text, f"missing LaTeX token {token!r}"
    # A caption + label make it paper-droppable.
    assert r"\caption" in text
    assert r"\label" in text


def test_write_all_tables_includes_leaderboard(
    small_output, tmp_path: Path
) -> None:
    written = write_all_tables(small_output, tmp_path)
    names = {p.name for p in written}
    assert "leaderboard.tex" in names
    assert "summary.csv" in names


# ---------------------------------------------------------------------------
# B3 regression: a table that fails to write must not vanish silently
# ---------------------------------------------------------------------------


def test_write_all_tables_raises_when_a_table_fails(
    small_output, tmp_path: Path, monkeypatch
) -> None:
    """Regression for B3: `except Exception: continue` let a table disappear.

    A *missing* results file is the one failure nobody notices -- you cannot see
    the table that isn't there. Strict mode is the default so the run stops.
    """
    from spectramr.core.metrics.meta_evaluation import tables as tables_mod

    def _boom(_output, _path):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(tables_mod, "write_correlation_csv", _boom)

    with pytest.raises(tables_mod.TableWriteError, match=re.escape("correlation.csv")):
        tables_mod.write_all_tables(small_output, tmp_path)


def test_write_all_tables_non_strict_reports_every_failure(
    small_output, tmp_path: Path, monkeypatch
) -> None:
    """Non-strict still writes the healthy tables and never fabricates the broken one."""
    from spectramr.core.metrics.meta_evaluation import tables as tables_mod

    def _boom(_output, _path):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(tables_mod, "write_correlation_csv", _boom)

    written = tables_mod.write_all_tables(small_output, tmp_path, strict=False)
    names = {p.name for p in written}
    assert "summary.csv" in names
    assert "correlation.csv" not in names
