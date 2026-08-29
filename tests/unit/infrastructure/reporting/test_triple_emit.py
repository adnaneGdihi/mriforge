"""Tests for ``triple_emit`` — Phase 3.1 of the beautify backlog.

Locks ``TODO/backlog_beautify_logging_and_reporting.md`` Phase 3.1: every
results table renders to LaTeX, Markdown, and CSV in a single call so
each consumer (paper, PR review, downstream analysis) reads its native
format without re-parsing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mriforge.infrastructure.reporting.tables import triple_emit


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Toy results frame: 3 methods × 2 metrics."""
    return pd.DataFrame(
        {
            "method": ["baseline", "ours", "ablation"],
            "psnr": [32.1234, 34.5678, 33.0000],
            "ssim": [0.91, 0.94, 0.92],
        }
    )


def test_triple_emit_writes_three_files(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    """Returns paths to .tex, .md, and .csv, all on disk."""
    paths = triple_emit(sample_df, tmp_path, stem="results")
    assert set(paths) == {"tex", "md", "csv"}
    for p in paths.values():
        assert p.exists()
        assert p.read_text().strip(), f"{p.name} is empty"


def test_csv_round_trips_data_exactly(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    """CSV is the machine-readable canonical — must equal the input frame."""
    paths = triple_emit(sample_df, tmp_path, stem="results")
    reloaded = pd.read_csv(paths["csv"])
    pd.testing.assert_frame_equal(
        reloaded.reset_index(drop=True),
        sample_df.reset_index(drop=True),
        check_dtype=False,
    )


def test_markdown_contains_method_names(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    """Markdown rendering preserves the method-name column."""
    paths = triple_emit(sample_df, tmp_path, stem="results")
    text = paths["md"].read_text()
    for method in sample_df["method"]:
        assert method in text


def test_latex_includes_caption_and_label_when_given(
    sample_df: pd.DataFrame, tmp_path: Path
) -> None:
    """Caption + label arguments propagate into the LaTeX wrapper."""
    paths = triple_emit(
        sample_df,
        tmp_path,
        stem="results",
        caption="Main results comparison",
        label="tab:main_results",
    )
    tex = paths["tex"].read_text()
    assert r"\caption{Main results comparison}" in tex
    assert r"\label{tab:main_results}" in tex
    assert r"\begin{table}" in tex
    assert r"\end{table}" in tex


def test_latex_omits_caption_and_label_when_absent(
    sample_df: pd.DataFrame, tmp_path: Path
) -> None:
    """Without ``caption`` / ``label`` kwargs, the wrapper is minimal."""
    paths = triple_emit(sample_df, tmp_path, stem="results")
    tex = paths["tex"].read_text()
    assert r"\caption" not in tex
    assert r"\label" not in tex
    assert r"\begin{table}" in tex


def test_stem_drives_all_three_filenames(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    """``stem`` shared across the three formats — same basename, different extension."""
    paths = triple_emit(sample_df, tmp_path, stem="my_custom_stem")
    assert paths["tex"].name == "my_custom_stem.tex"
    assert paths["md"].name == "my_custom_stem.md"
    assert paths["csv"].name == "my_custom_stem.csv"


def test_creates_output_dir_if_absent(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    """``out_dir`` is created when it does not exist."""
    target = tmp_path / "deep" / "nested" / "dir"
    assert not target.exists()
    triple_emit(sample_df, target, stem="results")
    assert target.is_dir()
    assert (target / "results.csv").exists()
