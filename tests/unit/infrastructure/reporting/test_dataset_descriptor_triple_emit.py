"""Tests for the dataset descriptor table emitter.

Locks the Phase 3.1 refactor from
``TODO/backlog_beautify_logging_and_reporting.md``: the descriptor
now flows through ``triple_emit`` so reviewers get LaTeX + Markdown
+ CSV in one call (rather than just LaTeX + Markdown).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mriforge.infrastructure.reporting.tables.dataset_descriptor import make


@pytest.fixture()
def sample_cohort() -> dict:
    return {
        "n_total": 200,
        "train_n": 160,
        "val_n": 20,
        "test_n": 20,
        "age_mean": 42.0,
        "age_std": 14.5,
        "sex_split": "F:50/M:50",
        "scanners": "GE/Siemens",
        "field_strength": "1.5 T",
        "pathology": "tumor 30, control 170",
        "split_rule": "subject-level",
        "doi": "10.1234/example",
    }


def test_descriptor_emits_three_files(tmp_path: Path, sample_cohort: dict) -> None:
    """The refactored ``make`` writes ``.tex``, ``.md``, and ``.csv``."""
    df = pd.DataFrame({"x": []})  # unused by descriptor; uniform signature
    paths = make(df, tmp_path, cohort=sample_cohort)
    assert paths is not None
    for key in ("latex", "markdown", "csv"):
        assert paths[key].exists()
        assert paths[key].read_text().strip(), f"{key} file is empty"


def test_descriptor_csv_contains_all_fields(tmp_path: Path, sample_cohort: dict) -> None:
    """Every cohort field appears in the machine-readable CSV."""
    paths = make(pd.DataFrame(), tmp_path, cohort=sample_cohort)
    csv = pd.read_csv(paths["csv"])
    assert "Field" in csv.columns
    assert "Value" in csv.columns
    # All 12 cohort fields should be rows.
    assert len(csv) == 12
    # Spot-check a few — values are stringified in the CSV.
    assert "Total subjects" in csv["Field"].tolist()
    assert "200" in csv["Value"].astype(str).tolist()


def test_descriptor_markdown_has_field_value_columns(
    tmp_path: Path, sample_cohort: dict
) -> None:
    """The markdown rendering uses the expected ``Field | Value`` schema."""
    paths = make(pd.DataFrame(), tmp_path, cohort=sample_cohort)
    md = paths["markdown"].read_text()
    assert "| Field | Value |" in md
    assert "Total subjects" in md


def test_descriptor_latex_includes_caption_and_label(
    tmp_path: Path, sample_cohort: dict
) -> None:
    """LaTeX rendering wraps the table with caption + label."""
    paths = make(pd.DataFrame(), tmp_path, cohort=sample_cohort)
    tex = paths["latex"].read_text()
    assert r"\caption{Dataset descriptor" in tex
    assert r"\label{tab:dataset}" in tex


def test_descriptor_returns_none_when_cohort_empty(tmp_path: Path) -> None:
    """Empty cohort short-circuits to ``None`` (existing contract)."""
    assert make(pd.DataFrame(), tmp_path, cohort=None) is None
    assert make(pd.DataFrame(), tmp_path, cohort={}) is None
