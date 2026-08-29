"""Tests for the three audit_plan_novel reporting figures + KSD certificate.

These tests render a small synthetic dataframe through each new
plotter and assert the output PDF exists and is non-empty. Plotters
that fail silently on bad input must also return ``None`` rather than
raise; we test both branches.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


# Importing the package triggers registration of all plotters.
from mriforge.infrastructure.reporting.plotters import (  # noqa: E402
    PLOTTERS,
    active_acquisition_trajectory,
    bloch_consistency_residual,
    qmap_riemannian_vs_euclidean,
)
from mriforge.infrastructure.reporting.plotters.ksd_certificate import (
    render_ksd_certificate,
)


def test_three_new_plotters_registered() -> None:
    for fig_id in (
        "fig_2_15_active_acquisition_trajectory",
        "fig_b3_bloch_consistency_residual",
        "fig_b7_qmap_riemannian_vs_euclidean",
    ):
        assert fig_id in PLOTTERS, f"plotter {fig_id} is not registered"


def test_active_acquisition_trajectory_from_dataframe(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "step": [0, 1, 2, 0, 1, 2],
            "psnr_db": [20.0, 22.0, 24.0, 19.0, 21.0, 23.0],
            "policy": ["ib"] * 3 + ["random"] * 3,
        }
    )
    out = active_acquisition_trajectory.make(df, tmp_path / "fig.pdf")
    assert out is not None
    assert out.is_file()
    assert out.stat().st_size > 1000  # non-trivial PDF.


def test_active_acquisition_trajectory_from_csvs(tmp_path: Path) -> None:
    csv_a = tmp_path / "ib.csv"
    csv_b = tmp_path / "random.csv"
    csv_a.write_text("step,psnr_db,acquired_lines\n0,18.0,4\n1,21.0,5\n2,24.0,6\n")
    csv_b.write_text("step,psnr_db,acquired_lines\n0,18.0,4\n1,19.0,5\n2,21.0,6\n")
    out = active_acquisition_trajectory.make(
        None,
        tmp_path / "fig.pdf",
        csv_paths=[csv_a, csv_b],
    )
    assert out is not None
    assert out.is_file()


def test_active_acquisition_trajectory_empty_returns_none(tmp_path: Path) -> None:
    out = active_acquisition_trajectory.make(
        pd.DataFrame(columns=["step", "psnr_db", "policy"]),
        tmp_path / "fig.pdf",
    )
    assert out is None


def test_bloch_consistency_residual_renders(tmp_path: Path) -> None:
    rows = []
    for tissue in ("WM", "GM", "CSF"):
        for step in (0, 100, 200):
            rows.append(
                {
                    "tissue": tissue,
                    "bloch_eq_residual": 1.0 / (step + 1),
                    "step": step,
                }
            )
    df = pd.DataFrame(rows)
    out = bloch_consistency_residual.make(df, tmp_path / "fig.pdf")
    assert out is not None
    assert out.is_file()


def test_bloch_consistency_residual_missing_columns_returns_none(tmp_path: Path) -> None:
    df = pd.DataFrame({"tissue": ["WM"], "step": [0]})  # no residual
    out = bloch_consistency_residual.make(df, tmp_path / "fig.pdf")
    assert out is None


def test_qmap_riemannian_vs_euclidean_violins(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "method": ["euclidean"] * 10 + ["riemannian"] * 10,
            "error": list(np.random.rand(10) + 0.5) + list(np.random.rand(10) * 0.3),
        }
    )
    out = qmap_riemannian_vs_euclidean.make(df, tmp_path / "fig.pdf")
    assert out is not None
    assert out.is_file()


def test_qmap_riemannian_vs_euclidean_with_slices(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "method": ["euclidean"] * 5 + ["riemannian"] * 5,
            "error": list(np.linspace(0.1, 1.0, 5)) + list(np.linspace(0.05, 0.5, 5)),
        }
    )
    slices = {
        "euclidean": np.random.rand(16, 16),
        "riemannian": np.random.rand(16, 16) * 0.5,
    }
    out = qmap_riemannian_vs_euclidean.make(
        df,
        tmp_path / "fig.pdf",
        slices=slices,
    )
    assert out is not None
    assert out.is_file()


def test_ksd_certificate_pass_outcome(tmp_path: Path) -> None:
    result = SimpleNamespace(
        passed=True,
        ksd_squared=0.001,
        p_value=0.42,
        threshold=0.05,
        n_samples=256,
        bandwidth=None,
        message="KSD² = 1e-3, p = 0.42 (threshold 0.050). PASS",
    )
    out = render_ksd_certificate(
        tmp_path / "cert.pdf",
        result=result,
        config_path="experiments/inprogress/novel_2026/idea_7_ksd_defensibility_audit.yaml",
    )
    assert out.is_file()
    assert out.stat().st_size > 1000


def test_ksd_certificate_fail_outcome(tmp_path: Path) -> None:
    result = SimpleNamespace(
        passed=False,
        ksd_squared=2.3,
        p_value=0.001,
        threshold=0.05,
        n_samples=128,
        bandwidth=0.7,
        message="KSD² = 2.3, p = 0.001 (threshold 0.050). FAIL — model does not pass goodness-of-fit.",
    )
    out = render_ksd_certificate(tmp_path / "cert_fail.pdf", result=result)
    assert out.is_file()
