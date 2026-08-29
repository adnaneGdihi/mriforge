"""Unit tests for :mod:`mriforge.infrastructure.orchestration.campaign_report_generator`.

The generator produces an HTML dashboard + side-car plot PNGs and a
JSON evaluation_report. These tests exercise:

* the smoke path with a populated 3-arm report,
* the empty-results edge case,
* missing-metric handling (one arm lacks a metric others have),
* determinism of the HTML output for the same input.

No real training artefacts are touched; ``tmp_path`` is the only on-disk
output location.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from mriforge.infrastructure.orchestration.campaign_evaluator import (
    CampaignReport,
    PairwiseResult,
    UncertaintyReport,
)
from mriforge.infrastructure.orchestration.campaign_report_generator import (
    CampaignReportGenerator,
)


# ── Helpers ─────────────────────────────────────────────────────────


def _make_leaderboard(rows: list[dict]) -> pd.DataFrame:
    """Construct a leaderboard frame in the schema the generator expects."""
    return pd.DataFrame(rows, columns=["experiment", "metric", "mean", "std", "n"])


def _build_full_report(name: str = "campaign_x") -> CampaignReport:
    """Build a populated report with 3 arms, 2 metrics."""
    lb = _make_leaderboard(
        [
            {"experiment": "arm_a", "metric": "psnr", "mean": 30.5, "std": 0.4, "n": 5},
            {"experiment": "arm_b", "metric": "psnr", "mean": 31.2, "std": 0.3, "n": 5},
            {"experiment": "arm_c", "metric": "psnr", "mean": 29.8, "std": 0.5, "n": 5},
            {"experiment": "arm_a", "metric": "ssim", "mean": 0.85, "std": 0.01, "n": 5},
            {"experiment": "arm_b", "metric": "ssim", "mean": 0.87, "std": 0.01, "n": 5},
            {"experiment": "arm_c", "metric": "ssim", "mean": 0.83, "std": 0.02, "n": 5},
        ]
    )

    # Pairwise vs baseline (arm_a is baseline).
    pairwise = [
        PairwiseResult(
            experiment_a="arm_a",
            experiment_b="arm_b",
            metric="psnr",
            paired_ttest={"p_value": 0.01},
            wilcoxon={"p_value": 0.02},
            effect_size={"cohens_d": 1.2, "interpretation": "large"},
            bootstrap_ci={"low": 0.4, "high": 1.0},
        ),
        PairwiseResult(
            experiment_a="arm_a",
            experiment_b="arm_c",
            metric="psnr",
            paired_ttest={"p_value": 0.20},
            wilcoxon={"p_value": 0.22},
            effect_size={"cohens_d": -0.3, "interpretation": "small"},
            bootstrap_ci={"low": -0.6, "high": 0.1},
        ),
    ]

    sig_psnr = pd.DataFrame(
        [
            [1.0, 0.01, 0.20],
            [0.01, 1.0, 0.05],
            [0.20, 0.05, 1.0],
        ],
        index=["arm_a", "arm_b", "arm_c"],
        columns=["arm_a", "arm_b", "arm_c"],
    )

    uncertainty = [
        UncertaintyReport(
            experiment_name="arm_a",
            ece=0.04,
            sharpness=0.10,
            uncertainty_error_correlation=0.55,
        ),
        UncertaintyReport(
            experiment_name="arm_b",
            ece=0.03,
            sharpness=0.08,
            uncertainty_error_correlation=0.62,
        ),
        UncertaintyReport(
            experiment_name="arm_c",
            ece=0.07,
            sharpness=0.12,
            uncertainty_error_correlation=0.40,
        ),
    ]

    return CampaignReport(
        campaign_name=name,
        leaderboard=lb,
        pairwise_results=pairwise,
        significance_matrix={"psnr": sig_psnr},
        uncertainty_reports=uncertainty,
        summary="Three-arm campaign summary.\nArm B leads on PSNR.",
    )


# ── Smoke: full report renders end-to-end ────────────────────────────


def test_generate_full_report_smoke(tmp_path: Path) -> None:
    """Synthetic 3-arm report renders a non-empty HTML + plots + JSON."""
    report = _build_full_report()
    gen = CampaignReportGenerator(output_dir=str(tmp_path))

    html_path = gen.generate(report)

    html_file = Path(html_path)
    assert html_file.exists()
    assert html_file.suffix == ".html"
    content = html_file.read_text(encoding="utf-8")
    assert len(content) > 0
    # Sanity: campaign name and arm names propagate into the dashboard.
    assert "campaign_x" in content
    assert "arm_a" in content
    assert "arm_b" in content
    assert "arm_c" in content
    # JSON side-car is written.
    assert (tmp_path / "evaluation_report.json").exists()
    # Plots directory contains at least one PNG.
    pngs = list((tmp_path / "plots").glob("*.png"))
    assert len(pngs) >= 1


def test_generate_creates_named_plots(tmp_path: Path) -> None:
    """Each populated section produces its own plot file."""
    report = _build_full_report()
    gen = CampaignReportGenerator(output_dir=str(tmp_path))
    gen.generate(report)

    plots_dir = tmp_path / "plots"
    expected = {
        "leaderboard.png",
        "distributions.png",
        "significance_psnr.png",
        "effect_sizes.png",
        "uncertainty_calibration.png",
    }
    produced = {p.name for p in plots_dir.glob("*.png")}
    # All expected plots present (the source emits these exact names).
    assert expected.issubset(produced)


# ── Empty-results edge case ──────────────────────────────────────────


def test_generate_with_empty_report(tmp_path: Path) -> None:
    """An empty CampaignReport doesn't crash; HTML + JSON still emitted."""
    report = CampaignReport(campaign_name="empty_campaign", summary="(no arms)")
    gen = CampaignReportGenerator(output_dir=str(tmp_path))

    html_path = gen.generate(report)
    html_file = Path(html_path)

    assert html_file.exists()
    content = html_file.read_text(encoding="utf-8")
    assert len(content) > 0
    # Still references the campaign and the placeholder summary.
    assert "empty_campaign" in content
    assert "no arms" in content
    # JSON side-car still emitted.
    assert (tmp_path / "evaluation_report.json").exists()
    # No plot files written, since nothing to plot.
    pngs = list((tmp_path / "plots").glob("*.png"))
    assert pngs == []


def test_generate_with_empty_leaderboard_dataframe(tmp_path: Path) -> None:
    """A leaderboard that is an empty DataFrame is also treated as empty."""
    report = CampaignReport(
        campaign_name="empty_lb",
        leaderboard=_make_leaderboard([]),
        summary="empty leaderboard",
    )
    gen = CampaignReportGenerator(output_dir=str(tmp_path))
    html_path = gen.generate(report)

    assert Path(html_path).exists()
    pngs = list((tmp_path / "plots").glob("*.png"))
    assert pngs == []


# ── Missing-metric handling ──────────────────────────────────────────


def test_missing_metric_renders_with_nan_placeholder(tmp_path: Path) -> None:
    """If one arm lacks SSIM but others have it, no KeyError; nan flows through."""
    # arm_c is missing SSIM.
    lb = _make_leaderboard(
        [
            {"experiment": "arm_a", "metric": "psnr", "mean": 30.0, "std": 0.4, "n": 5},
            {"experiment": "arm_b", "metric": "psnr", "mean": 31.0, "std": 0.3, "n": 5},
            {"experiment": "arm_c", "metric": "psnr", "mean": 29.5, "std": 0.5, "n": 5},
            {"experiment": "arm_a", "metric": "ssim", "mean": 0.85, "std": 0.01, "n": 5},
            {"experiment": "arm_b", "metric": "ssim", "mean": 0.87, "std": 0.01, "n": 5},
            # arm_c has nan SSIM mean/std — generator must not KeyError.
            {
                "experiment": "arm_c",
                "metric": "ssim",
                "mean": float("nan"),
                "std": float("nan"),
                "n": 0,
            },
        ]
    )
    report = CampaignReport(
        campaign_name="missing_metric_campaign",
        leaderboard=lb,
        summary="One arm missing SSIM.",
    )
    gen = CampaignReportGenerator(output_dir=str(tmp_path))
    html_path = gen.generate(report)

    # No exception; output produced.
    assert Path(html_path).exists()
    # Leaderboard PNG produced despite the NaN row.
    assert (tmp_path / "plots" / "leaderboard.png").exists()
    # The HTML embeds the leaderboard table — verify both arms surface.
    content = Path(html_path).read_text(encoding="utf-8")
    assert "arm_a" in content
    assert "arm_c" in content


def test_pairwise_results_without_cohens_d_are_skipped(tmp_path: Path) -> None:
    """Effect-size plot is skipped entirely if no result has cohens_d set."""
    lb = _make_leaderboard(
        [
            {"experiment": "arm_a", "metric": "psnr", "mean": 30.0, "std": 0.4, "n": 5},
            {"experiment": "arm_b", "metric": "psnr", "mean": 31.0, "std": 0.3, "n": 5},
        ]
    )
    pairwise = [
        PairwiseResult(
            experiment_a="arm_a",
            experiment_b="arm_b",
            metric="psnr",
            paired_ttest={"p_value": 0.01},
            wilcoxon={"p_value": 0.02},
            effect_size={},  # no cohens_d
            bootstrap_ci={},
        ),
    ]
    report = CampaignReport(
        campaign_name="no_d_campaign",
        leaderboard=lb,
        pairwise_results=pairwise,
        summary="No effect size available.",
    )
    gen = CampaignReportGenerator(output_dir=str(tmp_path))
    gen.generate(report)
    # No effect-sizes PNG, but leaderboard PNG still exists.
    assert not (tmp_path / "plots" / "effect_sizes.png").exists()
    assert (tmp_path / "plots" / "leaderboard.png").exists()


def test_uncertainty_reports_without_ece_are_skipped(tmp_path: Path) -> None:
    """If every uncertainty report has ece=None, no calibration plot is made."""
    lb = _make_leaderboard(
        [
            {"experiment": "arm_a", "metric": "psnr", "mean": 30.0, "std": 0.4, "n": 5},
            {"experiment": "arm_b", "metric": "psnr", "mean": 31.0, "std": 0.3, "n": 5},
        ]
    )
    report = CampaignReport(
        campaign_name="no_ece_campaign",
        leaderboard=lb,
        uncertainty_reports=[
            UncertaintyReport(experiment_name="arm_a", ece=None),
            UncertaintyReport(experiment_name="arm_b", ece=None),
        ],
        summary="No ECE present.",
    )
    gen = CampaignReportGenerator(output_dir=str(tmp_path))
    gen.generate(report)
    assert not (tmp_path / "plots" / "uncertainty_calibration.png").exists()


# ── Determinism ─────────────────────────────────────────────────────


def test_html_dashboard_is_deterministic(tmp_path: Path) -> None:
    """Same input → byte-identical HTML (diffability for code review)."""
    report_a = _build_full_report(name="repro_campaign")
    report_b = _build_full_report(name="repro_campaign")

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    CampaignReportGenerator(output_dir=str(out_a)).generate(report_a)
    CampaignReportGenerator(output_dir=str(out_b)).generate(report_b)

    html_a = (out_a / "campaign_dashboard.html").read_bytes()
    html_b = (out_b / "campaign_dashboard.html").read_bytes()
    assert html_a == html_b


# ── Misc: constructor side-effects ──────────────────────────────────


def test_constructor_creates_plots_dir(tmp_path: Path) -> None:
    """Instantiating the generator creates output_dir/plots eagerly."""
    target = tmp_path / "campaign_out"
    assert not target.exists()
    CampaignReportGenerator(output_dir=str(target))
    assert (target / "plots").is_dir()


def test_generate_returns_string_path(tmp_path: Path) -> None:
    """generate() returns the HTML path as a string (not a Path)."""
    report = _build_full_report()
    gen = CampaignReportGenerator(output_dir=str(tmp_path))
    result = gen.generate(report)
    assert isinstance(result, str)
    assert result.endswith("campaign_dashboard.html")


# ── Sanity: report.to_dict survives nan placeholders ────────────────


def test_report_to_dict_preserves_nan_leaderboard() -> None:
    """Sanity check that CampaignReport.to_dict() survives a NaN row.

    This protects the JSON sidecar path used by CampaignReportGenerator.
    """
    lb = _make_leaderboard(
        [
            {"experiment": "arm_a", "metric": "psnr", "mean": 30.0, "std": 0.4, "n": 5},
            {
                "experiment": "arm_b",
                "metric": "psnr",
                "mean": float("nan"),
                "std": float("nan"),
                "n": 0,
            },
        ]
    )
    report = CampaignReport(campaign_name="nan_lb", leaderboard=lb, summary="nan row")
    d = report.to_dict()
    assert d["campaign_name"] == "nan_lb"
    rows = d["leaderboard"]
    assert len(rows) == 2
    # The NaN survives without raising.
    nan_row = next(r for r in rows if r["experiment"] == "arm_b")
    assert math.isnan(nan_row["mean"]) or nan_row["mean"] is None


def test_custom_figsize_and_dpi_accepted(tmp_path: Path) -> None:
    """Constructor honours non-default figsize/dpi without crashing."""
    report = _build_full_report()
    gen = CampaignReportGenerator(
        output_dir=str(tmp_path), figsize=(8, 5), dpi=80
    )
    html = gen.generate(report)
    assert Path(html).exists()
