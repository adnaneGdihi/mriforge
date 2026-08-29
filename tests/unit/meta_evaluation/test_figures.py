"""Smoke tests for the figure generator.

We do not validate pixel content; we only assert that ``render_all``
writes the expected PNG files for a typical pipeline output.
"""

from __future__ import annotations

import torch

from mriforge.core.metrics.meta_evaluation import (
    AggregatorConfig,
    MetaEvaluationPipeline,
    MetricSet,
    SimulatorConfig,
)


def _phantom(seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32), indexing="ij"
    )
    r = torch.sqrt(xx**2 + yy**2)
    return (
        (torch.exp(-(r**2) / 0.5) + 0.05 * torch.randn((32, 32), generator=gen))
        .unsqueeze(0)
        .float()
    )


def _small_output():
    """A minimal but complete pipeline output — enough for every renderer to fire."""
    metric_set = MetricSet(
        metrics={
            "mse": lambda p, t: float(((p.float() - t.float()) ** 2).mean().item()),
            "mae": lambda p, t: float((p.float() - t.float()).abs().mean().item()),
        },
        higher_is_better={"mse": False, "mae": False},
        differentiable={"mse": True, "mae": True},
    )
    pipeline = MetaEvaluationPipeline.from_defaults()
    pipeline.simulator_config = SimulatorConfig(
        families=["gaussian_noise", "gaussian_blur", "gamma"],
        n_severities_per_family=3,
        seed=2,
    )
    pipeline.aggregator_config = AggregatorConfig(min_positive_axes=3)
    return pipeline.run(metric_set, [("a", _phantom(0)), ("b", _phantom(1))])


def test_render_all_writes_pngs(tmp_path) -> None:
    from mriforge.core.metrics.meta_evaluation import render_figures

    metric_set = MetricSet(
        metrics={
            "mse": lambda p, t: float(((p.float() - t.float()) ** 2).mean().item()),
            "mae": lambda p, t: float((p.float() - t.float()).abs().mean().item()),
        },
        higher_is_better={"mse": False, "mae": False},
        differentiable={"mse": True, "mae": True},
    )
    pipeline = MetaEvaluationPipeline.from_defaults()
    pipeline.simulator_config = SimulatorConfig(
        families=["gaussian_noise", "gaussian_blur", "gamma"],
        n_severities_per_family=3,
        seed=2,
    )
    pipeline.aggregator_config = AggregatorConfig(min_positive_axes=3)
    output = pipeline.run(metric_set, [("a", _phantom(0)), ("b", _phantom(1))])

    written = render_figures(output, tmp_path)
    # All 7+ figures should land (the consensus renderer writes 2).
    assert len(written) >= 6
    for p in written:
        assert p.exists()
        assert p.suffix == ".png"
        assert p.stat().st_size > 1024  # non-trivial PNG content


def test_sim2rank_breakdown_emits_separate_panels(tmp_path) -> None:
    """ADR / SCVR / CDRS each get their own PNG instead of a 1×3 grid."""
    from mriforge.core.metrics.meta_evaluation import render_figures

    metric_set = MetricSet(
        metrics={
            "mse": lambda p, t: float(((p.float() - t.float()) ** 2).mean().item()),
            "mae": lambda p, t: float((p.float() - t.float()).abs().mean().item()),
        },
        higher_is_better={"mse": False, "mae": False},
        differentiable={"mse": True, "mae": True},
    )
    pipeline = MetaEvaluationPipeline.from_defaults()
    pipeline.simulator_config = SimulatorConfig(
        families=["gaussian_noise", "gaussian_blur", "gamma"],
        n_severities_per_family=3,
        seed=3,
    )
    pipeline.aggregator_config = AggregatorConfig(min_positive_axes=3)
    output = pipeline.run(metric_set, [("a", _phantom(0)), ("b", _phantom(1))])
    render_figures(output, tmp_path)

    # Sim2Rank panels — one file per sub-axis (was a 1×3 subplot grid).
    for fname in (
        "fig6a_sim2rank_adr.png",
        "fig6b_sim2rank_scvr.png",
        "fig6c_sim2rank_cdrs.png",
    ):
        assert (tmp_path / fname).exists(), f"missing split panel {fname}"


def test_top_bottom_indices_helper() -> None:
    """The inlined top-3/bottom-3 helper matches the SSOT logic."""
    from mriforge.core.metrics.meta_evaluation.figures import (
        _top_bottom_indices,
    )

    top, bottom = _top_bottom_indices([0.1, 0.9, 0.5, 0.3, 0.7], k=2)
    assert top == [1, 4]
    assert bottom == [0, 3]
    # NaN-safe.
    top, _ = _top_bottom_indices([float("nan"), 0.5, 0.1], k=2)
    assert 0 not in top


def test_metric_trajectories_emits_one_png_per_metric(tmp_path) -> None:
    """Each metric gets its own trajectory PNG under trajectories/."""
    from mriforge.core.metrics.meta_evaluation import render_figures

    metric_set = MetricSet(
        metrics={
            "mse": lambda p, t: float(((p.float() - t.float()) ** 2).mean().item()),
            "mae": lambda p, t: float((p.float() - t.float()).abs().mean().item()),
        },
        higher_is_better={"mse": False, "mae": False},
        differentiable={"mse": True, "mae": True},
    )
    pipeline = MetaEvaluationPipeline.from_defaults()
    pipeline.simulator_config = SimulatorConfig(
        families=["gaussian_noise", "gaussian_blur", "gamma"],
        n_severities_per_family=3,
        seed=4,
    )
    pipeline.aggregator_config = AggregatorConfig(min_positive_axes=3)
    output = pipeline.run(metric_set, [("a", _phantom(0)), ("b", _phantom(1))])
    render_figures(output, tmp_path)

    traj_dir = tmp_path / "trajectories"
    assert traj_dir.is_dir(), "trajectories/ subdir missing"
    rendered = {p.stem for p in traj_dir.glob("*.png")}
    assert "mse" in rendered
    assert "mae" in rendered
    assert (tmp_path / "fig9_metric_trajectories_index.png").exists()


def _run_output(seed: int):
    metric_set = MetricSet(
        metrics={
            "mse": lambda p, t: float(((p.float() - t.float()) ** 2).mean().item()),
            "mae": lambda p, t: float((p.float() - t.float()).abs().mean().item()),
        },
        higher_is_better={"mse": False, "mae": False},
        differentiable={"mse": True, "mae": True},
    )
    pipeline = MetaEvaluationPipeline.from_defaults()
    pipeline.simulator_config = SimulatorConfig(
        families=["gaussian_noise", "gaussian_blur", "gamma"],
        n_severities_per_family=3,
        seed=seed,
    )
    pipeline.aggregator_config = AggregatorConfig(min_positive_axes=3)
    return pipeline.run(metric_set, [("a", _phantom(0)), ("b", _phantom(1))])


def test_normalized_leaderboard_figure_written(tmp_path) -> None:
    """The headline normalized [0,1] consensus leaderboard is emitted."""
    from mriforge.core.metrics.meta_evaluation import render_figures

    render_figures(_run_output(7), tmp_path)
    fig = tmp_path / "fig13_normalized_leaderboard.png"
    assert fig.exists()
    assert fig.stat().st_size > 1024


def test_severity_response_heatmap_written(tmp_path) -> None:
    """Metric x degradation-family Spearman heatmap is emitted."""
    from mriforge.core.metrics.meta_evaluation import render_figures

    render_figures(_run_output(8), tmp_path)
    fig = tmp_path / "fig14_severity_response_heatmap.png"
    assert fig.exists()
    assert fig.stat().st_size > 1024


def test_render_all_is_layout_warning_free(tmp_path) -> None:
    """No matplotlib layout warnings (the tight_layout + colorbar class).

    The old renderers called ``fig.tight_layout()`` after adding a colorbar
    or suptitle, which matplotlib reports as a UserWarning. The
    constrained-layout factory must eliminate that whole class.
    """
    import warnings

    from mriforge.core.metrics.meta_evaluation import render_figures

    output = _run_output(9)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        render_figures(output, tmp_path)
    layout_warnings = [
        str(w.message)
        for w in caught
        if "layout" in str(w.message).lower() or "tight" in str(w.message).lower()
    ]
    assert not layout_warnings, f"layout warnings leaked: {layout_warnings}"


# ---------------------------------------------------------------------------
# matplotlib optional-dependency guard (6.1). The module imports even in a
# degraded env; figure construction (routed through _report_style.figure)
# raises a clear error only when matplotlib is genuinely absent.
# ---------------------------------------------------------------------------


def test_figures_module_exposes_matplotlib_flag() -> None:
    from mriforge.core.metrics.meta_evaluation import figures

    assert isinstance(figures.MATPLOTLIB_AVAILABLE, bool)


def test_render_raises_clear_error_without_matplotlib(tmp_path, monkeypatch) -> None:
    """With matplotlib unavailable, the first figure factory call raises a
    clear ImportError instead of crashing on ``None.subplots``."""
    import pytest

    from mriforge.core.metrics.meta_evaluation import _report_style as rs

    monkeypatch.setattr(rs, "MATPLOTLIB_AVAILABLE", False)
    with pytest.raises(ImportError, match="matplotlib"):
        rs.figure()


# ---------------------------------------------------------------------------
# B3 regression: a figure that RAISES is a bug; a figure that returns None is not
# ---------------------------------------------------------------------------


def test_render_all_raises_when_a_figure_raises(tmp_path, monkeypatch) -> None:
    """Regression for B3: a raising renderer was logged at WARNING and skipped.

    The sweep then exited zero with a figure quietly missing from the report.
    """
    import pytest

    from mriforge.core.metrics.meta_evaluation import figures as figures_mod

    # Named to match the renderer it stands in for: render_all reports the failing
    # figure by `fn.__name__`, and the error must name it.
    def normalized_leaderboard(_output, _out_dir):
        raise RuntimeError("matplotlib exploded")

    monkeypatch.setattr(figures_mod, "normalized_leaderboard", normalized_leaderboard)

    with pytest.raises(figures_mod.FigureRenderError, match="normalized_leaderboard"):
        figures_mod.render_all(_small_output(), tmp_path)


def test_render_all_tolerates_a_renderer_that_returns_none(
    tmp_path, monkeypatch
) -> None:
    """A renderer whose diagnostics are absent returns None -- an expected skip.

    This must stay distinct from a raise: omitting a ranker is a user choice, not
    a defect, and it must not fail the run.
    """
    from mriforge.core.metrics.meta_evaluation import figures as figures_mod

    monkeypatch.setattr(
        figures_mod, "normalized_leaderboard", lambda _output, _out_dir: None
    )
    written = figures_mod.render_all(_small_output(), tmp_path)
    assert not any(p.name.startswith("fig13") for p in written)
