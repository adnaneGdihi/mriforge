"""Tests for the meta-evaluation simulator's real-asset threading.

Covers the WS1 wiring: ``run_simulator`` attaches per-content acquisition assets
to each ``DegradationSample``, and ``_build_sample_context`` prefers those real
assets over the synthetic birdcage stand-in — falling back to synthesis only for
fields the acquisition did not supply.
"""

from __future__ import annotations

import torch

from mriforge.core.metrics.context import MetricContext
from mriforge.core.metrics.meta_evaluation.simulator import (
    SimulatorConfig,
    _build_sample_context,
    run_simulator,
)


def _noise(clean: torch.Tensor, theta: float, gen: torch.Generator) -> torch.Tensor:
    return clean + theta * torch.randn(clean.shape, generator=gen)


def _config() -> SimulatorConfig:
    return SimulatorConfig(
        families=["gaussian_noise"],
        n_severities_per_family=2,
        operator_library={"gaussian_noise": _noise},
    )


def test_run_simulator_attaches_assets_by_content():
    clean = torch.zeros(1, 1, 8, 8)
    ctx = MetricContext(coil_maps=torch.ones(1, 4, 8, 8), acceleration=1.0)
    samples = run_simulator(
        [("subj0", clean), ("subj1", clean)],
        _config(),
        assets_by_content={"subj0": ctx},
    )
    by_content = {s.content_id: s for s in samples}
    # subj0 samples carry the real assets; subj1 (absent from the map) get None.
    assert all(s.assets is ctx for s in samples if s.content_id == "subj0")
    assert all(s.assets is None for s in samples if s.content_id == "subj1")
    assert by_content["subj0"].assets is ctx


def test_run_simulator_without_assets_is_backward_compatible():
    clean = torch.zeros(1, 1, 8, 8)
    samples = run_simulator([("subj0", clean)], _config())
    assert samples and all(s.assets is None for s in samples)


def test_build_sample_context_prefers_real_coil_maps():
    clean = torch.rand(1, 1, 16, 16)
    real_smaps = torch.ones(1, 4, 16, 16, dtype=torch.complex64)
    real = MetricContext(coil_maps=real_smaps, acceleration=3.0)

    ctx = _build_sample_context(
        clean, ("coil_maps", "y_kspace", "acceleration"), real=real
    )
    # Real coil maps used verbatim (identity), NOT the synthetic birdcage.
    assert ctx.coil_maps is real_smaps
    assert ctx.acceleration == 3.0
    # y_kspace was still synthesised (real assets did not supply it).
    assert ctx.y_kspace is not None


def test_build_sample_context_synthesises_when_no_real_assets():
    clean = torch.rand(1, 1, 16, 16)
    ctx = _build_sample_context(clean, ("coil_maps", "foreground_mask"), real=None)
    assert ctx.coil_maps is not None  # synthetic birdcage stand-in
    assert ctx.foreground_mask is not None


def test_precompute_passes_context_as_keyword_so_nr_metric_computes():
    """Raw registry metrics declare ``*, context=None`` (keyword-only).

    ``precompute_metric_values`` must pass the context as a KEYWORD — a
    positional third arg TypeErrors and is swallowed into NaN, which had been
    silently NaN-ing the entire no-reference battery on the raw-metric path.
    Regression: a context-needing metric (``ndcr``) must return finite values.
    """
    import math

    from mriforge.core.metrics import get_metric
    from mriforge.core.metrics.meta_evaluation.simulator import precompute_metric_values
    from mriforge.infrastructure.physics.asset_preparation import prepare_metric_context

    c, h, w = 4, 16, 16
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
    )
    phantom = torch.exp(-(yy**2 + xx**2) / 0.3).clamp(0, 1)[None, None].float()
    smaps = torch.ones(1, c, h, w, dtype=torch.complex64) / (c**0.5)
    asset = prepare_metric_context(
        coil_images=phantom.to(torch.complex64) * smaps, smaps=smaps, p99=1.0
    )

    samples = run_simulator(
        [("subj_0", phantom.squeeze(0))],
        _config(),
        assets_by_content={"subj_0": asset},
    )
    vals = precompute_metric_values(samples, {"ndcr": get_metric("ndcr")})["ndcr"]
    assert vals and all(math.isfinite(v) for v in vals)
