"""Regression tests for the 2026-06 ``twin_dps_strategy`` audit fixes.

Two BROKEN_STRATEGY_ONLY findings from the 2026-05-31 strategy audit are
covered here (both are entirely in-file in
``infrastructure/training/strategies/twin_dps_strategy.py``):

F1 — False docstring claim of a nonexistent ``_sampling_step`` override.
     The parent ``DiffusionTrainingStrategy`` exposes no per-step
     ``_sampling_step`` hook; the module docstring used to claim this class
     overrides it. The corrected docstring must not assert that override, and
     no such method may exist on the class or its parent.

F2 — ``lambda_y`` knob read+stamped but never consumed (silent no-op,
     CLAUDE.md pitfall #15). ``compute_guidance_gradient`` only consumes
     ``lambda_M``; the k-space DPS term scaled by ``lambda_y`` is owned by the
     inherited / caller-side sampler. The fix drops the dead ``self.lambda_y``
     mirror attribute from ``__init__`` and proves the helper is invariant to
     ``lambda_y`` (it never reads it).

The helper is self-contained, so — mirroring
``tests/unit/training/test_twin_dps_guidance_gradient.py`` — we bypass the
heavy ``DiffusionTrainingStrategy.__init__`` via ``__new__`` and set only the
attributes the tested path touches.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from mriforge.infrastructure.training.strategies import (  # noqa: E402
    twin_dps_strategy as tds_mod,
)
from mriforge.infrastructure.training.strategies.diffusion import (  # noqa: E402
    DiffusionTrainingStrategy,
)
from mriforge.infrastructure.training.strategies.twin_dps_strategy import (  # noqa: E402
    TwinLikelihoodDPSStrategy,
)
from mriforge.models.losses.phase_stego_score import PhaseStegoScoreLoss  # noqa: E402


@pytest.fixture
def marker_template(tmp_path: Path) -> Path:
    """16x16 Fourier-domain marker template for a small probe image."""
    torch.manual_seed(0)
    template = torch.zeros(1, 1, 16, 16, dtype=torch.complex64)
    coords = [(2, 3), (5, 7), (9, 4), (12, 11)]
    for (yi, xi), phase in zip(coords, [0.1, 0.7, 1.3, 2.0]):
        template[0, 0, yi, xi] = torch.complex(
            torch.cos(torch.tensor(phase)), torch.sin(torch.tensor(phase))
        )
    path = tmp_path / "marker.pt"
    torch.save(template, path)
    return path


def _stub_without_lambda_y(
    marker_template: Path, lambda_M: float = 1.0
) -> TwinLikelihoodDPSStrategy:
    """A TwinDPS stub that deliberately never sets ``lambda_y``.

    If ``compute_guidance_gradient`` (or any path it touches) read
    ``self.lambda_y`` it would ``AttributeError`` here — proving the helper is
    invariant to the dropped knob.
    """
    strat = TwinLikelihoodDPSStrategy.__new__(TwinLikelihoodDPSStrategy)
    strat.lambda_M = lambda_M
    strat.sigma_M = 0.05
    strat.phase_stego_basis = "fourier"
    strat.guidance_callback_interval = 1
    strat._marker_template_path = str(marker_template)
    strat._marker_template = None
    strat._marker_score_loss = PhaseStegoScoreLoss(
        sigma_M=strat.sigma_M, basis=strat.phase_stego_basis
    )
    return strat


def _identity_tweedie(x_t: torch.Tensor) -> torch.Tensor:
    return x_t


# --------------------------------------------------------------------------- #
# F1 — no false `_sampling_step` override claim, and no such method exists.
# --------------------------------------------------------------------------- #
class TestNoSamplingStepOverride:
    def test_module_docstring_does_not_claim_sampling_step_override(self) -> None:
        doc = tds_mod.__doc__ or ""
        # The corrected docstring must not assert that this class *overrides*
        # `_sampling_step`. It may still mention the absence of the hook.
        assert "overrides :meth:`DiffusionTrainingStrategy._sampling_step`" not in doc
        assert "exposes no per-step" in doc

    def test_class_defines_no_sampling_step_method(self) -> None:
        # Neither the strategy nor its parent has a `_sampling_step` interface;
        # the original docstring advertised an override that silently no-ops.
        assert not hasattr(TwinLikelihoodDPSStrategy, "_sampling_step")
        assert not hasattr(DiffusionTrainingStrategy, "_sampling_step")


# --------------------------------------------------------------------------- #
# F2 — lambda_y is not consumed and not mirrored as a dead attribute.
# --------------------------------------------------------------------------- #
class TestLambdaYNotConsumed:
    def test_init_source_does_not_assign_self_lambda_y(self) -> None:
        # The dead `self.lambda_y = float(td.lambda_y)` mirror must be gone:
        # `lambda_y` is validated by the schema/audit, not stored on `self`.
        src = inspect.getsource(TwinLikelihoodDPSStrategy.__init__)
        assert "self.lambda_y" not in src
        # lambda_M IS consumed and therefore still read+stored.
        assert "self.lambda_M = float(td.lambda_M)" in src

    def test_guidance_gradient_helper_never_reads_lambda_y(self) -> None:
        # The helper that owns the marker term must not reference lambda_y.
        src = inspect.getsource(TwinLikelihoodDPSStrategy.compute_guidance_gradient)
        assert "lambda_y" not in src
        assert "self.lambda_M" in src

    def test_guidance_gradient_runs_without_lambda_y_set(
        self, marker_template: Path
    ) -> None:
        # A stub with no `lambda_y` attribute must still produce a finite
        # gradient — i.e. the helper does not read the dropped knob.
        strat = _stub_without_lambda_y(marker_template)
        assert not hasattr(strat, "lambda_y")
        x_t = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
        grad = strat.compute_guidance_gradient(x_t, _identity_tweedie)
        assert grad.shape == x_t.shape
        assert torch.isfinite(grad.abs()).all()

    def test_gradient_invariant_to_lambda_y_but_scales_with_lambda_M(
        self, marker_template: Path
    ) -> None:
        # The gradient depends ONLY on lambda_M (linear), never on lambda_y.
        torch.manual_seed(7)
        x_t = torch.randn(1, 1, 16, 16, dtype=torch.complex64)

        g1 = _stub_without_lambda_y(marker_template, lambda_M=1.0).compute_guidance_gradient(
            x_t, _identity_tweedie
        )
        g2 = _stub_without_lambda_y(marker_template, lambda_M=2.0).compute_guidance_gradient(
            x_t, _identity_tweedie
        )
        assert torch.allclose(g2, 2.0 * g1, atol=1e-5)

        # Setting an arbitrary lambda_y on the instance must not change output.
        strat = _stub_without_lambda_y(marker_template, lambda_M=1.0)
        strat.lambda_y = 99.0  # type: ignore[attr-defined]  # ignored by the helper
        g_with = strat.compute_guidance_gradient(x_t, _identity_tweedie)
        assert torch.allclose(g_with, g1, atol=1e-6)
