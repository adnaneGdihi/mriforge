"""Regression tests for the 2026-05-31 strategy-audit findings on
``physics_driven_strategy``.

Three findings were applied (see TODO/_high_triage_full.txt ->
``### physics_driven_strategy``):

1. [high] The Bloch path re-ran ``self.env.generator(input_batch)`` a second
   time (doubled compute, no FP32 AMP guard, bypassed ``forward_kwargs``). The
   fix reuses the parent's stashed ``self._last_prediction`` seam instead.
2. [medium] The Bloch loss silently no-op'd (``else: pass`` / no-else) when the
   batch lacked ``T1``/``T2`` or the prediction was not ``[B, T, 3, H, W]``,
   despite ``lambda_bloch_residual > 0``. The fix raises instead of dropping
   the enabled loss term.
3. [low] A failed ``B0MapSimulator`` import was swallowed to a warning even when
   ``physics.b0_simulation.enabled`` was True. The fix re-raises so the run
   fails fast.

The override methods are exercised in isolation via ``__new__`` + minimal fakes
(the established pattern in test_universal_reconstruction_strategy.py): the
heavy base ``__init__`` is bypassed and only the attributes the tested path
reads are wired.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from mriforge.infrastructure.training.strategies.physics_driven_strategy import (
    PhysicsDrivenTrainingStrategy,
)
from mriforge.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)
from tests.utils.config_block_stub import block_stub


class _CountingGen:
    """Records how many times it is invoked as ``generator(x)``."""

    def __init__(self):
        self.calls = 0

    def __call__(self, x, **kwargs):
        self.calls += 1
        return x


def _physics_config(lambda_bloch: float):
    """Minimal ``config`` exposing ``config.losses.physics.lambda_bloch_residual``."""
    physics = SimpleNamespace(lambda_bloch_residual=lambda_bloch)
    losses = SimpleNamespace(physics=physics)
    return SimpleNamespace(losses=losses)


def _make_bloch_strategy(
    monkeypatch,
    *,
    lambda_bloch: float,
    last_prediction: torch.Tensor | None,
    parent_losses: dict | None = None,
):
    """Build a bare strategy wired only for the Bloch ``_compute_losses_impl`` path.

    The parent ``_compute_losses_impl`` is stubbed so the test exercises only the
    physics-driven override and the seam contract (``self._last_prediction``).
    """
    strat = PhysicsDrivenTrainingStrategy.__new__(PhysicsDrivenTrainingStrategy)
    gen = _CountingGen()
    strat.env = SimpleNamespace(generator=gen)
    strat.device = torch.device("cpu")
    strat.config = _physics_config(lambda_bloch)
    strat._last_prediction = last_prediction

    base = (
        parent_losses
        if parent_losses is not None
        else {
            "g_total_loss": torch.zeros((), requires_grad=True),
        }
    )

    def _fake_parent(self, input_batch, target_batch, epoch, **kwargs):
        # Mirror the real parent: populate the seam, return the loss dict.
        self._last_prediction = last_prediction
        return dict(base)

    monkeypatch.setattr(
        ReconstructionTrainingStrategy,
        "_compute_losses_impl",
        _fake_parent,
        raising=True,
    )
    return strat, gen


# ---------------------------------------------------------------------------
# Finding #1 — reuse the parent prediction seam; no second generator forward.
# ---------------------------------------------------------------------------


class TestNoSecondGeneratorForward:
    def test_reuses_last_prediction_no_extra_generator_call(self, monkeypatch):
        # A valid 5-D Bloch trajectory so the loss path completes.
        m_pred = torch.randn(1, 4, 3, 8, 8)
        strat, gen = _make_bloch_strategy(
            monkeypatch, lambda_bloch=1.0, last_prediction=m_pred
        )
        batch = {
            "T1": torch.rand(1, 1, 8, 8) + 0.5,
            "T2": torch.rand(1, 1, 8, 8) + 0.1,
        }
        x = torch.randn(1, 1, 8, 8)
        out = strat._compute_losses_impl(x, x, epoch=0, batch=batch)
        # The override must NOT invoke the generator a second time: the parent
        # path (stubbed) is the only forward; the Bloch term reads the seam.
        assert gen.calls == 0
        assert "bloch_loss" in out

    def test_raises_when_seam_is_unpopulated(self, monkeypatch):
        strat, gen = _make_bloch_strategy(
            monkeypatch, lambda_bloch=1.0, last_prediction=None
        )
        # Force the parent stub to leave the seam None as well.
        monkeypatch.setattr(
            ReconstructionTrainingStrategy,
            "_compute_losses_impl",
            lambda self, i, t, e, **k: {"g_total_loss": torch.zeros(())},
            raising=True,
        )
        batch = {"T1": torch.rand(1, 1, 8, 8), "T2": torch.rand(1, 1, 8, 8)}
        x = torch.randn(1, 1, 8, 8)
        with pytest.raises(RuntimeError, match="_last_prediction is None"):
            strat._compute_losses_impl(x, x, epoch=0, batch=batch)

    def test_source_no_longer_calls_generator_in_bloch_path(self):
        src = inspect.getsource(PhysicsDrivenTrainingStrategy._compute_losses_impl)
        # The redundant second forward must be gone; we read the seam instead.
        # Grep the exact buggy assignment (``output = self.env.generator(...)``)
        # rather than the bare call — the explanatory comment legitimately names
        # the call it replaced.
        assert "output = self.env.generator(input_batch)" not in src
        assert "output = self._last_prediction" in src


# ---------------------------------------------------------------------------
# Finding #2 — raise (don't silently drop) when the enabled Bloch term cannot
# be computed.
# ---------------------------------------------------------------------------


class TestBlochMisconfigRaises:
    def test_missing_t1_t2_raises(self, monkeypatch):
        m_pred = torch.randn(1, 4, 3, 8, 8)
        strat, _gen = _make_bloch_strategy(
            monkeypatch, lambda_bloch=1.0, last_prediction=m_pred
        )
        x = torch.randn(1, 1, 8, 8)
        # No 'batch' kwarg => full_batch is empty => T1/T2 absent.
        with pytest.raises(RuntimeError, match="missing required"):
            strat._compute_losses_impl(x, x, epoch=0)

    def test_wrong_shape_raises(self, monkeypatch):
        # 4-D prediction: cannot support dM/dt -> must raise, not pass.
        m_pred = torch.randn(1, 1, 8, 8)
        strat, _gen = _make_bloch_strategy(
            monkeypatch, lambda_bloch=1.0, last_prediction=m_pred
        )
        batch = {"T1": torch.rand(1, 1, 8, 8), "T2": torch.rand(1, 1, 8, 8)}
        x = torch.randn(1, 1, 8, 8)
        with pytest.raises(RuntimeError, match=r"shape"):
            strat._compute_losses_impl(x, x, epoch=0, batch=batch)

    def test_wrong_channel_count_raises(self, monkeypatch):
        # 5-D but channel dim != 3 -> not a (Mx,My,Mz) trajectory.
        m_pred = torch.randn(1, 4, 2, 8, 8)
        strat, _gen = _make_bloch_strategy(
            monkeypatch, lambda_bloch=1.0, last_prediction=m_pred
        )
        batch = {"T1": torch.rand(1, 1, 8, 8), "T2": torch.rand(1, 1, 8, 8)}
        x = torch.randn(1, 1, 8, 8)
        with pytest.raises(RuntimeError, match=r"shape"):
            strat._compute_losses_impl(x, x, epoch=0, batch=batch)

    def test_disabled_bloch_does_not_raise(self, monkeypatch):
        # lambda_bloch == 0 => the entire Bloch block is skipped; even with no
        # maps the parent recon losses pass through untouched.
        strat, gen = _make_bloch_strategy(
            monkeypatch, lambda_bloch=0.0, last_prediction=None
        )
        x = torch.randn(1, 1, 8, 8)
        out = strat._compute_losses_impl(x, x, epoch=0)
        assert "bloch_loss" not in out
        assert gen.calls == 0

    def test_source_has_no_silent_pass_in_bloch_path(self):
        src = inspect.getsource(PhysicsDrivenTrainingStrategy._compute_losses_impl)
        # The old silent-drop branches are gone.
        assert "simple pass for now" not in src
        assert "else:\n                    # Warn once?" not in src


# ---------------------------------------------------------------------------
# Finding #3 — B0 simulator import failure must re-raise when enabled.
# ---------------------------------------------------------------------------


class TestB0ImportFailureRaises:
    def _make_setup_strategy(self, *, b0_enabled: bool):
        strat = PhysicsDrivenTrainingStrategy.__new__(PhysicsDrivenTrainingStrategy)
        strat.device = torch.device("cpu")
        b0 = SimpleNamespace(
            enabled=b0_enabled, style="random", probability=1.0, max_hz=None
        )
        physics = SimpleNamespace(b0_simulation=b0, field_strength=3.0)
        # `optimization` is a REQUIRED field on TrainingSettings, so None is a
        # state the schema forbids. The reader guards with
        # `hasattr(config, "optimization")` -- which is True for a None value --
        # and then walks `opt.precision`, so the double, not the guard, was wrong.
        strat.config = SimpleNamespace(
            physics=physics, optimization=block_stub("optimization")
        )
        # Make the parent helpers no-op so we isolate the B0-import branch.
        strat._verify_strategy_config = lambda **kw: None
        strat._log_config_features = lambda *a, **k: None
        strat.logging_service = None
        return strat

    def test_enabled_import_failure_reraises(self, monkeypatch):
        strat = self._make_setup_strategy(b0_enabled=True)

        import builtins

        real_import = builtins.__import__

        def _boom(name, *args, **kwargs):
            if name == "mriforge.infrastructure.physics.field_simulation":
                raise ImportError("simulated missing B0MapSimulator")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _boom)
        with pytest.raises(RuntimeError, match="B0MapSimulator import failed"):
            strat._setup_strategy_specific_components()

    def test_disabled_import_failure_does_not_run(self, monkeypatch):
        # When b0_simulation is disabled the import branch is never reached, so a
        # broken import must NOT cause a failure.
        strat = self._make_setup_strategy(b0_enabled=False)

        import builtins

        real_import = builtins.__import__

        def _boom(name, *args, **kwargs):
            if name == "mriforge.infrastructure.physics.field_simulation":
                raise ImportError("simulated missing B0MapSimulator")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _boom)
        # Should complete without raising; _b0_simulator stays None.
        strat._setup_strategy_specific_components()
        assert strat._b0_simulator is None

    def test_source_reraises_on_import_error(self):
        src = inspect.getsource(
            PhysicsDrivenTrainingStrategy._setup_strategy_specific_components
        )
        assert "except ImportError as exc" in src
        assert "raise RuntimeError" in src
        # The old warn-and-disable text is gone.
        assert "B0 simulation disabled." not in src


# ---------------------------------------------------------------------------
# SPB-001 — exhausted kwarg-strip retry must raise an informative RuntimeError,
# not a bare ``raise`` ("No active exception to re-raise") with ``hr_fakes``
# unbound.
# ---------------------------------------------------------------------------


class _AlwaysRejectGen:
    """Generator forward that rejects whatever kwargs it is given.

    On each call it raises a matched-format TypeError naming a present kwarg
    (so the progressive-strip loop pops it and retries). Once the dict is
    drained, the no-kwargs call raises a matched-format TypeError naming a
    kwarg that is no longer present — the strategy then re-raises it so the
    genuine model incompatibility surfaces (NN#3: never swallowed silently).
    """

    def __init__(self):
        self.calls = 0

    def __call__(self, x, **kwargs):
        self.calls += 1
        if kwargs:
            name = next(iter(kwargs))
            raise TypeError(f"forward() got an unexpected keyword argument '{name}'")
        # Dict drained but the generator is still incompatible: name a kwarg
        # that is NOT in the (now empty) injected set so the strategy re-raises
        # the real error instead of looping forever.
        raise TypeError("forward() got an unexpected keyword argument 'phantom_only'")


class TestKwargStripDoesNotSwallow:
    """SPB-001: the kwarg-strip retry surfaces a real error, never a bare
    ``raise`` ("No active exception to re-raise") with ``hr_fakes`` unbound."""

    def _make_strategy(self, gen):
        strat = PhysicsDrivenTrainingStrategy.__new__(PhysicsDrivenTrainingStrategy)
        strat.env = SimpleNamespace(generator=gen)
        strat.device = torch.device("cpu")
        # `__new__` skips `__init__`, so every attribute `_generate_predictions`
        # reads must be mirrored here by hand. `_b0_simulator` is initialised to
        # None in `__init__` and read at `physics_driven_strategy.py:~
        # (self._b0_simulator is not None and "b0_map" not in forward_kwargs)`;
        # omitting it raised AttributeError before the code under test ever ran,
        # leaving both tests below red on `dev` (issue #189).
        strat._b0_simulator = None
        return strat

    def test_unstrippable_typeerror_surfaces_not_no_active_exception(self):
        # The generator rejects every kwarg, then the drained call raises a
        # TypeError naming a kwarg outside the injected set: the strategy must
        # re-raise THAT error (the real model bug), never the uninformative
        # ``RuntimeError: No active exception to re-raise``.
        gen = _AlwaysRejectGen()
        strat = self._make_strategy(gen)
        lr = torch.randn(1, 1, 8, 8)
        forward_kwargs = {"target": torch.randn(1, 1, 8, 8), "mask": torch.ones(1, 8)}
        with pytest.raises(TypeError) as exc_info:
            strat._generate_predictions(lr, forward_kwargs, {})
        assert "No active exception to re-raise" not in str(exc_info.value)
        assert "phantom_only" in str(exc_info.value)
        # All present kwargs were stripped, then the drained call surfaced.
        assert gen.calls == 3

    def test_non_kwarg_typeerror_propagates_immediately(self):
        # A TypeError that is NOT a "unexpected keyword argument" message is a
        # genuine internal bug and must propagate on the first call (no strip).
        class _GenInternalBug:
            def __init__(self):
                self.calls = 0

            def __call__(self, x, **kwargs):
                self.calls += 1
                raise TypeError("internal: bad dtype in forward")

        gen = _GenInternalBug()
        strat = self._make_strategy(gen)
        lr = torch.randn(1, 1, 8, 8)
        with pytest.raises(TypeError, match="internal: bad dtype"):
            strat._generate_predictions(lr, {"target": torch.zeros(1)}, {})
        assert gen.calls == 1

    def test_source_uses_runtimeerror_not_bare_raise_in_else(self):
        src = inspect.getsource(PhysicsDrivenTrainingStrategy._generate_predictions)
        # The exhausted-attempts ``else`` clause now raises an informative
        # RuntimeError instead of a bare ``raise`` (which would surface as
        # "No active exception to re-raise" with ``hr_fakes`` unbound).
        assert "raise RuntimeError(" in src
        assert "exhausted" in src


# ---------------------------------------------------------------------------
# SPB-002 — the kwarg-name regex is hoisted to module scope (compiled once),
# not recompiled inside the per-step ``_generate_predictions``.
# ---------------------------------------------------------------------------


class TestKwargPatternHoisted:
    def test_module_level_pattern_exists(self):
        from mriforge.infrastructure.training.strategies import (
            physics_driven_strategy as mod,
        )

        assert hasattr(mod, "_KWARG_PAT")

    def test_pattern_matches_typeerror_message(self):
        from mriforge.infrastructure.training.strategies.physics_driven_strategy import (
            _KWARG_PAT,
        )

        msg = "forward() got an unexpected keyword argument 'query_coords'"
        m = _KWARG_PAT.search(msg)
        assert m is not None
        assert m.group(1) == "query_coords"

    def test_pattern_handles_double_quotes(self):
        from mriforge.infrastructure.training.strategies.physics_driven_strategy import (
            _KWARG_PAT,
        )

        msg = 'forward() got an unexpected keyword argument "target"'
        m = _KWARG_PAT.search(msg)
        assert m is not None
        assert m.group(1) == "target"

    def test_source_no_longer_compiles_inside_method(self):
        src = inspect.getsource(PhysicsDrivenTrainingStrategy._generate_predictions)
        # The per-call ``import re`` + ``_re.compile`` are gone; the method
        # references the hoisted module-level pattern instead.
        assert "import re as _re" not in src
        assert "_re.compile" not in src
        assert "_KWARG_PAT.search" in src
