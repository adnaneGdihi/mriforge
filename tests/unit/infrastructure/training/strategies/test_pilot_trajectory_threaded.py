"""Regression: PILOT must CONSUME its learned trajectory in the recon forward.

2026-05-31 audit bug — ``PILOTStrategy`` added ``self._trajectory`` to
``opt_g`` and added an ``L2``-style feasibility penalty on it, but never
threaded it into the reconstruction forward. The parent
(``ReconstructionMixin._prepare_batch_context_reconstruction``) sources the
trajectory from the batch dict (``kwargs["batch"]["trajectory"]``) into
``batch_context["trajectory"]``, which the NUFFT-adjoint recon path then
consumes. Because PILOT never injected it, the trajectory received only the
feasibility gradient and *no data gradient* — so it could never influence the
output and the paradigm did not actually function.

Fix: inject ``self._trajectory`` into the batch dict forwarded to ``super()``
(mirroring ``GIRFAwareStrategy``'s waveform→trajectory injection). These tests
drive ``_compute_losses_impl`` with a stubbed parent that captures the kwargs
it receives, and assert the learned trajectory tensor reaches the parent's
batch.
"""

from __future__ import annotations

import types

import torch

from spectramr.infrastructure.training.strategies.pilot_strategy import PILOTStrategy


def _make_pilot(trajectory, *, codesign_enabled=False):
    """Construct PILOTStrategy bypassing __init__ (no DI/env build needed).

    Only the attributes ``_compute_losses_impl`` touches are set:
    ``_trajectory`` and ``_codesign`` (the latter gates the feasibility term).
    """
    s = object.__new__(PILOTStrategy)
    s.env = types.SimpleNamespace(generator=torch.nn.Identity())
    s._trajectory = trajectory
    s._codesign = types.SimpleNamespace(enabled=codesign_enabled)
    return s


def _patch_parent_capture(monkeypatch, s, captured):
    """Stub the immediate-parent ``_compute_losses_impl`` so ``super()`` returns
    a sentinel dict and records the kwargs (incl. ``batch``) it was called with.
    """
    parent = type(s).__mro__[1]

    def _stub(self, input_batch=None, target_batch=None, epoch=0, **kw):
        captured.update(kw)
        return {"loss_total": torch.zeros(())}

    monkeypatch.setattr(parent, "_compute_losses_impl", _stub, raising=False)


_LR = torch.randn(1, 2, 4, 4)
_HR = torch.randn(1, 2, 4, 4)


class TestPILOTTrajectoryThreaded:
    def test_trajectory_injected_into_parent_batch(self, monkeypatch):
        """KEY property: the learned trajectory tensor reaches the parent forward
        via ``kwargs["batch"]["trajectory"]`` so the recon loss flows gradient
        back into it."""
        traj = torch.nn.Parameter(torch.randn(8, 2))
        s = _make_pilot(traj)
        captured: dict = {}
        _patch_parent_capture(monkeypatch, s, captured)

        out = s._compute_losses_impl(
            input_batch=_LR, target_batch=_HR, epoch=0, batch={"x": _LR}
        )

        assert isinstance(out, dict)
        assert "batch" in captured, "PILOT did not forward a batch dict to super()"
        assert captured["batch"]["trajectory"] is s._trajectory, (
            "learned trajectory must be the SAME tensor object the parent recon "
            "forward consumes, so the data/recon loss gradient reaches it"
        )

    def test_existing_batch_keys_preserved(self, monkeypatch):
        """Injection must not clobber other batch entries (copy-then-set)."""
        traj = torch.nn.Parameter(torch.randn(8, 2))
        s = _make_pilot(traj)
        captured: dict = {}
        _patch_parent_capture(monkeypatch, s, captured)

        s._compute_losses_impl(
            input_batch=_LR,
            target_batch=_HR,
            epoch=0,
            batch={"measured_kspace": _LR, "mask": _HR},
        )

        forwarded = captured["batch"]
        assert forwarded["trajectory"] is s._trajectory
        assert forwarded["measured_kspace"] is _LR
        assert forwarded["mask"] is _HR

    def test_no_batch_dict_still_forwards_named(self, monkeypatch):
        """No batch dict in kwargs → parent still invoked via the named
        convention without crashing (graceful degradation)."""
        traj = torch.nn.Parameter(torch.randn(8, 2))
        s = _make_pilot(traj)
        captured: dict = {}
        _patch_parent_capture(monkeypatch, s, captured)

        out = s._compute_losses_impl(input_batch=_LR, target_batch=_HR, epoch=0)
        assert isinstance(out, dict)
        # Without a batch dict to thread through, no trajectory injection occurs.
        assert "batch" not in captured


# ---------------------------------------------------------------------------
# Regression: model_kwargs.n_shots / samples_per_shot are READ (pitfall #15)
# ---------------------------------------------------------------------------
#
# 2026-05-31 audit bug — ``__init__`` used ``getattr(self.config.model.
# model_kwargs, "n_shots", 16)``. ``model_kwargs`` is a ``dict[str, Any]``, so
# getattr never finds the key as an attribute and always returns the 16/256
# defaults — any YAML override of these knobs was a silent no-op. The fix uses
# dict ``.get()``, validates ``> 0`` (raise, no silent fallback), and stamps
# the resolved values onto ``self._n_shots`` / ``self._samples_per_shot``.
#
# These tests drive the *real* ``__init__`` body that reads the knobs, by
# constructing the strategy bypassing ``object.__init__``-level DI: we stub the
# parent chain's ``__init__`` so only ``PILOTStrategy.__init__`` runs, with a
# hand-built config exposing ``model.model_kwargs`` and ``acquisition.codesign``.


def _ns(**kw):
    return types.SimpleNamespace(**kw)


def _make_pilot_via_init(monkeypatch, *, model_kwargs, codesign_enabled=False):
    """Run the real ``PILOTStrategy.__init__`` body with a stub parent init.

    Only the attributes ``__init__`` reads are supplied:
    ``config.model.model_kwargs``, ``config.acquisition.codesign``, ``device``,
    and ``env.opt_g`` (None → no param-group registration).
    """
    parent = PILOTStrategy.__mro__[1]
    constraints = _ns(
        gradient_max_mT_per_m=40.0,
        slew_rate_max_T_per_m_per_s=200.0,
        gradient_dwell_us=10.0,
    )
    codesign = _ns(
        enabled=codesign_enabled,
        feasibility_lambda=0.01,
        constraints=constraints,
    )
    config = _ns(
        model=_ns(model_kwargs=model_kwargs),
        acquisition=_ns(codesign=codesign),
    )

    def _stub_parent_init(self, *a, **k):
        # The base __init__ normally wires config/device/env; supply them here.
        self.config = config
        self.device = torch.device("cpu")
        self.env = _ns(opt_g=None)

    monkeypatch.setattr(parent, "__init__", _stub_parent_init, raising=False)
    return PILOTStrategy()


class TestPILOTKnobsAreRead:
    def test_n_shots_and_samples_per_shot_override_is_read(self, monkeypatch):
        """YAML override of model_kwargs.{n_shots, samples_per_shot} must drive
        the trajectory size. Pre-fix getattr() ignored these (4096 = 16*256);
        post-fix the trajectory has 8*64 = 512 points."""
        s = _make_pilot_via_init(
            monkeypatch, model_kwargs={"n_shots": 8, "samples_per_shot": 64}
        )
        assert s._n_shots == 8
        assert s._samples_per_shot == 64
        # The flat trajectory has n_shots * samples_per_shot rows of (kx, ky).
        assert s._trajectory.shape[0] == 512, (
            "n_shots/samples_per_shot knobs were not read — the trajectory still "
            "uses the 16/256 defaults (4096 points)"
        )
        assert s._trajectory.shape == (512, 2)

    def test_defaults_when_kwargs_absent(self, monkeypatch):
        """Empty model_kwargs → documented 16/256 defaults (4096 points)."""
        s = _make_pilot_via_init(monkeypatch, model_kwargs={})
        assert s._n_shots == 16
        assert s._samples_per_shot == 256
        assert s._trajectory.shape[0] == 16 * 256

    def test_non_positive_n_shots_raises(self, monkeypatch):
        """No silent fallback (pitfall #9): n_shots <= 0 must raise."""
        import pytest

        with pytest.raises(ValueError, match="n_shots must be > 0"):
            _make_pilot_via_init(
                monkeypatch, model_kwargs={"n_shots": 0, "samples_per_shot": 64}
            )

    def test_non_positive_samples_per_shot_raises(self, monkeypatch):
        """No silent fallback (pitfall #9): samples_per_shot <= 0 must raise."""
        import pytest

        with pytest.raises(ValueError, match="samples_per_shot must be > 0"):
            _make_pilot_via_init(
                monkeypatch, model_kwargs={"n_shots": 4, "samples_per_shot": -3}
            )


# ---------------------------------------------------------------------------
# Regression: feasibility penalty must not difference across shot boundaries
# ---------------------------------------------------------------------------
#
# 2026-05-31 audit bug — the flat ``[n_shots*samples_per_shot, 2]`` trajectory
# was differenced with ``torch.diff(..., dim=0)``, so shot i's last sample was
# differenced against shot i+1's first sample, injecting a spurious slew /
# gradient spike at every shot boundary. The fix reshapes to
# ``[n_shots, samples_per_shot, 2]`` and differences along the per-shot axis.


def _make_pilot_for_penalty(traj, n_shots, samples_per_shot, *, feas_lambda=1.0):
    """Build a PILOTStrategy (bypassing __init__) wired for _feasibility_penalty.

    Constraints are set very low so that the inter-shot jump, if it leaked into
    the diff, would dominate g_overshoot / s_overshoot — making the boundary
    bug loud and easy to detect.
    """
    s = object.__new__(PILOTStrategy)
    s.device = torch.device("cpu")
    s._trajectory = traj
    s._n_shots = n_shots
    s._samples_per_shot = samples_per_shot
    constraints = _ns(
        gradient_max_mT_per_m=0.0001,  # ~0 mT/m → any gradient overshoots
        slew_rate_max_T_per_m_per_s=0.0001,
        gradient_dwell_us=10.0,
    )
    s._codesign = _ns(
        enabled=True,
        feasibility_lambda=feas_lambda,
        constraints=constraints,
    )
    return s


class TestPILOTFeasibilityShotBoundary:
    def test_reordering_shots_does_not_change_penalty(self):
        """The feasibility penalty must be invariant to the ORDER of shots.

        Construction avoids float32 catastrophic cancellation: the two
        trajectories are built from the SAME per-shot sample tensors, only the
        concatenation order changes. Reordering leaves every within-shot diff
        BIT-IDENTICAL but completely changes the inter-shot boundary pair. A
        correct per-shot penalty (post-reshape) is therefore BIT-EXACT equal;
        the pre-fix flat ``dim=0`` diff straddles the large inter-shot gap and
        diverges wildly.
        """
        spp = 16
        torch.manual_seed(0)
        # Two shots with tiny within-shot motion (so any boundary leak dominates)
        # but endpoints FAR apart (~10 units) so a straddling diff is enormous.
        shot_a = torch.randn(spp, 2) * 0.01 - 5.0
        shot_b = torch.randn(spp, 2) * 0.01 + 5.0

        flat_ab = torch.cat([shot_a, shot_b], dim=0)
        flat_ba = torch.cat([shot_b, shot_a], dim=0)

        s_ab = _make_pilot_for_penalty(torch.nn.Parameter(flat_ab), 2, spp)
        s_ba = _make_pilot_for_penalty(torch.nn.Parameter(flat_ba), 2, spp)
        penalty_ab = s_ab._feasibility_penalty().item()
        penalty_ba = s_ba._feasibility_penalty().item()

        # Bit-exact: within-shot diffs are identical between orderings, and the
        # only difference (the ignored inter-shot pair) must contribute nothing.
        assert penalty_ab == penalty_ba, (
            "feasibility penalty changed when shots were reordered — the diff is "
            "still straddling the shot boundary (dim=0 flat bug). "
            f"[A,B]={penalty_ab} vs [B,A]={penalty_ba}"
        )

    def test_single_shot_value_independent_of_split(self):
        """A trajectory's penalty must not depend on whether we DECLARE it as one
        shot or split the SAME flat samples into several shots — except that the
        multi-shot version drops the (physically meaningless) boundary diffs.

        We assert the multi-shot penalty is STRICTLY LESS THAN the single-shot
        penalty when the inter-shot gaps are large: the single-shot (flat)
        computation, the pre-fix behaviour, includes those huge boundary diffs;
        the correct multi-shot computation excludes them.
        """
        spp = 8
        torch.manual_seed(3)
        shot0 = torch.randn(spp, 2) * 0.01
        shot1 = torch.randn(spp, 2) * 0.01 + 50.0  # huge gap at the boundary
        flat = torch.cat([shot0, shot1], dim=0)

        # Declared as ONE flat shot of 2*spp samples → boundary diff IS included
        # (this is exactly the pre-fix flat-diff behaviour).
        s_flat = _make_pilot_for_penalty(
            torch.nn.Parameter(flat.clone()), 1, 2 * spp
        )
        penalty_flat = s_flat._feasibility_penalty().item()

        # Declared as TWO shots → the 50-unit boundary diff is excluded.
        s_split = _make_pilot_for_penalty(torch.nn.Parameter(flat.clone()), 2, spp)
        penalty_split = s_split._feasibility_penalty().item()

        assert penalty_split < penalty_flat, (
            "splitting into shots did not drop the inter-shot boundary diff — "
            f"split={penalty_split} should be < flat={penalty_flat}"
        )

    def test_penalty_shape_reduces_correctly(self):
        """Sanity: penalty is a finite scalar for a multi-shot trajectory."""
        n_shots, spp = 3, 16
        torch.manual_seed(1)
        traj = torch.nn.Parameter(torch.randn(n_shots * spp, 2) * 0.01)
        s = _make_pilot_for_penalty(traj, n_shots, spp)
        out = s._feasibility_penalty()
        assert out.ndim == 0
        assert torch.isfinite(out)
