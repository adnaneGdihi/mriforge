"""MRS losses — the self-supervised fit that makes mri_spectroscopy trainable.

The headline test is ``test_recovers_known_resonances``: it optimises the real
registered loss back to known parameters on analytic physics, which is a proof of
the objective rather than a smoke check. ``test_the_residual_tells_the_truth_...``
pins the property that distinguishes this fit from the perfusion one.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.config.schemas.enums import Regime
from spectramr.infrastructure.physics.signal_models.spectroscopy import lorentzian_fid
from spectramr.models.losses.spectroscopy_losses import (
    MRSFIDResidualLoss,
    MRSPriorKnowledgeLoss,
)

_T, _DWELL = 512, 5e-4
_TRUE_A = [1.0, 0.6, 0.35]
_TRUE_F = [120.0, -80.0, 260.0]
_TRUE_LW = [5.0, 8.0, 6.0]
_TRUE_PH = [0.0, 0.3, -0.2]


def _time_axis() -> torch.Tensor:
    return torch.arange(_T, dtype=torch.float32) * _DWELL


def _measured() -> torch.Tensor:
    return lorentzian_fid(
        _time_axis(),
        torch.tensor(_TRUE_A),
        torch.tensor(_TRUE_F),
        torch.tensor(_TRUE_LW),
        torch.tensor(_TRUE_PH),
    )


def _fit(freq_init: list[float], steps: int = 4000) -> tuple[float, torch.Tensor]:
    """Optimise the registered loss back to the resonance parameters.

    Frequency gets its own much larger lr: it is measured in Hz (~100) while
    amplitudes are ~1, so one shared step size cannot move both.
    """
    torch.manual_seed(0)
    measured, t = _measured(), _time_axis()
    a = torch.full((3,), 0.5, requires_grad=True)
    f = torch.tensor(freq_init, requires_grad=True)
    lw = torch.full((3,), 10.0, requires_grad=True)
    ph = torch.zeros(3, requires_grad=True)
    opt = torch.optim.Adam(
        [{"params": [a, lw, ph], "lr": 2e-2}, {"params": [f], "lr": 2.0}]
    )
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.999)
    loss_fn = MRSFIDResidualLoss()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(
            t_s=t,
            amplitudes=a,
            frequencies_hz=f,
            linewidths_hz=lw,
            phases_rad=ph,
            measured_fid=measured,
        )
        loss.backward()
        opt.step()
        sched.step()
    return float(loss.detach()), torch.stack([a, f, lw, ph]).detach()


class TestMRSFIDResidual:
    def test_zero_at_the_true_parameters(self) -> None:
        assert float(
            MRSFIDResidualLoss()(
                t_s=_time_axis(),
                amplitudes=torch.tensor(_TRUE_A),
                frequencies_hz=torch.tensor(_TRUE_F),
                linewidths_hz=torch.tensor(_TRUE_LW),
                phases_rad=torch.tensor(_TRUE_PH),
                measured_fid=_measured(),
            )
        ) == pytest.approx(0.0, abs=1e-7)

    def test_recovers_known_resonances(self) -> None:
        """The OBJECTIVE recovers known resonances, on analytic physics.

        MRS quantification IS this fit: the amplitudes are the concentrations.
        It is self-supervised — the measured FID supplies its own supervision —
        which is what makes the regime trainable on real MRS data, since real
        acquisitions carry no concentration labels. Verifiable with no data at all.

        SCOPE: this optimises free tensors against the registered loss; it does
        NOT call the strategy, so it verifies the loss + signal model, not the
        wiring. The wiring is covered by
        test_mrs_quantification_strategy.py::test_the_registry_resolved_model_is_the_one_actually_called.
        """
        residual, fitted = _fit([100.0, -100.0, 250.0])
        assert residual < 1e-3
        torch.testing.assert_close(
            fitted[0], torch.tensor(_TRUE_A), rtol=0.05, atol=0.02
        )
        torch.testing.assert_close(
            fitted[1], torch.tensor(_TRUE_F), rtol=0.02, atol=1.0
        )
        torch.testing.assert_close(
            fitted[2], torch.tensor(_TRUE_LW), rtol=0.05, atol=0.3
        )

    def test_the_residual_tells_the_truth_unlike_the_perfusion_one(self) -> None:
        """The property that decides how to READ a converged MRS fit.

        This objective is identifiable but NOT convex: the frequency capture
        range is roughly +/-30 Hz, and past it the optimiser lands in a local
        minimum. Crucially, that minimum is **three orders of magnitude worse**,
        so a bad fit announces itself.

        Contrast `tofts_residual`, where a ~1e-4 residual coexists happily with a
        NEGATIVE vp — there the data cannot identify the parameter, the residual
        LIES, and a constraint is the answer. Here the residual is honest and the
        answer is initialisation, which is exactly why AMARES supplies prior
        knowledge of metabolite positions. Do not "fix" a bad frequency fit with
        a constraint.
        """
        good_residual, _ = _fit([150.0, -50.0, 290.0])  # off by 30 Hz — recovers
        bad_residual, bad = _fit([220.0, 20.0, 360.0])  # off by 100 Hz — does not

        assert good_residual < 1e-3
        assert bad_residual > 100 * good_residual, (
            "a wrong MRS fit must be VISIBLE in the residual — that is what makes "
            "it different from the perfusion vp trap"
        )
        assert not torch.allclose(
            bad[1].sort().values, torch.tensor([-80.0, 120.0, 260.0]), atol=2.0
        )

    def test_a_shape_mismatch_raises_rather_than_broadcasting(self) -> None:
        """[3] params against a [4, T] FID would broadcast to a plausible scalar."""
        with pytest.raises(ValueError, match="does not match"):
            MRSFIDResidualLoss()(
                t_s=_time_axis(),
                amplitudes=torch.tensor(_TRUE_A),
                frequencies_hz=torch.tensor(_TRUE_F),
                linewidths_hz=torch.tensor(_TRUE_LW),
                phases_rad=torch.tensor(_TRUE_PH),
                measured_fid=torch.zeros(4, _T, dtype=torch.complex64),
            )

    def test_the_forward_model_is_injectable_so_the_config_knob_can_dispatch(
        self,
    ) -> None:
        """The strategy passes the model it resolved from the SignalModelRegistry.

        Without this seam the key `data.spectroscopy.signal_model` would be
        re-hardcoded here and the knob would be a decoy (#15).
        """
        calls = {"n": 0}

        def _sentinel(t_s, a, f, lw, ph):
            calls["n"] += 1
            return lorentzian_fid(t_s, a, f, lw, ph)

        MRSFIDResidualLoss(forward_model=_sentinel)(
            t_s=_time_axis(),
            amplitudes=torch.tensor(_TRUE_A),
            frequencies_hz=torch.tensor(_TRUE_F),
            linewidths_hz=torch.tensor(_TRUE_LW),
            phases_rad=torch.tensor(_TRUE_PH),
            measured_fid=_measured(),
        )
        assert calls["n"] == 1

    def test_phase_is_constrained_because_the_residual_is_complex(self) -> None:
        """A magnitude residual would leave phase free while converging nicely.

        Phase is a fitted parameter, so the residual has to see it. This pins
        that a phase error actually costs something.
        """
        common = {
            "t_s": _time_axis(),
            "amplitudes": torch.tensor(_TRUE_A),
            "frequencies_hz": torch.tensor(_TRUE_F),
            "linewidths_hz": torch.tensor(_TRUE_LW),
            "measured_fid": _measured(),
        }
        right = float(MRSFIDResidualLoss()(phases_rad=torch.tensor(_TRUE_PH), **common))
        wrong = float(
            MRSFIDResidualLoss()(phases_rad=torch.tensor(_TRUE_PH) + 1.0, **common)
        )
        assert wrong > 1e-3
        assert wrong > 100 * max(right, 1e-9)


class TestMRSPriorKnowledge:
    def test_physical_parameters_cost_nothing(self) -> None:
        loss = MRSPriorKnowledgeLoss()
        value = loss(
            amplitudes=torch.tensor([1.0, 0.5]),
            linewidths_hz=torch.tensor([5.0, 8.0]),
        )
        assert float(value) == pytest.approx(0.0, abs=1e-9)

    def test_negative_amplitudes_and_out_of_range_linewidths_are_penalised(
        self,
    ) -> None:
        loss = MRSPriorKnowledgeLoss(min_linewidth_hz=1.0, max_linewidth_hz=50.0)
        assert float(loss(torch.tensor([-1.0]), torch.tensor([5.0]))) > 0
        assert float(loss(torch.tensor([1.0]), torch.tensor([0.1]))) > 0  # too narrow
        assert float(loss(torch.tensor([1.0]), torch.tensor([500.0]))) > 0  # too broad

    def test_the_amplitude_term_is_dead_under_softplus(self) -> None:
        """The trap this loss shares with the perfusion box loss.

        Under the strategy's default parameter_activation="softplus" the
        amplitudes reaching this loss are already positive, so clamp(-a) is
        identically zero WITH ZERO GRADIENT. A healthy prior-knowledge loss is
        therefore NOT evidence that non-negativity is being learned — softplus
        did it structurally. Only the linewidth bounds are live.
        """
        loss = MRSPriorKnowledgeLoss()
        raw = torch.tensor([-3.0, -1.0, 2.0], requires_grad=True)
        activated = torch.nn.functional.softplus(raw)
        value = loss(amplitudes=activated, linewidths_hz=torch.tensor([5.0, 5.0, 5.0]))
        assert float(value) == pytest.approx(0.0, abs=1e-9)
        value.backward()
        assert float(raw.grad.abs().sum()) == pytest.approx(0.0, abs=1e-12), (
            "the amplitude term must have NO gradient under softplus — if it "
            "does, this test's premise is wrong"
        )

    def test_inverted_bounds_raise(self) -> None:
        with pytest.raises(ValueError, match="min_linewidth_hz"):
            MRSPriorKnowledgeLoss(min_linewidth_hz=50.0, max_linewidth_hz=1.0)


def test_registered_and_tagged_for_spectroscopy() -> None:
    """They back mri_spectroscopy's LIVE claim; a lost tag must fail here."""
    from spectramr.models.losses.registry import LossRegistry

    for name in ("mrs_fid_residual", "mrs_prior_knowledge"):
        assert LossRegistry._loss_domains[name]["workflows"] == frozenset(
            {Regime.SPECTROSCOPIC}
        )


def test_selectable_from_yaml_via_the_loss_weight_ssot() -> None:
    """Registered is not enough — it must be reachable from a config."""
    from spectramr.models.losses.weights import _schema_defaults

    defaults = _schema_defaults()
    for name in ("mrs_fid_residual", "mrs_prior_knowledge"):
        assert name in defaults
        section, value = defaults[name]
        assert section == "physics"
        assert value == 0.0  # 0.0 = "not requested"
