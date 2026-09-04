"""Tests for MRSQuantificationStrategy — the trainable backing for mri_spectroscopy.

There is no MRS data on this cluster, but the Lorentzian model is analytic, so the
strategy's real claims ARE testable: that the knob dispatches, that the activated
parameters are the ones both fitted and graded, and that the batch contract holds.
The objective itself is proved in test_spectroscopy_losses.py.
"""

from __future__ import annotations

import types

import pytest
import torch

from spectramr.infrastructure.physics.signal_models.registry import get_signal_model
from spectramr.infrastructure.physics.signal_models.spectroscopy import lorentzian_fid
from spectramr.infrastructure.training.strategies.mrs_quantification_strategy import (
    MRSQuantificationStrategy,
)

_T = 32
_M = 2
_SIZE = 4
_DWELL = 5e-4


def _time_axis() -> torch.Tensor:
    return torch.arange(_T, dtype=torch.float32) * _DWELL


class _ParamNet(torch.nn.Module):
    """[B, 2T, H, W] FID -> [B, 4M, H, W] raw resonance params. No terminal tanh."""

    def __init__(self) -> None:
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(2 * _T, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 4 * _M, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def _strategy(**overrides: object) -> MRSQuantificationStrategy:
    strat = object.__new__(MRSQuantificationStrategy)
    strat.env = types.SimpleNamespace(generator=_ParamNet(), losses={})
    strat.config = types.SimpleNamespace(
        losses=types.SimpleNamespace(
            image_losses=[], kspace_losses=[], complex_losses=[]
        )
    )
    strat._num_points = _T
    strat._dwell_time_s = _DWELL
    strat._num_resonances = _M
    strat._time_axis_checked = False
    strat._parameter_activation = "softplus"
    # Resolved from the REAL registry, exactly as setup does. A bare callable
    # stub would let the strategy pass while dispatching to physics the
    # SignalModelRegistry never vouched for.
    strat._signal_model = get_signal_model("mrs_lorentzian")
    strat._lambda_fid = 1.0
    strat._lambda_prior = 0.0
    for key, value in overrides.items():
        setattr(strat, key, value)
    return strat


def _batch(**extra: object) -> dict:
    """An MRS batch. `target` is the FID itself: the arm is self-supervised."""
    torch.manual_seed(0)
    fid = torch.rand(2, 2 * _T, _SIZE, _SIZE) * 0.5
    batch = {"input": fid, "target": fid, "t_s": _time_axis()}
    batch.update(extra)
    return batch


# ---------------------------------------------------------------------------
# Registration / mounting.
# ---------------------------------------------------------------------------


def test_strategy_registered_and_config_mounted() -> None:
    from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    assert "mrs_quantification" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    assert "mrs_quantification" in TrainingStrategyConfigSchema.model_fields


def test_data_block_is_mounted_not_silently_dropped() -> None:
    """DataConfigSchema is extra="ignore" — an unmounted block vanishes quietly."""
    from spectramr.config.schemas.data import DataConfigSchema

    assert "spectroscopy" in DataConfigSchema.model_fields
    data = DataConfigSchema(spectroscopy={"enabled": True, "num_resonances": 5})
    assert data.spectroscopy.num_resonances == 5


def test_strategy_is_spectroscopy_tagged_for_the_ledger() -> None:
    from spectramr.config.schemas.enums import Regime, Task

    caps = MRSQuantificationStrategy.__dict__["capabilities"]
    assert caps.workflows == frozenset({Regime.SPECTROSCOPIC})
    assert caps.tasks == frozenset({Task.PARAMETER_MAPPING})


# ---------------------------------------------------------------------------
# The seams.
# ---------------------------------------------------------------------------


def test_loss_keys_and_finite() -> None:
    out = _strategy()._compute_losses_impl(input_batch=_batch())
    assert "g_total_loss" in out
    assert "loss_mrs_fid_residual" in out
    assert torch.isfinite(out["g_total_loss"])


def test_the_registry_resolved_model_is_the_one_actually_called() -> None:
    """The knob's reader must reach the physics, not just be stored.

    Resolving a spec at setup and then calling a hardcoded function anyway would
    leave `data.spectroscopy.signal_model` just as inert as before while LOOKING
    wired (#15). The only way to see the difference is to swap the resolved model
    for a sentinel and assert the sentinel runs.
    """
    import dataclasses

    calls = {"n": 0}
    real = get_signal_model("mrs_lorentzian")

    def _sentinel(t_s, a, f, lw, ph):
        calls["n"] += 1
        return real.fn(t_s, a, f, lw, ph)

    strat = _strategy(_signal_model=dataclasses.replace(real, fn=_sentinel))
    strat._compute_losses_impl(input_batch=_batch())
    assert calls["n"] == 1, (
        "MRSFIDResidualLoss did not call the model the strategy resolved from the "
        "registry — data.spectroscopy.signal_model is a decoy."
    )


def test_validation_returns_the_same_activated_params_the_fit_uses() -> None:
    """Assert the SEAM: the graded tensor must be the fitted tensor.

    The exact defect that shipped green in the perfusion strategy — an activation
    applied in the fit but not on the validation path, so the arm trained on
    softplus(out) and was graded on out.
    """
    strat = _strategy()
    batch = _batch()
    strat._compute_losses_impl(input_batch=batch)
    fitted = strat._last_prediction
    validated = strat._validation_forward(batch["input"], batch)
    torch.testing.assert_close(fitted, validated)


def test_softplus_is_applied_to_amplitude_and_linewidth_but_not_frequency() -> None:
    """The activation is PER-PARAMETER — only two of the four are sign-constrained.

    Forcing frequency positive would make half the spectrum unreachable (a
    resonance below the carrier has a negative offset), and phase is legitimately
    signed too.
    """
    strat = _strategy()
    fid = _batch()["input"]
    a, f, lw, ph = strat.predict_resonance_parameters(fid)
    assert bool((a >= 0).all()), "amplitudes must be softplus-positive"
    assert bool((lw >= 0).all()), "linewidths must be softplus-positive"

    raw = strat.env.generator(fid)
    torch.testing.assert_close(f, raw[:, _M : 2 * _M])  # frequency passed through
    torch.testing.assert_close(ph, raw[:, 3 * _M : 4 * _M])  # phase passed through


def test_softplus_is_why_the_loss_is_finite_at_all() -> None:
    """A negative linewidth turns exp(-pi*lw*t) into a GROWING exponential.

    The same numerical trap as Ktrans in the extended-Tofts kernel: an untrained
    generator emits negatives from step 0. Without the activation the FID
    overflows. Here the clamp inside lorentzian_fid is the backstop, so this pins
    that the activation keeps the linewidths in range rather than relying on it.
    """
    strat = _strategy()
    _, _, lw, _ = strat.predict_resonance_parameters(_batch()["input"])
    assert bool(torch.isfinite(lw).all())
    assert bool((lw >= 0).all())


def test_fid_channels_to_complex_round_trips() -> None:
    """[B, 2T, H, W] real/imag channels -> [B, H, W, T] complex, time LAST."""
    torch.manual_seed(0)
    real = torch.randn(2, _T, _SIZE, _SIZE)
    imag = torch.randn(2, _T, _SIZE, _SIZE)
    packed = torch.cat([real, imag], dim=1)
    out = MRSQuantificationStrategy._fid_channels_to_complex(packed)
    assert out.shape == (2, _SIZE, _SIZE, _T)
    torch.testing.assert_close(out.real, real.permute(0, 2, 3, 1))
    torch.testing.assert_close(out.imag, imag.permute(0, 2, 3, 1))


def test_an_odd_channel_count_raises() -> None:
    with pytest.raises(ValueError, match="even channel count"):
        MRSQuantificationStrategy._fid_channels_to_complex(torch.randn(1, 7, 2, 2))


# ---------------------------------------------------------------------------
# The batch contract.
# ---------------------------------------------------------------------------


def test_a_missing_time_axis_raises() -> None:
    """Assuming a default dwell time would silently rescale every frequency."""
    batch = _batch()
    del batch["t_s"]
    with pytest.raises(ValueError, match="t_s"):
        _strategy()._compute_losses_impl(input_batch=batch)


def test_a_time_axis_disagreeing_with_the_config_raises() -> None:
    """What makes data.spectroscopy.dwell_time_s a real knob rather than a doc.

    lorentzian_fid integrates any uniform axis happily, so a wrong dwell time
    refits to different, wrong frequencies with a healthy-looking residual.
    """
    wrong = torch.arange(_T, dtype=torch.float32) * (_DWELL * 3)
    with pytest.raises(ValueError, match="dwell_time_s"):
        _strategy()._compute_losses_impl(input_batch=_batch(t_s=wrong))


def test_a_time_axis_of_the_wrong_length_raises() -> None:
    with pytest.raises(ValueError, match="num_points"):
        _strategy()._compute_losses_impl(
            input_batch=_batch(t_s=torch.arange(_T + 5, dtype=torch.float32) * _DWELL)
        )


def test_the_time_axis_guard_runs_once_not_every_step() -> None:
    """torch.allclose returns a Python bool — a device->host sync (CLAUDE.md #9).

    The axis is a property of the acquisition, so one check is all it can need.
    """
    strat = _strategy()
    batch = _batch()
    for _ in range(4):
        strat._compute_losses_impl(input_batch=batch)
    assert strat._time_axis_checked is True


def test_a_non_mapping_batch_raises() -> None:
    with pytest.raises(ValueError, match="mapping batch"):
        _strategy()._compute_losses_impl(input_batch=torch.randn(2, 2 * _T, 4, 4))


def test_the_fit_reduces_the_residual_on_a_synthetic_fid() -> None:
    """End-to-end through the STRATEGY, on a real Lorentzian FID.

    Not a value assertion on the loss — those are vacuous. This trains the actual
    generator through the actual strategy and asserts the objective goes down,
    which is the weakest claim that still proves the whole path is connected.
    """
    torch.manual_seed(0)
    t = _time_axis()
    params = {
        "amplitudes": torch.full((1, _SIZE, _SIZE, _M), 1.0),
        "frequencies_hz": torch.full((1, _SIZE, _SIZE, _M), 60.0),
        "linewidths_hz": torch.full((1, _SIZE, _SIZE, _M), 5.0),
        "phases_rad": torch.zeros(1, _SIZE, _SIZE, _M),
    }
    clean = lorentzian_fid(t, **params)  # [1, H, W, T]
    channels = torch.cat(
        [clean.real.permute(0, 3, 1, 2), clean.imag.permute(0, 3, 1, 2)], dim=1
    )

    strat = _strategy()
    opt = torch.optim.Adam(strat.env.generator.parameters(), lr=1e-3)
    batch = {"input": channels, "target": channels, "t_s": t}

    first = float(strat._compute_losses_impl(input_batch=batch)["g_total_loss"])
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        loss = strat._compute_losses_impl(input_batch=batch)["g_total_loss"]
        loss.backward()
        opt.step()
    last = float(strat._compute_losses_impl(input_batch=batch)["g_total_loss"])
    assert last < first, f"the objective did not decrease: {first} -> {last}"


def test_no_weighted_loss_raises_rather_than_training_a_plain_regressor() -> None:
    """An arm advertising MRS physics with every lambda at 0 is pitfall #16."""
    from spectramr.config.schemas.data import DataConfigSchema

    strat = object.__new__(MRSQuantificationStrategy)
    strat.config = types.SimpleNamespace(
        data=DataConfigSchema(spectroscopy={"enabled": True}),
        model=types.SimpleNamespace(in_channels=2048, out_channels=12),
        training=types.SimpleNamespace(mrs_quantification=None),
    )
    strat._verify_strategy_config = lambda **_: None
    strat._get_loss_weight = lambda _name: 0.0
    with pytest.raises(ValueError, match="non-zero weight"):
        strat._setup_strategy_specific_components()
