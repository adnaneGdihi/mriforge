"""Tests for PerfusionKineticMappingStrategy — the trainable backing for mri_perfusion.

There is no DCE data on this cluster, but the tracer kinetics are analytic, so
the strategy's real claim IS testable: synthesise a curve from known
(Ktrans, ve, vp) and recover the parameters through the same forward model the
loss uses. That is a proof of the objective, not a smoke check.
"""

from __future__ import annotations

import types

import pytest
import torch

from mriforge.infrastructure.physics.signal_models.registry import get_signal_model
from mriforge.infrastructure.physics.signal_models.perfusion_kinetics import (
    extended_tofts_forward,
    parker_population_aif,
)
from mriforge.infrastructure.training.strategies.perfusion_kinetic_strategy import (
    PerfusionKineticMappingStrategy,
)

_T = 24
_SIZE = 6
_DT_S = 12.5  # must equal data.perfusion.temporal_resolution_s for the arm


def _time_axis() -> torch.Tensor:
    return torch.arange(_T, dtype=torch.float32) * _DT_S


class _MapNet(torch.nn.Module):
    """[B, T, H, W] curve -> [B, 3, H, W] (Ktrans, ve, vp). No terminal tanh."""

    def __init__(self) -> None:
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(_T, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def _strategy(**overrides: object) -> PerfusionKineticMappingStrategy:
    strat = object.__new__(PerfusionKineticMappingStrategy)
    strat.env = types.SimpleNamespace(generator=_MapNet(), losses={})
    strat.config = types.SimpleNamespace(
        losses=types.SimpleNamespace(
            image_losses=[], kspace_losses=[], complex_losses=[]
        )
    )
    strat._aif_source = "population"
    strat._num_frames = _T
    strat._temporal_resolution_s = _DT_S
    strat._time_axis_checked = False
    strat._parameter_activation = "softplus"
    # Resolved from the REAL registry, exactly as _setup_strategy_specific_
    # components does. Stubbing a bare callable here would let the strategy pass
    # while dispatching to physics the SignalModelRegistry never vouched for —
    # the seam under test is precisely that the ledger's claim and the runtime
    # dispatch resolve the same key.
    strat._kinetic_model = get_signal_model("extended_tofts")
    strat._lambda_tofts = 1.0
    strat._lambda_aif = 0.0
    strat._lambda_box = 0.0
    strat._lambda_smooth = 0.0
    for key, value in overrides.items():
        setattr(strat, key, value)
    return strat


def _batch(**extra: object) -> dict:
    """A DCE batch. `target` is the curve itself: the arm is self-supervised."""
    torch.manual_seed(0)
    curve = torch.rand(2, _T, _SIZE, _SIZE) * 0.5
    batch = {"input": curve, "target": curve, "t_s": _time_axis()}
    batch.update(extra)
    return batch


# ---------------------------------------------------------------------------
# Registration / mounting.
# ---------------------------------------------------------------------------


def test_strategy_registered_and_config_mounted() -> None:
    from mriforge.config.schemas.training.base import TrainingStrategyConfigSchema
    from mriforge.infrastructure.training.strategy_factory import (
        TrainingStrategyFactory,
    )

    assert "perfusion_kinetic" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    assert "perfusion_kinetic" in TrainingStrategyConfigSchema.model_fields


def test_data_block_is_mounted_not_silently_dropped() -> None:
    """DataConfigSchema is extra="ignore" — an unmounted block vanishes quietly."""
    from mriforge.config.schemas.data import DataConfigSchema

    assert "perfusion" in DataConfigSchema.model_fields
    data = DataConfigSchema(perfusion={"enabled": True, "num_frames": 40})
    assert data.perfusion.num_frames == 40


def test_strategy_is_perfusion_tagged_for_the_ledger() -> None:
    from mriforge.config.schemas.enums import Regime, Task

    caps = PerfusionKineticMappingStrategy.__dict__["capabilities"]
    assert caps.workflows == frozenset({Regime.PERFUSION})
    assert caps.tasks == frozenset({Task.PARAMETER_MAPPING})


# ---------------------------------------------------------------------------
# The objective.
# ---------------------------------------------------------------------------


def test_loss_keys_and_finite() -> None:
    out = _strategy()._compute_losses_impl(input_batch=_batch())
    assert "g_total_loss" in out
    assert "loss_tofts_residual" in out
    assert torch.isfinite(out["g_total_loss"])


def _fit_kinetics(
    t_s: torch.Tensor,
    aif: torch.Tensor,
    measured: torch.Tensor,
    lambda_box: float = 0.0,
    steps: int = 600,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Optimise (Ktrans, ve, vp) against the self-supervised Tofts residual.

    The lr is decayed because the residual is L1: a constant gradient magnitude
    means a fixed lr oscillates at scale ~lr rather than settling.
    """
    from mriforge.models.losses.perfusion_losses import (
        PerfusionPhysiologicalBoxLoss,
        ToftsResidualLoss,
    )

    ktrans = torch.full((4,), 0.05, requires_grad=True)
    ve = torch.full((4,), 0.50, requires_grad=True)
    vp = torch.full((4,), 0.01, requires_grad=True)
    opt = torch.optim.Adam([ktrans, ve, vp], lr=2e-2)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.995)
    tofts, box = ToftsResidualLoss(), PerfusionPhysiologicalBoxLoss()

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = tofts(
            t_s=t_s, aif=aif, ktrans=ktrans, ve=ve, vp=vp, measured_curve=measured
        )
        if lambda_box:
            loss = loss + lambda_box * box(ktrans=ktrans, ve=ve, vp=vp)
        loss.backward()
        opt.step()
        sched.step()

    residual = float(
        tofts(
            t_s=t_s, aif=aif, ktrans=ktrans, ve=ve, vp=vp, measured_curve=measured
        ).detach()
    )
    return ktrans.detach(), ve.detach(), vp.detach(), residual


def test_recovers_known_kinetic_parameters() -> None:
    """The OBJECTIVE recovers known kinetics, on analytic physics.

    Synthesise a curve from known (Ktrans, ve, vp), then optimise the maps back
    through ToftsResidualLoss — the same self-supervised objective the strategy
    uses, needing no ground-truth maps. That is what makes perfusion trainable on
    unlabelled DCE data, and it is verifiable with no real data at all.

    SCOPE — be precise about what this proves. It optimises free tensors against
    the registered loss; it does NOT call the strategy, so it verifies the loss +
    signal model, not the wiring. The wiring is covered separately by
    test_tofts_residual_is_called_by_keyword_not_position (the strategy feeds the
    loss correctly) and test_validation_returns_the_same_activated_maps_the_fit_uses
    (the graded tensor is the fitted one).

    Ktrans and ve are asserted; vp is NOT — see
    test_box_loss_rescues_the_degenerate_vp: vp is genuinely ill-posed here.
    """
    torch.manual_seed(0)
    t_s = _time_axis()
    aif = parker_population_aif(t_s)
    measured = extended_tofts_forward(
        t_s, aif, torch.full((4,), 0.20), torch.full((4,), 0.30), torch.full((4,), 0.05)
    )

    ktrans, ve, _vp, residual = _fit_kinetics(t_s, aif, measured)

    assert residual < 1e-3
    torch.testing.assert_close(ktrans, torch.full((4,), 0.20), rtol=0.2, atol=0.03)
    torch.testing.assert_close(ve, torch.full((4,), 0.30), rtol=0.2, atol=0.05)


def test_box_loss_rescues_the_degenerate_vp() -> None:
    """The extended-Tofts inverse problem is ILL-POSED for vp, and this is the cure.

    vp scales the AIF directly while Ktrans/ve scale its convolution, and the two
    trade off. So a near-perfect curve fit coexists with a physically IMPOSSIBLE
    vp: fitting the residual alone drives vp NEGATIVE (measured here) while the
    residual sits at ~1e-4 and looks like a triumph. A blood-plasma volume
    fraction below zero is not a small error — it is meaningless.

    A constraint fixes it, and essentially FREE: the residual is unchanged
    (~1e-4 both ways) because it is not fighting the data, it is resolving an
    ambiguity the data cannot.

    WHICH constraint, in the arm, is the subtle part. This test optimises FREE
    tensors, so the box loss's clamp(-vp) term is what rescues vp. The strategy
    defaults to parameter_activation='softplus', which forces vp > 0
    STRUCTURALLY — so on the arm the clamp(-vp) term is identically zero with
    zero gradient, and softplus is doing this job, not the box loss. The box
    loss's live contribution there is the ve+vp<=1 term.

    So read this test as: the vp degeneracy is REAL, and something must resolve
    it. Do not read it as "the box loss rescues vp on the arm" — under the
    default activation it cannot. If you set parameter_activation='none', it is
    the box loss or nothing.
    """
    torch.manual_seed(0)
    t_s = torch.linspace(0.0, 300.0, 60)  # 5 s — data.perfusion's default
    aif = parker_population_aif(t_s)
    measured = extended_tofts_forward(
        t_s, aif, torch.full((4,), 0.20), torch.full((4,), 0.30), torch.full((4,), 0.05)
    )

    _k_free, _v_free, vp_free, residual_free = _fit_kinetics(t_s, aif, measured)
    _k_box, _v_box, vp_box, residual_box = _fit_kinetics(t_s, aif, measured, 1.0)

    assert torch.all(
        vp_free < 0
    ), f"expected the unconstrained fit to drive vp negative, got {vp_free}"
    assert torch.all(
        vp_box > 0
    ), f"the box loss must pull vp back into physical range, got {vp_box}"
    # The constraint resolves an ambiguity rather than fighting the data, so the
    # fit quality is unharmed — within the same order of magnitude.
    assert residual_box < 10 * residual_free


def test_loss_reduces() -> None:
    torch.manual_seed(0)
    strat = _strategy()
    opt = torch.optim.Adam(strat.env.generator.parameters(), lr=2e-3)
    batch = _batch()
    first = None
    out = None
    for _ in range(80):
        opt.zero_grad(set_to_none=True)
        out = strat._compute_losses_impl(input_batch=batch)
        out["g_total_loss"].backward()
        opt.step()
        if first is None:
            first = float(out["g_total_loss"].detach())
    assert out is not None and first is not None
    assert float(out["g_total_loss"].detach()) < first


# ---------------------------------------------------------------------------
# The anti-facade gates.
# ---------------------------------------------------------------------------


def test_tofts_residual_is_called_by_keyword_not_position() -> None:
    """The highest-value regression in this file.

    ToftsResidualLoss.forward is (t_s, aif, ktrans, ve, vp, measured_curve) —
    six positional tensors, NOT (pred, target). Routed through the generic
    ComposedLoss.forward(pred, target) path it binds pred -> t_s and
    target -> aif, then fails the 1-D check inside the kinetics with a shape
    error that reads like a data bug rather than a wiring bug.
    """
    from mriforge.models.losses import perfusion_losses

    seen: dict[str, object] = {}
    original = perfusion_losses.ToftsResidualLoss.forward

    def _record(self, *args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return original(self, *args, **kwargs)

    perfusion_losses.ToftsResidualLoss.forward = _record
    try:
        _strategy()._compute_losses_impl(input_batch=_batch())
    finally:
        perfusion_losses.ToftsResidualLoss.forward = original

    assert seen["args"] == (), "tofts_residual was called positionally"
    assert set(seen["kwargs"]) == {
        "t_s",
        "aif",
        "ktrans",
        "ve",
        "vp",
        "measured_curve",
    }


def test_raises_when_measured_aif_is_declared_but_absent() -> None:
    """Substituting the population AIF would change the physics silently (#9)."""
    strat = _strategy(_aif_source="measured")
    with pytest.raises(ValueError, match="no 'aif' key"):
        strat._compute_losses_impl(input_batch=_batch())


def test_measured_aif_actually_reaches_the_kinetic_model() -> None:
    """Assert the measured AIF is USED, not merely that the loss is finite.

    The previous version of this test asserted only `torch.isfinite(...)`, which
    is true whether or not batch['aif'] is ever read — it would have passed with
    the population AIF silently substituted, leaving aif_source unverified as a
    knob (#15). Record the tensor that reaches the loss instead.
    """
    from mriforge.models.losses import perfusion_losses

    measured = parker_population_aif(_time_axis()) * 1.5
    seen: dict[str, torch.Tensor] = {}
    original = perfusion_losses.ToftsResidualLoss.forward

    def _record(self, **kwargs):
        seen["aif"] = kwargs["aif"]
        return original(self, **kwargs)

    perfusion_losses.ToftsResidualLoss.forward = _record
    try:
        _strategy(_aif_source="measured")._compute_losses_impl(
            input_batch=_batch(aif=measured)
        )
    finally:
        perfusion_losses.ToftsResidualLoss.forward = original

    torch.testing.assert_close(seen["aif"], measured)
    population = parker_population_aif(_time_axis())
    assert not torch.allclose(seen["aif"], population), (
        "the measured AIF is indistinguishable from the population one here — "
        "pick a scale that makes the substitution detectable"
    )


def test_population_aif_is_used_when_that_is_declared() -> None:
    """The other half: 'population' must NOT read a stray 'aif' batch key."""
    from mriforge.models.losses import perfusion_losses

    seen: dict[str, torch.Tensor] = {}
    original = perfusion_losses.ToftsResidualLoss.forward

    def _record(self, **kwargs):
        seen["aif"] = kwargs["aif"]
        return original(self, **kwargs)

    perfusion_losses.ToftsResidualLoss.forward = _record
    try:
        _strategy(_aif_source="population")._compute_losses_impl(
            input_batch=_batch(aif=parker_population_aif(_time_axis()) * 99.0)
        )
    finally:
        perfusion_losses.ToftsResidualLoss.forward = original

    torch.testing.assert_close(seen["aif"], parker_population_aif(_time_axis()))


def test_temporal_resolution_mismatch_raises() -> None:
    """data.perfusion.temporal_resolution_s must be READ, not documentation (#15).

    A t_s in the wrong units refits to different, wrong kinetics with a perfectly
    healthy residual — extended_tofts_forward integrates any uniform axis
    happily, and Parker's constants are in minutes.
    """
    batch = _batch()
    batch["t_s"] = _time_axis() / 60.0  # minutes, not seconds
    with pytest.raises(ValueError, match="does not match the batch's 't_s' spacing"):
        _strategy()._compute_losses_impl(input_batch=batch)


def test_raises_without_a_time_axis() -> None:
    """Assuming a default t_s would silently change the fitted kinetics."""
    batch = _batch()
    del batch["t_s"]
    with pytest.raises(ValueError, match="'t_s' batch key"):
        _strategy()._compute_losses_impl(input_batch=batch)


def test_builder_losses_are_not_folded_without_a_map_target() -> None:
    """The self-supervised arm has no target — folding would grade maps vs curve.

    losses.image_losses grades a (pred, target) pair. With no kinetic_maps
    target, folding the block against the measured CURVE would compare parameter
    maps to a concentration series (pitfall #18). Better to leave the declared
    block unapplied than to apply it to the wrong pair.
    """
    from mriforge.models.losses.charbonnier_loss import CharbonnierLoss

    strat = _strategy()
    strat.env.losses = {"hfen": CharbonnierLoss()}
    strat.config.losses = types.SimpleNamespace(
        image_losses=[{"name": "hfen", "weight": 0.2}],
        kspace_losses=[],
        complex_losses=[],
    )
    out = strat._compute_losses_impl(input_batch=_batch())
    assert "loss_builder_aux" not in out


def test_builder_losses_folded_when_a_map_target_exists() -> None:
    """A supervised arm DOES fold — the pair is then commensurate."""
    from mriforge.models.losses.charbonnier_loss import CharbonnierLoss
    from mriforge.models.losses.hfen_loss import HFENLoss

    strat = _strategy()
    strat.env.losses = {"l1": CharbonnierLoss(), "hfen": HFENLoss()}
    strat.config.losses = types.SimpleNamespace(
        image_losses=[{"name": "l1", "weight": 1.0}, {"name": "hfen", "weight": 0.2}],
        kspace_losses=[],
        complex_losses=[],
    )
    out = strat._compute_losses_impl(
        input_batch=_batch(kinetic_maps=torch.rand(2, 3, _SIZE, _SIZE))
    )
    assert "loss_builder_aux" in out and "loss_hfen" in out


def test_softplus_activation_is_actually_applied() -> None:
    """The knob must be read, or it is a facade (#15)."""
    curve = _batch()["input"]
    softplus = _strategy(_parameter_activation="softplus")
    softplus.env.generator.body[-1].bias.data.fill_(-5.0)
    assert torch.all(
        softplus.predict_parameter_maps(curve) > 0
    ), "softplus was declared but never applied"

    raw = _strategy(_parameter_activation="none")
    raw.env.generator.body[-1].bias.data.fill_(-5.0)
    assert torch.all(
        raw.predict_parameter_maps(curve) < 0
    ), "'none' must leave the maps raw"


def test_validation_returns_the_same_activated_maps_the_fit_uses() -> None:
    """REGRESSION: the graded maps must BE the fitted maps.

    The activation used to live inside _split_parameter_maps, so only the Tofts
    refit ever saw activated values. The raw pre-activation logits flowed to the
    smoothness prior, the supervised image-loss fold, _last_prediction, and —
    worst — _validation_forward, which returned generator(curve) directly. Under
    softplus that is logits, roughly half negative: `negative_voxels` would have
    reported ~50% and every map metric would have graded a tensor the fit never
    optimised (pitfall #18).

    The old unit test passed throughout, because it called _split_parameter_maps
    in ISOLATION. It proved the activation existed; it never proved the activated
    tensor was the one that reached the graders. Assert the seam, not the unit.
    """
    strat = _strategy(_parameter_activation="softplus")
    strat.env.generator.body[-1].bias.data.fill_(-5.0)
    batch = _batch()

    validated = strat._validation_forward(batch["input"], batch)
    assert torch.all(validated > 0), (
        "_validation_forward returned pre-activation logits — the PERFUSION "
        "metrics would grade a tensor the kinetic fit never used"
    )

    strat._compute_losses_impl(input_batch=batch)
    torch.testing.assert_close(strat._last_prediction, validated)


def test_supervised_fold_grades_activated_maps_not_logits() -> None:
    """The image-loss fold must compare physical maps to physical map targets.

    Folding raw logits against a `kinetic_maps` target compares a
    pre-activation tensor to physical Ktrans/ve/vp — a metric-claim mismatch
    that would silently train against a wrong pair.
    """
    from mriforge.models.losses.charbonnier_loss import CharbonnierLoss

    seen: dict[str, torch.Tensor] = {}
    strat = _strategy(_parameter_activation="softplus")
    strat.env.generator.body[-1].bias.data.fill_(-5.0)
    strat.env.losses = {"hfen": CharbonnierLoss()}
    strat.config.losses = types.SimpleNamespace(
        image_losses=[{"name": "hfen", "weight": 0.2}],
        kspace_losses=[],
        complex_losses=[],
    )
    original = strat._apply_builder_image_losses

    def _record(pred, target, components):
        seen["pred"] = pred
        return original(pred, target, components)

    strat._apply_builder_image_losses = _record
    strat._compute_losses_impl(
        input_batch=_batch(kinetic_maps=torch.rand(2, 3, _SIZE, _SIZE))
    )
    assert torch.all(
        seen["pred"] > 0
    ), "the fold received pre-activation logits, not physical parameter maps"


def test_softplus_is_why_the_loss_is_finite_at_all() -> None:
    """Pins WHY softplus is the default: without it the curve overflows to inf.

    The Tofts kernel is exp(-(Ktrans/ve)*t) and ve is clamped positive inside the
    kinetics, so the sign of kep follows Ktrans. A negative Ktrans flips the
    exponent positive and exp(+large) -> inf. An untrained generator emits
    negatives from step 0, so `none` is NaN on the first batch. This is a
    numerical requirement, not a style preference — hence the default.
    """
    torch.manual_seed(0)
    batch = _batch()

    finite = _strategy(_parameter_activation="softplus")._compute_losses_impl(
        input_batch=batch
    )
    assert torch.isfinite(finite["g_total_loss"])

    raw = _strategy(_parameter_activation="none")
    raw.env.generator.body[-1].bias.data.fill_(-1.0)  # force Ktrans < 0
    exploded = raw._compute_losses_impl(input_batch=batch)
    assert not torch.isfinite(exploded["g_total_loss"]), (
        "a negative Ktrans should overflow the Tofts kernel — if this now stays "
        "finite the kinetics gained a guard and `none` may be safe to default"
    )


def test_compute_losses_rejects_a_bare_tensor_batch() -> None:
    with pytest.raises(ValueError, match="mapping batch"):
        _strategy()._compute_losses_impl(input_batch=torch.rand(2, _T, _SIZE, _SIZE))


def test_compute_losses_accepts_a_canonical_trainingbatch() -> None:
    """REGRESSION: the real pipeline passes batch=<TrainingBatch> in kwargs."""
    from mriforge.data.batch_types import BatchAdapter

    tb = BatchAdapter.from_dict(_batch())
    out = _strategy()._compute_losses_impl(
        input_batch=tb.input, target_batch=tb.target, epoch=0, batch=tb
    )
    assert torch.isfinite(out["g_total_loss"])
    assert "loss_tofts_residual" in out


def test_time_axis_guard_runs_once_not_every_step() -> None:
    """The guard must not put a host sync back in the training loop (#9).

    `torch.allclose` returns a Python bool — a device->host transfer. Running it
    per iteration would reintroduce exactly the per-step sync that vectorising
    `extended_tofts_forward` removed. The time axis is a property of the
    acquisition, so one check is all it can ever need.
    """
    calls = {"n": 0}
    strat = _strategy()
    original = strat._check_time_axis

    def _count(time_axis):
        calls["n"] += 1
        return original(time_axis)

    strat._check_time_axis = _count
    batch = _batch()
    for _ in range(5):
        strat._compute_losses_impl(input_batch=batch)

    # The wrapper is called every step (it is the cheap early-return), but the
    # allclose behind it runs exactly once — that is what the latch pins.
    assert calls["n"] == 5
    assert strat._time_axis_checked is True


def test_the_registry_resolved_model_is_the_one_actually_called() -> None:
    """The knob's reader must reach the physics, not just be stored.

    ``data.perfusion.kinetic_model`` is resolved through the SignalModelRegistry
    at setup — but resolving a spec and then calling a hardcoded function anyway
    would leave the knob just as inert as before, while LOOKING wired (#15). The
    only way to see the difference is to swap the resolved model for a sentinel
    and assert the sentinel is what runs.

    This is the seam, not the unit: ``_kinetic_model`` being set proves nothing.
    """
    import dataclasses

    calls: dict[str, int] = {"n": 0}
    real = get_signal_model("extended_tofts")

    def _sentinel(t_s, aif, ktrans, ve, vp):  # noqa: ANN001, ANN202
        calls["n"] += 1
        return real.fn(t_s, aif, ktrans, ve, vp)

    strat = _strategy(_kinetic_model=dataclasses.replace(real, fn=_sentinel))
    strat._compute_losses_impl(input_batch=_batch())

    assert calls["n"] == 1, (
        "ToftsResidualLoss did not call the model the strategy resolved from the "
        "registry — data.perfusion.kinetic_model is a decoy."
    )


def test_a_signal_model_with_the_wrong_parameter_contract_is_rejected() -> None:
    """Sharing a regime does NOT make two models interchangeable.

    gamma_variate is PERFUSION physics with signature
    (t_s, amplitude, t0_s, alpha, beta_s). Dispatching Tofts kinetics to it would
    bind ktrans->amplitude and ve->t0_s and return a plausible curve, so the
    regime check alone cannot catch it — the parameter contract can.
    """
    from mriforge.config.schemas.enums import Regime

    spec = get_signal_model("gamma_variate")
    assert spec.regime is Regime.PERFUSION  # the regime check would PASS...
    assert spec.parameters != ("ktrans", "ve", "vp")  # ...and this is what catches it
