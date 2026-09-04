"""Contract tests for the compatibility matrix (cached-cascade WS-X).

Phase-0 freeze: pins the pure-rule registration scaffolding + the
``check_combination(ctx) -> list[CompatMessage]`` signature. WS-X Phase B adds
the seven rule families (and parametrizes over MODEL x STRATEGY x LOSS).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import pytest

from spectramr.domain.entities import ResolvedExperimentContext
from spectramr.domain.entities.experiment_context import DataProfile
from spectramr.infrastructure.validation import compatibility_matrix as cm
from spectramr.models.capabilities import (
    AdapterCapabilities,
    LossCapabilities,
    ModelCapabilities,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def _isolated_rules() -> Iterator[None]:
    """Save/restore the global rule list so tests don't leak rules."""
    saved = list(cm._RULES)
    try:
        yield
    finally:
        cm._RULES[:] = saved


def _ctx() -> ResolvedExperimentContext:
    return ResolvedExperimentContext(
        config_version="6.1", model_type="unet", strategy_name="GAN", loss_names=("l1",)
    )


def test_empty_rule_set_returns_no_messages() -> None:
    # Phase-0: the frozen matrix ships empty; a consistent combo yields [].
    assert cm.check_combination(_ctx()) == []


def test_register_rule_is_invoked_by_check(_isolated_rules: None) -> None:
    @cm.register_rule
    def _always_warn(ctx: ResolvedExperimentContext) -> list[cm.CompatMessage]:
        return [
            cm.CompatMessage(
                rule="probe",
                message=f"saw model {ctx.model_type}",
                severity="warning",
                fix_hint="n/a",
            )
        ]

    msgs = cm.check_combination(_ctx())
    assert len(msgs) == 1
    assert msgs[0].rule == "probe"
    assert "unet" in msgs[0].message
    assert _always_warn in cm.list_rules()


def test_compat_message_is_frozen_and_structured() -> None:
    msg = cm.CompatMessage(rule="r", message="m")
    assert msg.severity == "error"
    assert msg.category == "compat"
    assert msg.fix_hint is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        msg.message = "x"  # type: ignore[misc]  # frozen


def test_check_combination_aggregates_multiple_rules(_isolated_rules: None) -> None:
    cm.register_rule(lambda ctx: [cm.CompatMessage(rule="a", message="a")])
    cm.register_rule(lambda ctx: [cm.CompatMessage(rule="b", message="b")])
    rules_now = cm.check_combination(_ctx())
    assert {m.rule for m in rules_now} == {"a", "b"}


# --------------------------------------------------------------------------
# WS-X Phase B — the five rule families. These run against the rules that are
# registered at import time (no _isolated_rules fixture), so they also assert
# the rules are wired in.
# --------------------------------------------------------------------------


def _mk(**kw: object) -> ResolvedExperimentContext:
    """Build a context with sensible provenance + caller-supplied profiles."""
    kw.setdefault("config_version", "6.1")
    kw.setdefault("model_type", "unet")
    kw.setdefault("strategy_name", "GAN")
    return ResolvedExperimentContext(**kw)  # type: ignore[arg-type]


def _fired(ctx: ResolvedExperimentContext) -> set[str]:
    return {m.rule for m in cm.check_combination(ctx)}


def test_default_context_is_clean() -> None:
    # All profiles unannotated -> every rule fail-soft -> no messages.
    assert cm.check_combination(_mk()) == []


def test_domain_chain_input_mismatch_fires() -> None:
    ctx = _mk(
        data_profile=DataProfile(domain="image"),
        model_profile=ModelCapabilities(input_domain="kspace"),
    )
    assert "domain_chain" in _fired(ctx)


def test_domain_chain_matching_input_is_clean() -> None:
    ctx = _mk(
        data_profile=DataProfile(domain="image"),
        model_profile=ModelCapabilities(input_domain="image"),
    )
    assert "domain_chain" not in _fired(ctx)


def test_domain_chain_adapter_bridges_is_clean() -> None:
    ctx = _mk(
        data_profile=DataProfile(domain="image"),
        model_profile=ModelCapabilities(input_domain="kspace"),
        adapter_chain=(
            AdapterCapabilities(bridges_from={"domain": "image"}, bridges_to={"domain": "kspace"}),
        ),
    )
    assert "domain_chain" not in _fired(ctx)


def test_domain_chain_internal_crossing_model_is_clean() -> None:
    # An internal domain-crossing model (registered input_domain != output_domain,
    # e.g. hamiltonian_trajectory_generator: kspace -> image via an internal iFFT)
    # bridges the domains itself. data_profile.domain is proxied from
    # model.target_domain (== the model OUTPUT), so it equals the output, not the
    # input-feeding data; comparing that proxy to input_domain false-rejects the
    # legitimate crossing. The exemption applies only when the proxy == output.
    ctx = _mk(
        data_profile=DataProfile(domain="image"),  # proxy == model OUTPUT domain
        model_profile=ModelCapabilities(input_domain="kspace", output_domain="image"),
    )
    assert "domain_chain" not in _fired(ctx)


def test_domain_chain_crossing_model_proxy_not_output_still_fires() -> None:
    # The exemption is narrow: it applies ONLY when the proxied data domain equals
    # the model OUTPUT (a self-consistent IR). A proxy matching neither input nor
    # output is a real inconsistency and must still fire.
    ctx = _mk(
        data_profile=DataProfile(domain="complex"),
        model_profile=ModelCapabilities(input_domain="kspace", output_domain="image"),
    )
    assert "domain_chain" in _fired(ctx)


def test_domain_chain_output_loss_mismatch_does_not_fire() -> None:
    # The model.output -> loss.domain leg is intentionally NOT checked by the
    # matrix: the LossBuilder auto-bridges domain-specific loss blocks
    # (image/kspace/complex_losses) to the output domain, so an output<->loss
    # "mismatch" is a false positive. That leg is owned by
    # config_health_checker.loss_domain_consistency (bridge-aware).
    ctx = _mk(
        loss_names=("kspace_l1",),
        model_profile=ModelCapabilities(output_domain="image"),
        loss_profile=(LossCapabilities(domain="kspace"),),
    )
    assert "domain_chain" not in _fired(ctx)


def test_required_config_fields_fires() -> None:
    ctx = _mk(unresolved_required_fields=("training.diffusion.num_timesteps",))
    msgs = cm.check_combination(ctx)
    assert any(m.rule == "required_config_fields" for m in msgs)
    # the offending dotted key is surfaced for the fix
    assert any("num_timesteps" in m.message for m in msgs)


def test_required_config_fields_clean_when_all_resolved() -> None:
    assert "required_config_fields" not in _fired(_mk(unresolved_required_fields=()))


def test_paired_data_fires_when_model_requires_but_unpaired() -> None:
    ctx = _mk(
        data_profile=DataProfile(paired=False),
        model_profile=ModelCapabilities(requires_paired_data=True),
    )
    assert "paired_data" in _fired(ctx)


def test_paired_data_clean_when_paired_or_unannotated() -> None:
    assert "paired_data" not in _fired(
        _mk(
            data_profile=DataProfile(paired=True),
            model_profile=ModelCapabilities(requires_paired_data=True),
        )
    )
    # paired unknown (None) -> fail-soft, no fire even if model requires pairing
    assert "paired_data" not in _fired(
        _mk(model_profile=ModelCapabilities(requires_paired_data=True))
    )


def test_complex_channel_fires_when_model_rejects_complex() -> None:
    ctx = _mk(
        data_profile=DataProfile(channel_layout="complex"),
        model_profile=ModelCapabilities(accepts_complex=False),
    )
    assert "complex_channel" in _fired(ctx)


def test_complex_channel_clean_when_model_accepts() -> None:
    ctx = _mk(
        data_profile=DataProfile(channel_layout="complex"),
        model_profile=ModelCapabilities(accepts_complex=True),
    )
    assert "complex_channel" not in _fired(ctx)


def test_spatial_rank_fires_on_disjoint_ranks() -> None:
    ctx = _mk(
        data_profile=DataProfile(spatial_dims=(2,)),
        model_profile=ModelCapabilities(spatial_dims=(3,)),
    )
    assert "spatial_rank" in _fired(ctx)


def test_spatial_rank_clean_on_overlap_or_adapter() -> None:
    assert "spatial_rank" not in _fired(
        _mk(
            data_profile=DataProfile(spatial_dims=(2,)),
            model_profile=ModelCapabilities(spatial_dims=(2, 3)),
        )
    )
    bridged = _mk(
        data_profile=DataProfile(spatial_dims=(2,)),
        model_profile=ModelCapabilities(spatial_dims=(3,)),
        adapter_chain=(
            AdapterCapabilities(
                bridges_from={"spatial_dims": "2"}, bridges_to={"spatial_dims": "3"}
            ),
        ),
    )
    assert "spatial_rank" not in _fired(bridged)


def test_fully_consistent_context_is_clean() -> None:
    ctx = _mk(
        loss_names=("l1",),
        data_profile=DataProfile(
            domain="image", paired=True, spatial_dims=(2,), channel_layout="magnitude"
        ),
        model_profile=ModelCapabilities(
            input_domain="image",
            output_domain="image",
            spatial_dims=(2,),
            accepts_complex=False,
            requires_paired_data=True,
        ),
        loss_profile=(LossCapabilities(domain="image", requires_paired_target=True),),
    )
    assert cm.check_combination(ctx) == []


def test_validate_experiment_compatibility_seam_runs_end_to_end() -> None:
    # The Tier-1 seam resolves a (fail-soft) config to the IR and runs every
    # rule, returning a well-formed list — no crash on a minimal config.
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        config_version="6.1",
        model=SimpleNamespace(model_type="unet", target_domain=None, model_domain=None),
        training=SimpleNamespace(training_mode="reconstruction"),
        data=SimpleNamespace(),
        losses=None,
        physics=None,
    )
    msgs = cm.validate_experiment_compatibility(cfg)
    assert isinstance(msgs, list)
    assert all(isinstance(m, cm.CompatMessage) for m in msgs)


def test_every_rule_message_is_well_formed() -> None:
    # An aggressively-inconsistent context should fire several rules, and every
    # message must be actionable (non-empty fix_hint, valid severity).
    ctx = _mk(
        loss_names=("kspace_l1",),
        unresolved_required_fields=("training.diffusion.num_timesteps",),
        data_profile=DataProfile(
            domain="image", paired=False, spatial_dims=(2,), channel_layout="complex"
        ),
        model_profile=ModelCapabilities(
            input_domain="kspace",
            output_domain="image",
            spatial_dims=(3,),
            accepts_complex=False,
            requires_paired_data=True,
        ),
        loss_profile=(LossCapabilities(domain="kspace", requires_paired_target=True),),
    )
    msgs = cm.check_combination(ctx)
    assert len(msgs) >= 4
    for m in msgs:
        assert m.severity in {"error", "warning", "info"}
        assert m.message and m.fix_hint
        assert m.category == "compat"
