"""Combinatorial compatibility rules -- pure functions over the experiment IR.

Cached-cascade WS-X, Phase B. Every rule here is a **pure function over**
:class:`~mriforge.domain.entities.experiment_context.ResolvedExperimentContext`
``check(ctx) -> list[CompatMessage]`` -- no registry or config digging inside a
rule, because the :func:`context_resolver.resolve` step already populated every
profile. That purity is what lets the rules unit-test without constructing
models, and is asserted by the architecture fitness suite + WS-C "context
purity" check.

**Phase-0 freeze:** this module is importable with the rule *signature* and
registration scaffolding frozen, and an empty rule set (``check_combination``
returns ``[]``). WS-X Phase B fills in the seven rule families:

1. domain chain ``data -> model.input/output -> loss.domain -> metric domain``
2. strategy <-> model paradigm (``strategy_profile.supported_paradigms``)
3. ``required_config_fields`` resolve non-None (the ``training.diffusion.
   num_timesteps``-class bug)
4. paired-data requirement
5. complex / channel-layout agreement
6. spatial rank with adapter-chain bridging
7. conditioning vs multi-contrast

Each message carries a ``fix_hint`` (mirrors ``HealthCheckResult`` /
``ProbeResult``). Rules register through the existing
``config/schemas/validator_registry.py`` lazy-bootstrap (WS-X Phase B) so the
audit runs them at the Tier-1 boundary -- they are *not* run inside Pydantic
load, preserving config-layer purity.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mriforge.domain.entities.experiment_context import ResolvedExperimentContext

__all__ = [
    "CompatMessage",
    "CompatRule",
    "check_combination",
    "list_rules",
    "register_rule",
    "validate_experiment_compatibility",
]


def validate_experiment_compatibility(config: object) -> list[CompatMessage]:
    """Tier-1 seam: resolve a config to the IR, then run every rule over it.

    ``check_combination(resolve(config))`` in one call — the entry point the
    audit ladder / ``cli/app.py`` invokes at the Tier-1 boundary (outside the
    Pydantic load, preserving config-layer purity). Returns the (possibly empty)
    list of :class:`CompatMessage`; callers treat any ``error``-severity message
    as a rejected experiment.
    """
    from mriforge.infrastructure.validation.context_resolver import resolve

    return check_combination(resolve(config))


@dataclass(frozen=True)
class CompatMessage:
    """One compatibility finding over a resolved context.

    Mirrors the structured-diagnostic shape of ``HealthCheckResult`` /
    ``ProbeResult`` so the finding ledger and the audit renderer can consume
    them uniformly.
    """

    rule: str
    message: str
    severity: str = "error"  # "error" | "warning" | "info"
    category: str = "compat"
    fix_hint: str | None = None
    yaml_keys: tuple[str, ...] = ()


#: A compatibility rule: a pure function from the resolved IR to messages.
CompatRule = Callable[[ResolvedExperimentContext], list[CompatMessage]]

# The rule set. Frozen-empty in Phase 0; WS-X Phase B appends via @register_rule.
_RULES: list[CompatRule] = []


def register_rule(fn: CompatRule) -> CompatRule:
    """Register a pure compatibility rule.

    The rule MUST be a pure function of its ``ResolvedExperimentContext``
    argument -- no registry/config access (the resolver already did that).
    """
    _RULES.append(fn)
    return fn


def list_rules() -> tuple[CompatRule, ...]:
    """Return the registered rules (read-only view)."""
    return tuple(_RULES)


def check_combination(ctx: ResolvedExperimentContext) -> list[CompatMessage]:
    """Run every registered rule over the resolved context.

    Pure aggregation: returns the concatenation of all rules' messages. An
    empty list means "this combination is internally consistent". Callers
    (Tier-1 audit, contract tests, fuzz) treat a non-empty ``error``-severity
    result as a rejected combination.
    """
    messages: list[CompatMessage] = []
    for rule in _RULES:
        messages.extend(rule(ctx))
    return messages


# ---------------------------------------------------------------------------
# WS-X Phase B — the rule families (pure functions over the IR).
#
# INVARIANT: every rule is *fail-soft on unannotated facts* — if a field it
# would check is ``None`` it returns ``[]`` ("unannotated, skip"). This mirrors
# the capability-dataclass convention and keeps the rules non-breaking: a combo
# only fails a rule when BOTH sides are explicitly declared and conflict. (The
# contract tests rely on this: a default/unannotated context yields ``[]``.)
# ---------------------------------------------------------------------------


def _domain_set(d: object) -> set[str] | None:
    """Normalise a ``Domain | tuple[Domain, ...] | None`` to a set (or None)."""
    if d is None:
        return None
    if isinstance(d, str):
        return {d}
    try:
        return {str(x) for x in d}  # tuple/list of domains
    except TypeError:
        return {str(d)}


def _adapter_bridges(
    ctx: ResolvedExperimentContext, axis: str, src: set[str], dst: set[str]
) -> bool:
    """True iff some declared adapter bridges ``src -> dst`` on ``axis``."""
    for ad in ctx.adapter_chain:
        af = ad.bridges_from.get(axis)
        at = ad.bridges_to.get(axis)
        if af is not None and at is not None and str(af) in src and str(at) in dst:
            return True
    return False


def _domain_compatible(src: object, dst: object, ctx: ResolvedExperimentContext) -> bool:
    """Fail-soft domain compatibility: True if unannotated, directly overlapping,
    or bridged by a declared adapter."""
    s, d = _domain_set(src), _domain_set(dst)
    if s is None or d is None:
        return True  # unannotated -> skip
    if s & d:
        return True
    return _adapter_bridges(ctx, "domain", s, d)


@register_rule
def rule_domain_chain(ctx: ResolvedExperimentContext) -> list[CompatMessage]:
    """Rule 1: data.domain -> model.input_domain (adapter-bridgeable).

    The ``model.output -> loss.domain`` leg is intentionally NOT checked here:
    the LossBuilder auto-inserts an FFT/IFFT bridge per loss-block placement
    (``image_losses`` / ``kspace_losses`` / ``complex_losses``), so a
    domain-specific loss is ALWAYS bridged to the model's output domain — a naive
    output<->loss comparison false-positives (e.g. an ``image`` model with a
    ``kspace_losses`` entry the LossBuilder FFT-bridges). That leg is owned,
    correctly and bridge-aware, by ``config_health_checker``'s
    ``loss_domain_consistency`` / ``model_loss_output_domain`` checks; duplicating
    it here only re-introduced false rejections.
    """
    msgs: list[CompatMessage] = []
    dp, mp = ctx.data_profile, ctx.model_profile
    # Internal domain-crossing model exemption: ``data_profile.domain`` is proxied
    # from ``model.target_domain`` (the model OUTPUT), not the input-feeding data
    # (see context_resolver._resolve_data_profile). For a model registered with a
    # genuine input!=output crossing handled INTERNALLY (e.g.
    # hamiltonian_trajectory_generator: kspace input -> image output via an internal
    # iFFT), comparing that output-proxy to input_domain false-rejects the legitimate
    # crossing. Skip the data->input leg only when the model truly crosses AND the
    # proxy is self-consistent with the OUTPUT (so a genuinely inconsistent proxy
    # still fires). Non-crossing models (input==output, or output unannotated) are
    # unaffected.
    in_set, out_set = _domain_set(mp.input_domain), _domain_set(mp.output_domain)
    model_crosses = bool(in_set and out_set and not (in_set & out_set))
    if model_crosses and _domain_compatible(dp.domain, mp.output_domain, ctx):
        return msgs
    if not _domain_compatible(dp.domain, mp.input_domain, ctx):
        msgs.append(
            CompatMessage(
                rule="domain_chain",
                message=(
                    f"data domain {dp.domain!r} is not accepted by model "
                    f"{ctx.model_type!r} (input_domain {mp.input_domain!r})"
                ),
                fix_hint=(
                    "pick a model whose input_domain matches the data domain, "
                    "or declare a pre-model adapter bridging the two"
                ),
                yaml_keys=("data", "model.model_type"),
            )
        )
    return msgs


@register_rule
def rule_required_config_fields(ctx: ResolvedExperimentContext) -> list[CompatMessage]:
    """Rule 3: strategy-declared required config fields must resolve non-None.

    Catches the ``Missing required config: training.diffusion.num_timesteps``
    class of bug at the Tier-1 boundary. The resolver pre-computed the unset
    fields so this stays a pure function over the IR.
    """
    if not ctx.unresolved_required_fields:
        return []
    return [
        CompatMessage(
            rule="required_config_fields",
            message=(
                f"strategy {ctx.strategy_name!r} requires config field(s) that "
                f"are unset: {', '.join(ctx.unresolved_required_fields)}"
            ),
            fix_hint="set the listed dotted config key(s) in the YAML",
            yaml_keys=ctx.unresolved_required_fields,
        )
    ]


@register_rule
def rule_paired_data(ctx: ResolvedExperimentContext) -> list[CompatMessage]:
    """Rule 4: components requiring paired data vs data_profile.paired."""
    if ctx.data_profile.paired is not False:
        return []  # paired or unannotated -> ok
    needs: list[str] = []
    if ctx.model_profile.requires_paired_data is True:
        needs.append(f"model {ctx.model_type!r}")
    if any(lc.requires_paired_target is True for lc in ctx.loss_profile):
        needs.append("a configured loss")
    if not needs:
        return []
    return [
        CompatMessage(
            rule="paired_data",
            message=(
                f"{' and '.join(needs)} require paired (input, target) data, but "
                "the data profile is unpaired (paired=False)"
            ),
            fix_hint=(
                "provide paired data, or choose components that do not require "
                "pairing (e.g. a cycle/unpaired strategy)"
            ),
            yaml_keys=("data",),
        )
    ]


@register_rule
def rule_complex_channel(ctx: ResolvedExperimentContext) -> list[CompatMessage]:
    """Rule 5: complex data vs a model that declares it cannot accept complex."""
    dp, mp = ctx.data_profile, ctx.model_profile
    is_complex = (dp.channel_layout == "complex") or (_domain_set(dp.domain) == {"complex_image"})
    if is_complex and mp.accepts_complex is False:
        return [
            CompatMessage(
                rule="complex_channel",
                message=(
                    f"data is complex (layout={dp.channel_layout!r}, "
                    f"domain={dp.domain!r}) but model {ctx.model_type!r} declares "
                    "accepts_complex=False"
                ),
                fix_hint=(
                    "use a complex-capable model, or a pre-model adapter that "
                    "maps complex -> real (magnitude or real/imag interleave)"
                ),
                yaml_keys=("data", "model.model_type"),
            )
        ]
    return []


@register_rule
def rule_spatial_rank(ctx: ResolvedExperimentContext) -> list[CompatMessage]:
    """Rule 6: data spatial rank vs model spatial_dims (adapter-bridgeable)."""
    dp, mp = ctx.data_profile, ctx.model_profile
    if dp.spatial_dims is None or mp.spatial_dims is None:
        return []
    if set(dp.spatial_dims) & set(mp.spatial_dims):
        return []
    if _adapter_bridges(
        ctx,
        "spatial_dims",
        {str(x) for x in dp.spatial_dims},
        {str(x) for x in mp.spatial_dims},
    ):
        return []
    return [
        CompatMessage(
            rule="spatial_rank",
            message=(
                f"data spatial rank {dp.spatial_dims} is not supported by model "
                f"{ctx.model_type!r} (spatial_dims {mp.spatial_dims})"
            ),
            fix_hint=(
                "pick a model supporting the data's spatial rank, or add a "
                "rank-bridging adapter (e.g. slice<->volume)"
            ),
            yaml_keys=("data", "model.model_type"),
        )
    ]


# Rules 2 (strategy<->model paradigm) and 7 (model multi-contrast support) need
# capability fields the contracts do not yet carry (ModelCapabilities has no
# ``paradigm`` or ``multi_contrast`` field). They are intentionally deferred to
# the component capability-declaration work (WS-6) rather than guessed here;
# see docs/campaigns/cached_cascade/wsx_context_contract.md.
