"""Combinatorial YAML-combination fuzz (cached-cascade WS-X) -- Phase-0 stub.

WS-X Phase B fills this in: sweep ``{model x strategy x loss x data-domain x
physics}``, resolve -> check -> probe each combination, and assert every outcome
is BINARY (passes cleanly, or is rejected with a clear message). Any "silently
accepted then crashes in the probe" is campaign-blocking.

For now this only pins that the fuzz *harness* (resolver + matrix) is importable
and composes, then SKIPS the full sweep with an explicit reason so it shows up as
skipped (honest) rather than a vacuous pass.
"""

from __future__ import annotations

import itertools

import pytest

pytestmark = pytest.mark.fuzz


def test_fuzz_harness_imports() -> None:
    # The two halves the fuzz drives must import + compose.
    from spectramr.domain.entities import ResolvedExperimentContext
    from spectramr.infrastructure.validation.compatibility_matrix import check_combination
    from spectramr.infrastructure.validation.context_resolver import resolve

    assert callable(resolve)
    ctx = ResolvedExperimentContext(config_version="6.1", model_type="unet", strategy_name="GAN")
    # All five rules fail-soft on an unannotated context.
    assert check_combination(ctx) == []


def _grid_contexts():
    """Enumerate synthetic resolved contexts across the rule-relevant axes."""
    from spectramr.domain.entities import ResolvedExperimentContext
    from spectramr.domain.entities.experiment_context import DataProfile
    from spectramr.models.capabilities import LossCapabilities, ModelCapabilities

    data_domains = (None, "image", "kspace", "complex_image")
    model_in = (None, "image", "kspace")
    paired = (None, True, False)
    req_paired = (None, True)
    layouts = (None, "complex", "magnitude")
    accepts_cx = (None, True, False)
    spatials = (None, (2,), (3,), (2, 3))

    for dd, mi, pr, rp, lay, ac, dsp, msp in itertools.product(
        data_domains, model_in, paired, req_paired, layouts, accepts_cx, spatials, spatials
    ):
        yield ResolvedExperimentContext(
            config_version="6.1",
            model_type="m",
            strategy_name="s",
            loss_names=("l",),
            data_profile=DataProfile(domain=dd, paired=pr, channel_layout=lay, spatial_dims=dsp),
            model_profile=ModelCapabilities(
                input_domain=mi,
                requires_paired_data=rp,
                accepts_complex=ac,
                spatial_dims=msp,
            ),
            loss_profile=(LossCapabilities(requires_paired_target=rp),),
        )


def test_full_combination_sweep_binary_and_wellformed() -> None:
    """Every synthetic combination is BINARY: either clean (``[]``) or rejected
    with well-formed, actionable messages. No ambiguous / malformed output, and
    every rejection carries a fix hint — the campaign's "components agree or are
    rejected with a clear message" guarantee, exercised over the rule layer.

    (The full ``MODEL_REGISTRY x STRATEGY_CLASS_PATHS x LossRegistry`` sweep with
    a real forward-probe is deferred to WS-C, once the component workstreams'
    capability declarations are merged into the trunk — until then most registry
    components carry no capabilities and the rules would skip vacuously.)
    """
    from spectramr.infrastructure.validation.compatibility_matrix import (
        CompatMessage,
        check_combination,
    )

    n, rejected = 0, 0
    for ctx in _grid_contexts():
        msgs = check_combination(ctx)
        assert isinstance(msgs, list)
        # determinism / purity: same context -> same result, no exception.
        assert [m.rule for m in check_combination(ctx)] == [m.rule for m in msgs]
        for m in msgs:
            assert isinstance(m, CompatMessage)
            assert m.severity in {"error", "warning", "info"}
            assert m.message and m.fix_hint, "a rejection must be actionable"
        n += 1
        rejected += bool(msgs)
    # The grid must be non-trivial and contain BOTH clean and rejected outcomes
    # (otherwise the rules are vacuous in one direction).
    assert n > 500
    assert 0 < rejected < n


def test_known_invalid_combo_is_rejected_with_error() -> None:
    from spectramr.domain.entities import ResolvedExperimentContext
    from spectramr.domain.entities.experiment_context import DataProfile
    from spectramr.infrastructure.validation.compatibility_matrix import check_combination
    from spectramr.models.capabilities import ModelCapabilities

    ctx = ResolvedExperimentContext(
        config_version="6.1",
        model_type="m",
        strategy_name="s",
        data_profile=DataProfile(domain="image"),
        model_profile=ModelCapabilities(input_domain="kspace"),
    )
    msgs = check_combination(ctx)
    assert any(m.severity == "error" for m in msgs)
