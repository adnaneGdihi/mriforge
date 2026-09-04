"""Tier-1 audit guard for adaptive conditioning (3.1, 2026-05-26).

Rejects ``model.conditioning.enabled`` when (a) a source isn't a registered
encoder, or (b) the resolved strategy doesn't consume it — the silent-no-op
failure mode (CLAUDE.md #9). The pure decision logic is exercised directly;
the disabled fast-path is exercised through the check method.
"""

from __future__ import annotations

from types import SimpleNamespace

from spectramr.infrastructure.validation.config_health_checker import ConfigHealthChecker


def _err(sources, registered, supported):
    return ConfigHealthChecker._conditioning_source_error(sources, registered, supported)


_REG = {"severity_vec", "field_strength", "scanner_id", "site_id", "diffusion_t"}


def test_unknown_source_is_rejected():
    msg = _err(["severity_vec", "made_up"], _REG, {"severity_vec"})
    assert msg and "made_up" in msg


def test_strategy_consuming_nothing_is_rejected_silent_noop():
    # supported = empty set ⇒ strategy mixes in conditioning defaults but
    # populates no sources ⇒ enabling conditioning is a silent no-op.
    msg = _err(["severity_vec"], _REG, set())
    assert msg and "no-op" in msg.lower()


def test_source_not_supported_by_strategy_is_rejected():
    msg = _err(["field_strength"], _REG, {"severity_vec"})
    assert msg and "field_strength" in msg


def test_all_supported_passes():
    assert _err(["field_strength", "scanner_id"], _REG, {"field_strength", "scanner_id"}) is None


def test_unresolved_strategy_skips_strategy_check():
    # supported=None ⇒ strategy couldn't be resolved; only source-validity runs.
    assert _err(["field_strength"], _REG, None) is None
    assert _err(["nope"], _REG, None) is not None


def test_disabled_conditioning_passes():
    cfg = SimpleNamespace(
        model=SimpleNamespace(conditioning=SimpleNamespace(enabled=False, sources=[])),
        training=SimpleNamespace(),
    )
    r = ConfigHealthChecker().check_conditioning_sources_supported(cfg)
    assert r.passed and r.severity == "info"


def test_passing_message_is_honest_about_runtime_propagation(monkeypatch):
    """The Tier-1 check is structural; it must not over-claim runtime
    consumption. The passing message points at the mechanism-fires guard that
    actually enforces propagation, rather than asserting 'consumed'."""
    import spectramr.infrastructure.training.strategy_factory as sf

    class _Strat:
        _SUPPORTED_CONDITION_SOURCES = ("field_strength",)

    monkeypatch.setattr(
        sf.TrainingStrategyFactory,
        "get_strategy_class",
        staticmethod(lambda config: _Strat),
    )
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            conditioning=SimpleNamespace(enabled=True, sources=["field_strength"])
        ),
        training=SimpleNamespace(),
    )
    r = ConfigHealthChecker().check_conditioning_sources_supported(cfg)
    assert r.passed
    assert "consumed" not in r.message  # no over-claim of runtime consumption
    assert "supported by the strategy" in r.message
    assert "mechanism-fires guard" in r.message
