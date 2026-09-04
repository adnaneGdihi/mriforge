"""Tests for strategy detection from config.

Targets ``spectramr.infrastructure.inference.strategy_detector``. Replaces a
~30-branch ``if/elif`` chain with a declarative rule list. Tests focus
on:

- ``StrategyDetectionRule.matches``: pattern, equals, contains, checker
- ``StrategyDetectionRule._extract_by_path``: dotted-path extraction
- ``StrategyDetector.detect``: priority ordering (first match wins)
- ``create_default_detector``: full end-to-end on representative configs
"""

from __future__ import annotations

import pytest

from spectramr.infrastructure.inference.strategy_detector import (
    StrategyDetectionRule,
    StrategyDetector,
    create_default_detector,
)


# ---------------------------------------------------------------------------
# StrategyDetectionRule — extraction & matching
# ---------------------------------------------------------------------------


def test_extract_by_path_simple_key() -> None:
    """Single-level path extraction."""
    cfg = {"foo": "bar"}
    assert StrategyDetectionRule._extract_by_path(cfg, "foo") == "bar"


def test_extract_by_path_nested() -> None:
    """Dotted path traverses nested dicts."""
    cfg = {"a": {"b": {"c": 42}}}
    assert StrategyDetectionRule._extract_by_path(cfg, "a.b.c") == 42


def test_extract_by_path_missing_returns_none() -> None:
    """Missing keys → None."""
    assert StrategyDetectionRule._extract_by_path({}, "ghost") is None
    assert StrategyDetectionRule._extract_by_path({"a": {}}, "a.b") is None


def test_extract_by_path_non_dict_returns_none() -> None:
    """Non-dict mid-path → None."""
    assert StrategyDetectionRule._extract_by_path({"a": "string"}, "a.b") is None


# ---------------------------------------------------------------------------
# Rule matchers
# ---------------------------------------------------------------------------


def test_value_pattern_case_insensitive() -> None:
    """``value_pattern`` matching is case-insensitive."""
    rule = StrategyDetectionRule(
        strategy_name="gan", config_path="mode", value_pattern="GAN"
    )
    assert rule.matches({"mode": "gan"}) is True
    assert rule.matches({"mode": "Gan"}) is True
    assert rule.matches({"mode": "diffusion"}) is False


def test_value_equals_exact_match() -> None:
    """``value_equals`` does exact comparison."""
    rule = StrategyDetectionRule(
        strategy_name="vae", config_path="enabled", value_equals=True
    )
    assert rule.matches({"enabled": True}) is True
    assert rule.matches({"enabled": False}) is False


def test_contains_substring_match() -> None:
    """``contains`` checks case-insensitive substring."""
    rule = StrategyDetectionRule(
        strategy_name="diffusion", config_path="model_type", contains="DIFF"
    )
    assert rule.matches({"model_type": "cold_diffusion_v2"}) is True
    assert rule.matches({"model_type": "gan"}) is False


def test_custom_extractor_and_checker() -> None:
    """Rule with custom ``extractor`` + ``checker`` works."""
    rule = StrategyDetectionRule(
        strategy_name="big",
        extractor=lambda cfg: cfg.get("size", 0),
        checker=lambda v: v > 100,
    )
    assert rule.matches({"size": 200}) is True
    assert rule.matches({"size": 50}) is False


def test_extractor_keyerror_returns_false() -> None:
    """Extractor raising KeyError → match is False."""
    rule = StrategyDetectionRule(
        strategy_name="x",
        extractor=lambda cfg: cfg["definitely_missing"],
        checker=lambda v: True,
    )
    assert rule.matches({}) is False


def test_no_extractor_or_path_returns_false() -> None:
    """Rule with neither ``extractor`` nor ``config_path`` → no match."""
    rule = StrategyDetectionRule(strategy_name="x", value_pattern="foo")
    assert rule.matches({"foo": "x"}) is False


def test_no_checker_returns_false() -> None:
    """Rule with no checker / pattern → no match."""
    rule = StrategyDetectionRule(strategy_name="x", config_path="foo")
    assert rule.matches({"foo": "bar"}) is False


# ---------------------------------------------------------------------------
# StrategyDetector — priority ordering
# ---------------------------------------------------------------------------


def test_empty_detector_returns_default() -> None:
    """No rules → default."""
    detector = StrategyDetector()
    assert detector.detect({}, default="reconstruction") == "reconstruction"


def test_first_matching_rule_wins() -> None:
    """Multiple matching rules → first registered wins."""
    detector = StrategyDetector()
    detector.add_rule(
        StrategyDetectionRule(
            strategy_name="first", config_path="kind", value_pattern="x"
        )
    )
    detector.add_rule(
        StrategyDetectionRule(
            strategy_name="second", config_path="kind", value_pattern="x"
        )
    )
    assert detector.detect({"kind": "x"}) == "first"


def test_no_match_returns_explicit_default() -> None:
    """Custom default returned when no rule matches."""
    detector = StrategyDetector()
    detector.add_rule(
        StrategyDetectionRule(
            strategy_name="gan", config_path="kind", value_pattern="GAN"
        )
    )
    assert detector.detect({"kind": "diffusion"}, default="my_default") == "my_default"


# ---------------------------------------------------------------------------
# create_default_detector — end-to-end on real configs
# ---------------------------------------------------------------------------


def test_default_detector_detects_explicit_training_mode() -> None:
    """``training.training_mode='gan'`` resolves to ``gan``."""
    detector = create_default_detector()
    cfg = {"training": {"training_mode": "gan"}}
    assert detector.detect(cfg) == "gan"


def test_default_detector_detects_diffusion_via_training_mode() -> None:
    """Diffusion mode → ``diffusion``."""
    detector = create_default_detector()
    cfg = {"training": {"training_mode": "diffusion"}}
    assert detector.detect(cfg) == "diffusion"


def test_default_detector_adversarial_maps_to_gan() -> None:
    """``adversarial`` mode → ``gan`` (alias)."""
    detector = create_default_detector()
    cfg = {"training": {"training_mode": "adversarial"}}
    assert detector.detect(cfg) == "gan"


def test_default_detector_disentangled_section_detected() -> None:
    """Presence of ``disentangled`` config section → disentangled strategy."""
    detector = create_default_detector()
    cfg = {"disentangled": {"some_key": "value"}}
    assert detector.detect(cfg) == "disentangled"


def test_default_detector_cold_diffusion_via_model_type() -> None:
    """``model_type='cold_diffusion_v2'`` → ``cold_diffusion``."""
    detector = create_default_detector()
    cfg = {"model_type": "cold_diffusion_v2"}
    assert detector.detect(cfg) == "cold_diffusion"


def test_default_detector_unknown_falls_back_to_reconstruction() -> None:
    """Empty / unknown config → default ``reconstruction``."""
    detector = create_default_detector()
    assert detector.detect({}) == "reconstruction"


def test_default_detector_latent_dim_implies_vae() -> None:
    """Model config with ``latent_dim`` → VAE."""
    detector = create_default_detector()
    cfg = {"model": {"latent_dim": 128}}
    assert detector.detect(cfg) == "vae"


def test_default_detector_style_dim_implies_disentangled() -> None:
    """Model config with ``style_dim`` → disentangled."""
    detector = create_default_detector()
    cfg = {"model": {"style_dim": 256}}
    assert detector.detect(cfg) == "disentangled"


# ---------------------------------------------------------------------------
# The vacuous-presence defect (#1396) — tripwire, not a pin
# ---------------------------------------------------------------------------
#
# Every case above feeds a hand-built dict, and that is exactly why none of them
# sees the defect: ``detect({}) == "reconstruction"`` passes only because ``{}``
# lacks a key every real config carries. The production path passes
# ``TrainingSettings.model_dump()`` (``pipelines/infer.py:225``), and Pydantic
# emits *every declared field* — including the deprecated, always-``None``
# top-level ``diffusion`` (``config/settings.py:222``, whose own docstring says
# it is unused). The priority-4 rule tests ``"diffusion" in cfg``, i.e. key
# PRESENCE, so on that path it is a tautology: 523 of 647 ``inprogress/`` arms
# resolve to ``DiffusionInferenceStrategy``, 249 of them declaring a different
# ``training_mode``.
#
# This is an xfail rather than an assertion of today's behaviour on purpose — a
# test that pinned the tautology would have to be deleted by whoever fixes it,
# and would read as intent in the meantime. It asserts the *mechanism* and not
# an outcome, so it stays correct whichever way D16#2 resolves the routing: a
# field that is merely *declared* must never select a paradigm.


@pytest.mark.xfail(
    strict=True,
    reason="#1396: priority-4 tests key presence, and the deprecated top-level "
    "'diffusion' field is present in every model_dump(). Delete this marker "
    "with the fix (D16#2).",
)
def test_declared_but_unset_diffusion_field_does_not_select_a_paradigm() -> None:
    """A ``diffusion`` key that is present-but-``None`` is not a declaration."""
    detector = create_default_detector()
    cfg = {"diffusion": None, "training": {"training_mode": "reconstruction"}}
    assert detector.detect(cfg) != "diffusion"
