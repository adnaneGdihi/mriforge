"""Unit tests for the --only / --include / --exclude filter selectors on
:class:`CampaignOrchestrator`.

These tests exercise the pure filtering logic (``_selector_matches`` and
``_apply_filters``) without instantiating SLURM or touching disk.
"""

from __future__ import annotations

import pytest

from spectramr.config.schemas.campaign import CampaignExperimentSchema
from spectramr.infrastructure.orchestration.campaign_orchestrator import (
    CampaignOrchestrator,
)


def _arm(
    name: str,
    role: str = "variant",
    **tags: str,
) -> CampaignExperimentSchema:
    return CampaignExperimentSchema(
        name=name,
        config=f"experiments/{name}.yaml",
        role=role,
        tags=tags,
    )


@pytest.fixture
def arms() -> list[CampaignExperimentSchema]:
    return [
        _arm("ref",     role="baseline", semantic_role="reference",   arch="geo_mamba"),
        _arm("abl_a",   role="ablation", ablated="metric_sfc",        arch="geo_mamba"),
        _arm("abl_b",   role="ablation", ablated="film",              arch="geo_mamba"),
        _arm("base_u",  role="baseline", semantic_role="baseline",    arch="unet"),
        _arm("base_s",  role="baseline", semantic_role="baseline",    arch="swinir"),
    ]


# ── _selector_matches ──────────────────────────────────────────────────


def test_selector_matches_name(arms: list[CampaignExperimentSchema]) -> None:
    assert CampaignOrchestrator._selector_matches("name=abl_a", arms[1])
    assert not CampaignOrchestrator._selector_matches("name=abl_a", arms[0])


def test_selector_matches_role(arms: list[CampaignExperimentSchema]) -> None:
    assert CampaignOrchestrator._selector_matches("role=ablation", arms[1])
    assert not CampaignOrchestrator._selector_matches("role=ablation", arms[3])


def test_selector_matches_tag(arms: list[CampaignExperimentSchema]) -> None:
    assert CampaignOrchestrator._selector_matches("tag.arch=swinir", arms[4])
    assert not CampaignOrchestrator._selector_matches("tag.arch=swinir", arms[3])


def test_selector_rejects_missing_equals() -> None:
    arm = _arm("x")
    with pytest.raises(ValueError, match="key=value"):
        CampaignOrchestrator._selector_matches("namex", arm)


def test_selector_rejects_unknown_key() -> None:
    arm = _arm("x")
    with pytest.raises(ValueError, match="Unknown selector key"):
        CampaignOrchestrator._selector_matches("color=blue", arm)


# ── _apply_filters end-to-end ──────────────────────────────────────────


def test_no_filter_returns_all(arms: list[CampaignExperimentSchema]) -> None:
    o = CampaignOrchestrator(base_dir=".", dry_run=True)
    assert {a.name for a in o._apply_filters(arms)} == {
        "ref", "abl_a", "abl_b", "base_u", "base_s",
    }


def test_only_filters_by_exact_name(arms: list[CampaignExperimentSchema]) -> None:
    o = CampaignOrchestrator(base_dir=".", dry_run=True, only=["abl_a", "base_u"])
    assert {a.name for a in o._apply_filters(arms)} == {"abl_a", "base_u"}


def test_include_filters_by_role(arms: list[CampaignExperimentSchema]) -> None:
    o = CampaignOrchestrator(
        base_dir=".", dry_run=True, include=["role=ablation"]
    )
    assert {a.name for a in o._apply_filters(arms)} == {"abl_a", "abl_b"}


def test_exclude_filters_by_tag(arms: list[CampaignExperimentSchema]) -> None:
    o = CampaignOrchestrator(
        base_dir=".", dry_run=True, exclude=["tag.arch=geo_mamba"]
    )
    assert {a.name for a in o._apply_filters(arms)} == {"base_u", "base_s"}


def test_multiple_includes_combine_via_or(arms: list[CampaignExperimentSchema]) -> None:
    """A multi-include kept the *union* of matches."""
    o = CampaignOrchestrator(
        base_dir=".",
        dry_run=True,
        include=["role=ablation", "tag.arch=swinir"],
    )
    assert {a.name for a in o._apply_filters(arms)} == {
        "abl_a", "abl_b", "base_s",
    }


def test_only_combines_with_include_via_and(arms: list[CampaignExperimentSchema]) -> None:
    """--only narrows the set; --include narrows further."""
    o = CampaignOrchestrator(
        base_dir=".",
        dry_run=True,
        only=["abl_a", "abl_b", "ref"],
        include=["role=ablation"],
    )
    assert {a.name for a in o._apply_filters(arms)} == {"abl_a", "abl_b"}


def test_exclude_overrides_include(arms: list[CampaignExperimentSchema]) -> None:
    """An arm matched by both include and exclude is dropped."""
    o = CampaignOrchestrator(
        base_dir=".",
        dry_run=True,
        include=["role=ablation"],
        exclude=["name=abl_b"],
    )
    assert {a.name for a in o._apply_filters(arms)} == {"abl_a"}
