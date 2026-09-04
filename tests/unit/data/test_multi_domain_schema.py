"""Phase 4e — MultiDomainConfigSchema validation."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.data import (
    DataConfigSchema,
    DomainConfigSchema,
    MultiDomainConfigSchema,
)


def test_multi_domain_defaults_are_disabled() -> None:
    cfg = MultiDomainConfigSchema()
    assert cfg.enabled is False
    assert cfg.domains == []
    assert cfg.balancing == "round_robin"
    assert cfg.target_supervised is False


def test_multi_domain_enabled_requires_at_least_two_domains() -> None:
    with pytest.raises(ValidationError, match="at least 2 entries"):
        MultiDomainConfigSchema(
            enabled=True,
            domains=[DomainConfigSchema(name="source")],
        )


def test_multi_domain_enabled_rejects_duplicate_names() -> None:
    with pytest.raises(ValidationError, match="unique"):
        MultiDomainConfigSchema(
            enabled=True,
            domains=[
                DomainConfigSchema(name="src"),
                DomainConfigSchema(name="src"),
            ],
        )


def test_balancing_enum_is_closed() -> None:
    with pytest.raises(ValidationError):
        MultiDomainConfigSchema(balancing="random")  # type: ignore[arg-type]


def test_domain_weight_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        DomainConfigSchema(name="x", weight=0.0)


def test_domain_inherits_when_overrides_are_none() -> None:
    """data_root / index_path / dataset_type default to None ⇒ inherit."""
    d = DomainConfigSchema(name="x")
    assert d.data_root is None
    assert d.index_path is None
    assert d.dataset_type is None


def test_data_config_threads_multi_domain_block() -> None:
    cfg = DataConfigSchema(
        multi_domain={
            "enabled": True,
            "domains": [
                {"name": "m4raw", "data_root": "/a"},
                {"name": "fastmri", "data_root": "/b"},
            ],
            "balancing": "stratified",
            "target_supervised": True,
        }
    )
    assert cfg.multi_domain.enabled is True
    assert len(cfg.multi_domain.domains) == 2
    assert cfg.multi_domain.domains[0].name == "m4raw"
    assert cfg.multi_domain.balancing == "stratified"
    assert cfg.multi_domain.target_supervised is True


def test_multi_domain_three_domains_validates() -> None:
    """Multi-target setups (one src + two tgt) must work."""
    cfg = MultiDomainConfigSchema(
        enabled=True,
        domains=[
            DomainConfigSchema(name="src"),
            DomainConfigSchema(name="tgt_a"),
            DomainConfigSchema(name="tgt_b"),
        ],
    )
    assert len(cfg.domains) == 3
