"""The compatibility matrix's rule set and its one-owner boundaries."""

from __future__ import annotations

from spectramr.infrastructure.validation import compatibility_matrix as cm


def test_the_data_to_model_domain_leg_has_one_owner() -> None:
    """Planted violation: a rule named ``domain_chain`` must not be registered.

    ``config_health_checker.check_data_model_compatibility`` folds the declared
    adapter chain and knows which strategies adjoint k-space internally; the
    matrix rule compared the loader's domain to ``input_domain`` with neither and
    rejected 11 arms the health check accepts (2026-09-03).
    """
    names = {getattr(rule, "__name__", "") for rule in cm.list_rules()}
    assert "rule_domain_chain" not in names
    assert not hasattr(cm, "rule_domain_chain")


def test_the_remaining_rules_are_still_registered() -> None:
    names = {getattr(rule, "__name__", "") for rule in cm.list_rules()}
    assert {"rule_required_config_fields", "rule_paired_data", "rule_spatial_rank"} <= names
