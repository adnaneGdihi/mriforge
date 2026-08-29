"""Tests for the NRMetricConfig schema (mriforge.config.schemas.nr_metrics)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mriforge.config.schemas.metrics import MetricsConfigSchema
from mriforge.config.schemas.nr_metrics import NRMetricConfig


def test_defaults() -> None:
    cfg = NRMetricConfig()
    assert cfg.enabled_metrics == ()
    assert cfg.enable_prior_geometry is False
    assert cfg.gri_target == 0.09
    assert cfg.prior_ckpt is None


def test_research_mode_default_true() -> None:
    # The battery is validation-pending: research_mode defaults to True so a
    # config self-describes that these metrics are offline-only.
    assert NRMetricConfig().research_mode is True
    assert NRMetricConfig(research_mode=False).research_mode is False


def test_frozen_and_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        NRMetricConfig(not_a_field=1)  # type: ignore[call-arg]
    cfg = NRMetricConfig()
    with pytest.raises(ValidationError):
        cfg.gri_target = 0.5  # type: ignore[misc]


def test_effective_enabled_drops_gated_when_disabled() -> None:
    cfg = NRMetricConfig(enabled_metrics=("ndcr", "dpst", "hfrw", "pmmd"))
    # enable_prior_geometry=False ⇒ dpst/pmmd dropped
    assert set(cfg.effective_enabled()) == {"ndcr", "hfrw"}


def test_effective_enabled_keeps_gated_when_enabled() -> None:
    cfg = NRMetricConfig(
        enabled_metrics=("ndcr", "dpst"), enable_prior_geometry=True
    )
    assert set(cfg.effective_enabled()) == {"ndcr", "dpst"}


def test_as_nr_params_has_tunables_not_assets() -> None:
    params = NRMetricConfig(gri_target=0.12).as_nr_params()
    assert params["gri_target"] == 0.12
    assert "psdc_beta_ref" in params
    # asset paths / gating are NOT in the flat params dict
    assert "prior_ckpt" not in params
    assert "enable_prior_geometry" not in params


def test_composed_into_metrics_schema() -> None:
    m = MetricsConfigSchema(nr=NRMetricConfig(enabled_metrics=("ndcr", "hfrw")))
    assert m.nr.enabled_metrics == ("ndcr", "hfrw")
    # default factory present when omitted
    assert MetricsConfigSchema().nr.enabled_metrics == ()


def test_prior_geometry_keys_is_class_var_not_field() -> None:
    """Regression WS1-core-06: PRIOR_GEOMETRY_KEYS is an internal lookup table,
    not a user-facing knob. Without ``ClassVar`` Pydantic v2 treats it as a model
    field — it leaks into model_fields / model_dump() / JSON schema / provenance.
    """
    assert "PRIOR_GEOMETRY_KEYS" not in NRMetricConfig.model_fields
    assert "PRIOR_GEOMETRY_KEYS" not in NRMetricConfig().model_dump()
    # still accessible as a class constant (effective_enabled() reads it)
    assert isinstance(NRMetricConfig.PRIOR_GEOMETRY_KEYS, tuple)
    assert "pmmd" in NRMetricConfig.PRIOR_GEOMETRY_KEYS
