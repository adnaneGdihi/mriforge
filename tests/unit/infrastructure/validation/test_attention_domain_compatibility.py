"""Tests for ConfigHealthChecker.check_attention_domain_compatibility.

Uses lightweight SimpleNamespace config stubs (the convention in
test_health_checker_json_and_new_checks.py) to avoid constructing a full
frozen TrainingSettings.
"""

import types

import pytest

import mriforge.models.blocks.attention_domains as ad
from mriforge.infrastructure.validation.config_health_checker import (
    ConfigHealthChecker,
    HealthCheckResult,
)

pytestmark = pytest.mark.unit


def _cfg(model_type="kspace_cold_diffusion", **model_kwargs):
    return types.SimpleNamespace(
        model=types.SimpleNamespace(
            model_type=model_type, model_kwargs=model_kwargs
        )
    )


@pytest.fixture
def checker():
    return ConfigHealthChecker()


def _one(checker, cfg) -> HealthCheckResult:
    res = checker.check_attention_domain_compatibility(cfg)
    assert isinstance(res, list) and len(res) == 1
    return res[0]


class TestPasses:
    def test_kan_complex_unet_kspace_passes(self, checker):
        r = _one(
            checker,
            _cfg(
                attention_type="kan_dual_domain",
                backbone_type="complex_unet",
                force_pure_kspace=True,
            ),
        )
        assert r.passed and r.severity == "info"

    def test_dual_domain_complex_unet_passes(self, checker):
        r = _one(
            checker,
            _cfg(
                attention_type="dual_domain",
                backbone_type="complex_unet",
                force_pure_kspace=True,
            ),
        )
        assert r.passed

    def test_imgdomain_ablation_passes(self, checker):
        """fpk=False -> feature_domain=image, still supported."""
        r = _one(
            checker,
            _cfg(
                attention_type="dual_domain",
                backbone_type="complex_unet",
                force_pure_kspace=False,
            ),
        )
        assert r.passed

    def test_none_on_pure_kspace_unet_passes(self, checker):
        r = _one(
            checker,
            _cfg(
                attention_type="none",
                backbone_type="unet",
                force_pure_kspace=True,
            ),
        )
        assert r.passed

    def test_non_kspace_model_skips(self, checker):
        r = _one(checker, _cfg(model_type="unet_gan", attention_type="self"))
        assert r.passed and "not a kspace_cold_diffusion" in r.message

    def test_generator_alias_is_recognised(self, checker):
        """The active/ arm uses model_type=kspace_cold_diffusion_generator."""
        r = _one(
            checker,
            _cfg(
                model_type="kspace_cold_diffusion_generator",
                attention_type="none",
                backbone_type="unet",
                force_pure_kspace=True,
            ),
        )
        assert r.passed


class TestR1DeadAttention:
    def test_self_on_pure_kspace_unet_errors(self, checker):
        r = _one(
            checker,
            _cfg(
                attention_type="self",
                backbone_type="unet",
                force_pure_kspace=True,
            ),
        )
        assert not r.passed
        assert r.severity == "error"
        assert r.category == "attention_domain"
        assert "PureKSpaceUNet" in r.message
        assert "model.model_kwargs.attention_type" in r.yaml_keys
        assert r.fix_hint and "none" in r.fix_hint

    def test_default_attention_on_pure_kspace_unet_errors(self, checker):
        """Constructor default is 'self', so an omitted key must also fail."""
        r = _one(
            checker,
            _cfg(backbone_type="unet", force_pure_kspace=True),
        )
        assert not r.passed and r.severity == "error"


class TestR2DispatchCoverage:
    def test_spatial_on_complex_unet_errors(self, checker):
        r = _one(
            checker,
            _cfg(
                attention_type="spatial",
                backbone_type="complex_unet",
                force_pure_kspace=True,
            ),
        )
        assert not r.passed and r.severity == "error"
        assert "up-block" in r.message


class TestR0UnknownType:
    def test_unknown_attention_errors(self, checker):
        r = _one(
            checker,
            _cfg(
                attention_type="quantum",
                backbone_type="complex_unet",
                force_pure_kspace=True,
            ),
        )
        assert not r.passed and r.severity == "error"
        assert "advertised" in r.message


class TestR3DomainRatchet:
    def test_single_domain_attention_rejects_wrong_domain(self, checker, monkeypatch):
        """A future single-domain attention must be rejected for the other domain."""
        patched = dict(ad.ATTENTION_DOMAIN_SUPPORT)
        patched["kan_dual_domain"] = frozenset({"image"})
        monkeypatch.setattr(ad, "ATTENTION_DOMAIN_SUPPORT", patched)
        r = _one(
            checker,
            _cfg(
                attention_type="kan_dual_domain",
                backbone_type="complex_unet",
                force_pure_kspace=True,  # -> kspace, unsupported
            ),
        )
        assert not r.passed and r.severity == "error"
        assert "does not support feature_domain" in r.message

    def test_single_domain_attention_accepts_right_domain(self, checker, monkeypatch):
        patched = dict(ad.ATTENTION_DOMAIN_SUPPORT)
        patched["kan_dual_domain"] = frozenset({"image"})
        monkeypatch.setattr(ad, "ATTENTION_DOMAIN_SUPPORT", patched)
        r = _one(
            checker,
            _cfg(
                attention_type="kan_dual_domain",
                backbone_type="complex_unet",
                force_pure_kspace=False,  # -> image, supported
            ),
        )
        assert r.passed


class TestWiredIntoReport:
    def test_wired_into_run_all_checks(self, checker):
        """The check must be invoked by the full audit sweep.

        run_all_checks needs a complete TrainingSettings, so rather than
        stubbing every section we assert the wiring at the source level (the
        method is referenced inside run_all_checks).
        """
        import inspect

        src = inspect.getsource(checker.run_all_checks)
        assert "check_attention_domain_compatibility" in src
