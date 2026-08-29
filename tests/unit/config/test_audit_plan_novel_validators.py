"""Unit tests for the audit_plan_novel Tier-1 validators."""

from __future__ import annotations

import pytest

from mriforge.config.schemas.audit_plan_novel_validators import (
    _validate_acquisition_params_for_score_field,
    _validate_bloch_eq_atlas_coverage,
    _validate_hermitian_model_output_domain,
    _validate_ib_active_acquisition_prior,
    _validate_ksd_audit_compatibility,
    _validate_phys_eq_ssl_linear_latent_operator,
    _validate_riemannian_bloch_diffusion,
)


# ---- Idea 1 -------------------------------------------------------- #


def test_hermitian_model_with_kspace_output_passes() -> None:
    cfg = {
        "model": {"model_type": "hermitian_fno"},
        "training": {"output_domain": "kspace"},
    }
    assert _validate_hermitian_model_output_domain(cfg) == []


def test_hermitian_model_with_image_output_warns() -> None:
    cfg = {
        "model": {"model_type": "hermitian_fno"},
        "training": {"output_domain": "image"},
    }
    msgs = _validate_hermitian_model_output_domain(cfg)
    assert msgs and "Hermitian-equivariant" in msgs[0]


def test_non_hermitian_model_ignored() -> None:
    cfg = {
        "model": {"model_type": "enhanced_deep_unet"},
        "training": {"output_domain": "image"},
    }
    assert _validate_hermitian_model_output_domain(cfg) == []


# ---- Idea 2 -------------------------------------------------------- #


def test_score_field_requires_acquisition_params() -> None:
    cfg_missing = {
        "training": {"training_mode": "score_field_tomography"},
        "data": {"expose_acquisition_params": False},
    }
    msgs = _validate_acquisition_params_for_score_field(cfg_missing)
    # The remedy names the CANONICAL spelling. The flat `expose_acquisition_params`
    # is a retired rename record now, so telling the author to set it would be a
    # fix that does not load.
    assert msgs and "data.expose.acquisition_params" in msgs[0]


def test_score_field_with_acquisition_params_passes() -> None:
    cfg = {
        "training": {"training_mode": "score_field_tomography"},
        "data": {"expose_acquisition_params": True},
    }
    assert _validate_acquisition_params_for_score_field(cfg) == []


# ---- Idea 3 -------------------------------------------------------- #


def test_bloch_eq_missing_atlas_keys_warns() -> None:
    cfg = {
        "training": {
            "training_mode": "bloch_equivariant_translation",
            "bloch_equivariant_translation": {},
        }
    }
    msgs = _validate_bloch_eq_atlas_coverage(cfg)
    assert msgs and "atlas" in msgs[0].lower()


def test_bloch_eq_complete_atlas_passes() -> None:
    cfg = {
        "training": {
            "training_mode": "bloch_equivariant_translation",
            "bloch_equivariant_translation": {
                "t1_scale_ulf": 0.3,
                "t2_scale_ulf": 1.05,
                "lambda_bloch_eq": 1.0,
            },
        }
    }
    assert _validate_bloch_eq_atlas_coverage(cfg) == []


# ---- Idea 4 -------------------------------------------------------- #


def test_ib_linear_gaussian_always_ok() -> None:
    cfg = {
        "training": {
            "training_mode": "ib_active_acquisition",
            "ib_active_acquisition": {"mode": "linear_gaussian"},
        },
        "model": {"model_type": "enhanced_deep_unet"},
    }
    assert _validate_ib_active_acquisition_prior(cfg) == []


def test_ib_nonlinear_without_prior_errors() -> None:
    cfg = {
        "training": {
            "training_mode": "ib_active_acquisition",
            "ib_active_acquisition": {"mode": "nonlinear"},
        },
        "model": {"model_type": "enhanced_deep_unet"},
    }
    msgs = _validate_ib_active_acquisition_prior(cfg)
    assert msgs and "diffusion" in msgs[0].lower()


def test_ib_nonlinear_with_diffusion_prior_passes() -> None:
    cfg = {
        "training": {
            "training_mode": "ib_active_acquisition",
            "ib_active_acquisition": {"mode": "nonlinear"},
        },
        "model": {"model_type": "score_field_unet"},
    }
    assert _validate_ib_active_acquisition_prior(cfg) == []


# ---- Idea 5 -------------------------------------------------------- #


def test_riemannian_step_too_large_errors() -> None:
    cfg = {
        "training": {
            "training_mode": "riemannian_bloch_diffusion",
            "riemannian_bloch_diffusion": {
                "step_size_init": 0.9,
                "injectivity_safety_factor": 0.5,
            },
        },
        "model": {"model_type": "riemannian_diffusion_qmap"},
    }
    msgs = _validate_riemannian_bloch_diffusion(cfg)
    assert msgs and "step_size_init" in msgs[0]


def test_riemannian_wrong_model_errors() -> None:
    cfg = {
        "training": {"training_mode": "riemannian_bloch_diffusion"},
        "model": {"model_type": "enhanced_deep_unet"},
    }
    msgs = _validate_riemannian_bloch_diffusion(cfg)
    assert msgs and "riemannian_diffusion_qmap" in msgs[0]


# ---- Idea 6 -------------------------------------------------------- #


def test_phys_eq_ssl_linear_passes() -> None:
    cfg = {
        "training": {
            "training_mode": "physics_equivariant_ssl",
            "physics_eq_ssl": {"latent_operator_type": "linear"},
        }
    }
    assert _validate_phys_eq_ssl_linear_latent_operator(cfg) == []


def test_phys_eq_ssl_mlp_warns() -> None:
    cfg = {
        "training": {
            "training_mode": "physics_equivariant_ssl",
            "physics_eq_ssl": {"latent_operator_type": "mlp"},
        }
    }
    msgs = _validate_phys_eq_ssl_linear_latent_operator(cfg)
    assert msgs and "linear" in msgs[0]


# ---- Idea 7 -------------------------------------------------------- #


def test_ksd_audit_requires_generative_mode() -> None:
    cfg = {
        "audit": {"ksd": {"enabled": True}},
        "training": {"training_mode": "reconstruction"},
    }
    msgs = _validate_ksd_audit_compatibility(cfg)
    assert msgs and "generative" in msgs[0].lower()


def test_ksd_audit_with_diffusion_passes() -> None:
    cfg = {
        "audit": {"ksd": {"enabled": True}},
        "training": {"training_mode": "diffusion"},
    }
    assert _validate_ksd_audit_compatibility(cfg) == []


def test_ksd_disabled_always_passes() -> None:
    cfg = {
        "audit": {"ksd": {"enabled": False}},
        "training": {"training_mode": "reconstruction"},
    }
    assert _validate_ksd_audit_compatibility(cfg) == []


# ---- registry bootstrap ------------------------------------------- #


def test_register_audit_plan_novel_validators_idempotent() -> None:
    from mriforge.config.schemas.audit_plan_novel_validators import (
        register_audit_plan_novel_validators,
    )
    from mriforge.config.schemas.validator_registry import (
        ValidatorRegistry,
        get_validator_registry,
    )

    r1 = get_validator_registry()
    n_before = len(r1._validators)
    register_audit_plan_novel_validators(r1)
    n_after = len(r1._validators)
    assert n_after == n_before  # idempotent — already registered at bootstrap

    fresh = ValidatorRegistry()
    register_audit_plan_novel_validators(fresh)
    assert any(
        name.startswith("hermitian_model") or name.startswith("bloch_eq")
        for name in fresh._validators
    )


class TestExposeFlagsFollowTheDrain:
    """These validators read `data.expose_*`, which is now a retired spelling.

    `experiments/inprogress/` was drained to `data.expose.<leaf>` and the flat
    form raises at load. A validator reading only the flat name does not go
    quiet here -- it goes WRONG in the loud direction: it reports the flag as
    unset on an arm that sets it, and its remedy text tells the author to
    declare a key that no longer loads.

    The other corpus roots (active/, validated/, training/) are un-drained, so
    both spellings must still resolve.
    """

    @staticmethod
    def _cfg(expose_block: dict, **extra) -> dict:
        cfg = {"data": expose_block, "training": {"training_mode": "score_field_tomography"}}
        cfg.update(extra)
        return cfg

    def test_the_canonical_spelling_satisfies_the_check(self) -> None:
        from mriforge.config.schemas.audit_plan_novel_validators import (
            _validate_acquisition_params_for_score_field,
        )

        cfg = self._cfg({"expose": {"acquisition_params": True}})
        assert _validate_acquisition_params_for_score_field(cfg) == []

    def test_the_legacy_spelling_still_satisfies_it(self) -> None:
        from mriforge.config.schemas.audit_plan_novel_validators import (
            _validate_acquisition_params_for_score_field,
        )

        cfg = self._cfg({"expose_acquisition_params": True})
        assert _validate_acquisition_params_for_score_field(cfg) == []

    def test_neither_spelling_still_reports(self) -> None:
        """Anti-vacuity. A reader that returned True unconditionally would
        satisfy both tests above and silence the check for everyone."""
        from mriforge.config.schemas.audit_plan_novel_validators import (
            _validate_acquisition_params_for_score_field,
        )

        findings = _validate_acquisition_params_for_score_field(self._cfg({}))
        assert findings, "the check must still fire when the flag is absent"

    def test_the_remedy_names_a_spelling_that_loads(self) -> None:
        """The message told the author to set `data.expose_acquisition_params`,
        which now raises. A remedy that cannot be followed is worse than none."""
        from mriforge.config.schemas.audit_plan_novel_validators import (
            _validate_acquisition_params_for_score_field,
        )

        findings = _validate_acquisition_params_for_score_field(self._cfg({}))
        joined = " ".join(findings)
        assert "data.expose.acquisition_params" in joined
        assert "data.expose_acquisition_params" not in joined

