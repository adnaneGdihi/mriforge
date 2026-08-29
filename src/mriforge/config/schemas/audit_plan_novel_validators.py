"""Tier-1 audit checks for the seven novel research arms.

Each check is a function with the standard validator signature
``(config_dict) -> list[str]``: an empty list means PASS, a non-empty
list contains user-facing error / warning messages. Register them by
calling :func:`register_audit_plan_novel_validators(registry)` from
the validator-registry bootstrap.

These checks complement Pydantic schema validation (Tier 0) by
catching *cross-section* inconsistencies — for example, a
``hermitian_fno`` model whose ``output_domain`` is configured as
``image`` (it must be ``kspace`` for the equivariance proof to hold).

Each rule is associated with one of the seven ideas from
``TODO/audit/audit_plan_novel.md``.
"""

from __future__ import annotations

from typing import Any

from mriforge.config.schemas.validator_registry import (
    ValidationRule,
    ValidatorRegistry,
)

# --------------------------------------------------------------------- #
# Idea 1 — Hermitian-equivariant model must produce k-space output
# --------------------------------------------------------------------- #


_HERMITIAN_MODEL_TYPES = {"hermitian_fno", "hermitian_kspace_unet"}


def _expose_flag(config: dict[str, Any], leaf: str) -> bool:
    """Read a `data.expose.<leaf>` flag under either spelling.

    The flat `data.expose_<leaf>` spelling is a retired rename record: arms in
    `experiments/inprogress/` were drained to `data.expose.<leaf>` and the flat
    form now raises at load. Reading only the flat name made these validators
    warn on arms that DO set the flag, and told the author to set a spelling
    that no longer loads. The other corpus roots are un-drained, so both
    spellings still have to resolve.
    """
    data_cfg = config.get("data") or {}
    expose = data_cfg.get("expose")
    if isinstance(expose, dict) and leaf in expose:
        return bool(expose[leaf])
    return bool(data_cfg.get(f"expose_{leaf}", False))


def _validate_hermitian_model_output_domain(config: dict[str, Any]) -> list[str]:
    model_type = config.get("model", {}).get("model_type")
    if model_type not in _HERMITIAN_MODEL_TYPES:
        return []
    out_dom = config.get("training", {}).get("output_domain") or config.get("losses", {}).get(
        "output_domain"
    )
    # Equivariance holds under the centered-FFT round-trip; image-domain
    # output silently breaks the invariant. Soft-warn rather than error
    # because the user may intentionally use the model as an image-domain
    # feature extractor with a separate FFT layer downstream.
    if out_dom not in {"kspace", "complex_image", None}:
        return [
            f"Model type {model_type!r} is Hermitian-equivariant and was "
            f"designed for k-space output, but training.output_domain is "
            f"{out_dom!r}. The equivariance invariant only holds across "
            f"a centered-FFT round trip; using image-domain output without "
            f"an explicit FFT wrap breaks the proof. "
            f"Set output_domain: 'kspace' or wrap the model with an "
            f"explicit fft2c / ifft2c pair."
        ]
    return []


# --------------------------------------------------------------------- #
# Idea 2 — Score-field tomography requires data.expose_acquisition_params
# --------------------------------------------------------------------- #


def _validate_acquisition_params_for_score_field(config: dict[str, Any]) -> list[str]:
    mode = config.get("training", {}).get("training_mode")
    if mode != "score_field_tomography":
        return []
    expose = _expose_flag(config, "acquisition_params")
    if not expose:
        return [
            "training_mode='score_field_tomography' requires "
            "data.expose.acquisition_params=true so that the strategy "
            "receives (TR, TE, TI, α, B0) on every batch. "
            "Set data.expose.acquisition_params: true."
        ]
    return []


# --------------------------------------------------------------------- #
# Idea 3 — Bloch-equivariant translation must have an atlas with WM/GM/CSF
# --------------------------------------------------------------------- #


_REQUIRED_ATLAS_KEYS = ("t1_scale_ulf", "t2_scale_ulf", "lambda_bloch_eq")


def _validate_bloch_eq_atlas_coverage(config: dict[str, Any]) -> list[str]:
    training = config.get("training", {})
    if training.get("training_mode") != "bloch_equivariant_translation":
        return []
    bet = training.get("bloch_equivariant_translation", {}) or {}
    missing = [k for k in _REQUIRED_ATLAS_KEYS if k not in bet]
    if missing:
        return [
            f"training.bloch_equivariant_translation is missing required atlas "
            f"keys: {missing}. The Bloch-equivariance residual evaluates to "
            f"zero if the atlas is empty, allowing the model to ignore the "
            f"unpaired branch entirely (audit_plan_novel §Phase 3 Idea 3)."
        ]
    return []


# --------------------------------------------------------------------- #
# Idea 4 — Information-bottleneck nonlinear mode requires a registered
#          generative prior; the linear-Gaussian mode is the safe default.
# --------------------------------------------------------------------- #


def _validate_ib_active_acquisition_prior(config: dict[str, Any]) -> list[str]:
    training = config.get("training", {})
    if training.get("training_mode") not in {"ib_active_acquisition", "ib_bald"}:
        return []
    ib = training.get("ib_active_acquisition", {}) or {}
    mode = ib.get("mode", "linear_gaussian")
    if mode == "nonlinear":
        model_type = config.get("model", {}).get("model_type", "")
        # A nonlinear IB run needs a diffusion-style prior whose score is
        # callable. We check the model_type substring as a first-order
        # heuristic; a stricter check would query the registry's
        # ModelCapabilities — left for a follow-up PR.
        prior_ok = any(tok in model_type for tok in ("diffusion", "score", "edm", "flow"))
        if not prior_ok:
            return [
                f"training.ib_active_acquisition.mode='nonlinear' requires a "
                f"registered diffusion / score model to supply the Fisher "
                f"trace estimator, but model.model_type={model_type!r} does "
                f"not look like one. Either switch mode to 'linear_gaussian' "
                f"or set model.model_type to a registered score-based prior."
            ]
    return []


# --------------------------------------------------------------------- #
# Idea 5 — Riemannian Bloch diffusion needs a manifold-compatible model
#          and a sensible step size relative to the injectivity radius.
# --------------------------------------------------------------------- #


def _validate_riemannian_bloch_diffusion(config: dict[str, Any]) -> list[str]:
    training = config.get("training", {})
    if training.get("training_mode") != "riemannian_bloch_diffusion":
        return []
    model_type = config.get("model", {}).get("model_type", "")
    if model_type != "riemannian_diffusion_qmap":
        return [
            f"training_mode='riemannian_bloch_diffusion' expects "
            f"model.model_type='riemannian_diffusion_qmap', got {model_type!r}. "
            f"The reverse SDE relies on the model's tangent-projection layer; "
            f"plugging in a Euclidean-output model produces drift off the "
            f"manifold within a few steps."
        ]
    rbd = training.get("riemannian_bloch_diffusion", {}) or {}
    step = float(rbd.get("step_size_init", 0.05))
    safety = float(rbd.get("injectivity_safety_factor", 0.5))
    if step > safety:
        return [
            f"training.riemannian.step_size_init={step:.3f} exceeds "
            f"injectivity_safety_factor={safety:.3f}. Geodesic steps larger "
            f"than the injectivity radius can wrap around the manifold and "
            f"produce non-monotone exp/log maps; reduce step_size_init."
        ]
    return []


# --------------------------------------------------------------------- #
# Idea 6 — Physics-equivariant SSL: linear latent operators recommended
# --------------------------------------------------------------------- #


def _validate_phys_eq_ssl_linear_latent_operator(config: dict[str, Any]) -> list[str]:
    training = config.get("training", {})
    if training.get("training_mode") not in {"physics_equivariant_ssl", "phys_eq_ssl"}:
        return []
    pe = training.get("physics_eq_ssl", {}) or {}
    op_type = pe.get("latent_operator_type", "linear")
    if op_type != "linear":
        return [
            f"training.physics_eq_ssl.latent_operator_type={op_type!r} "
            f"breaks the intertwiner-identifiability argument "
            f"(audit_plan_novel §Phase 2 Idea 6); the linear latent "
            f"operator is the documented default. Set the field to "
            f"'linear' or accept that the recovered operator is no longer "
            f"identifiable up to orthogonal change of basis."
        ]
    return []


# --------------------------------------------------------------------- #
# Idea 7 — KSD audit: opt-in, model must support a callable score
# --------------------------------------------------------------------- #


def _validate_ksd_audit_compatibility(config: dict[str, Any]) -> list[str]:
    audit = config.get("audit", {}) or {}
    ksd = audit.get("ksd", {}) or {}
    if not ksd.get("enabled", False):
        return []
    mode = config.get("training", {}).get("training_mode", "")
    # KSD audits are only meaningful for generative-prior strategies.
    generative_modes = {
        "diffusion",
        "kspace_cold_diffusion",
        "cold_diffusion",
        "graph_cold_diffusion",
        "rectified_flow",
        "flow_matching",
        "edm",
        "score_field_tomography",
        "noise_adaptive_score",
        "spin_sde",
        "stochastic_interpolants",
        "pnp",
        "pnp_red",
        "tto",
        "test_time_optimization",
        "riemannian_bloch_diffusion",
    }
    if mode not in generative_modes:
        return [
            f"audit.ksd.enabled=true but training_mode={mode!r} does not "
            f"expose a callable score. KSD goodness-of-fit only makes "
            f"sense for generative / score-based strategies "
            f"({sorted(generative_modes)[:5]}...). "
            f"Either disable audit.ksd or use a compatible training_mode."
        ]
    return []


# --------------------------------------------------------------------- #
# Registry hook
# --------------------------------------------------------------------- #


_RULES = [
    ValidationRule(
        name="hermitian_model_output_domain",
        category="audit_plan_novel",
        description=(
            "Hermitian-equivariant models must produce k-space output for the "
            "equivariance invariant to hold (Idea 1)."
        ),
        validator_fn=_validate_hermitian_model_output_domain,
        severity="warning",
        applies_to_models=list(_HERMITIAN_MODEL_TYPES),
    ),
    ValidationRule(
        name="acquisition_params_required_for_score_field",
        category="audit_plan_novel",
        description=(
            "Score-field tomography requires data.expose.acquisition_params=true (Idea 2)."
        ),
        validator_fn=_validate_acquisition_params_for_score_field,
        severity="error",
        applies_to=["score_field_tomography"],
    ),
    ValidationRule(
        name="bloch_eq_atlas_coverage",
        category="audit_plan_novel",
        description=(
            "Bloch-equivariant translation atlas must declare T1/T2 scaling and λ_eq (Idea 3)."
        ),
        validator_fn=_validate_bloch_eq_atlas_coverage,
        severity="warning",
        applies_to=["bloch_equivariant_translation"],
    ),
    ValidationRule(
        name="active_acquisition_requires_prior_or_linear_gaussian",
        category="audit_plan_novel",
        description=("Nonlinear IB-acquisition requires a registered score-based prior (Idea 4)."),
        validator_fn=_validate_ib_active_acquisition_prior,
        severity="error",
        applies_to=["ib_active_acquisition", "ib_bald"],
    ),
    ValidationRule(
        name="riemannian_bloch_diffusion_step_size",
        category="audit_plan_novel",
        description=(
            "Riemannian step size must stay below the injectivity safety factor (Idea 5)."
        ),
        validator_fn=_validate_riemannian_bloch_diffusion,
        severity="error",
        applies_to=["riemannian_bloch_diffusion"],
    ),
    ValidationRule(
        name="phys_eq_ssl_linear_latent_operator_default",
        category="audit_plan_novel",
        description=("Operator-equivariant SSL latent operator should be linear (Idea 6)."),
        validator_fn=_validate_phys_eq_ssl_linear_latent_operator,
        severity="warning",
        applies_to=["physics_equivariant_ssl", "phys_eq_ssl"],
    ),
    ValidationRule(
        name="ksd_audit_requires_generative_prior",
        category="audit_plan_novel",
        description=("KSD audit only meaningful for generative-prior strategies (Idea 7)."),
        validator_fn=_validate_ksd_audit_compatibility,
        severity="warning",
    ),
]


def register_audit_plan_novel_validators(registry: ValidatorRegistry) -> None:
    """Register all seven Tier-1 checks on the given validator registry.

    Idempotent — re-registering an existing rule is a no-op (the
    registry's ``register`` raises on duplicates with a different rule).
    """
    for rule in _RULES:
        if rule.name not in registry._validators:
            registry.register(rule)


__all__ = [
    "register_audit_plan_novel_validators",
]
