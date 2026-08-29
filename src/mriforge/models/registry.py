"""
Model Registry Module.

Provides the central registry and decorator for registering model architectures.
This registry enables O(1) lookup of model classes by name and supports
different training modes (e.g., 'gan', 'diffusion', 'reconstruction').

Capability flags (e.g. ``supports_contrast_conditioning``) let the
audit ladder fail loudly when YAML opts into a feature the chosen
model does not implement — preventing the silent-fallback pitfall
documented in CLAUDE.md.
"""

from typing import Any

from mriforge.config.schemas.enums import Regime, Task
from mriforge.models.capabilities import Domain, ModelCapabilities

MODEL_REGISTRY: dict[str, dict[str, Any]] = {}


def register_model(
    name: str,
    training_mode: str,
    *,
    supports_contrast_conditioning: bool = False,
    supports_vendor_conditioning: bool = False,
    # Capability metadata (Phase 1 of experiment-spec-card design).
    # All default to None ("unannotated"); the audit skips checks for
    # unannotated fields so existing registrations stay green.
    # Domain fields accept either a single Domain literal or a tuple
    # of literals for models that genuinely handle multiple domains.
    spatial_dims: tuple[int, ...] | None = None,
    input_domain: "Domain | tuple[Domain, ...] | None" = None,
    output_domain: "Domain | tuple[Domain, ...] | None" = None,
    accepts_complex: bool | None = None,
    expects_real_imag_interleaved: bool | None = None,
    requires_paired_data: bool | None = None,
    output_field_units: str | None = None,
    trajectory_parametrization: str | None = None,
    override: bool = False,
    workflows: "frozenset[Regime] | None" = None,
    tasks: "frozenset[Task] | None" = None,
):
    """Decorator to register a model class.

    Args:
        name: Unique name of the model.
        training_mode: The training paradigm this model belongs to
            (e.g., 'gan', 'diffusion', 'reconstruction').
        supports_contrast_conditioning: True if the model's ``forward``
            accepts a ``contrast_idx`` (or ``contrast_id``) tensor and
            uses it for FiLM-style conditioning. Set this on every
            generator that participates in Pattern C (multi-contrast)
            training. The Tier-1 audit ``check_multi_contrast_model_support``
            uses this flag to fail loudly when YAML enables
            ``data.multi_contrast`` against a model that ignores the id.
        spatial_dims: Tuple of spatial-dim ranks the model supports
            (e.g. ``(2,)``, ``(3,)``, ``(2, 3)``). When set, the audit
            blocks YAMLs whose data block declares a different rank
            unless an explicit adapter bridges it.
        input_domain: Domain of the input tensor (``image``, ``kspace``,
            ``complex_image``, ``latent``, ``pde_grid``, ``mesh``).
        output_domain: Domain of the output tensor.
        accepts_complex: True if the forward path accepts
            ``torch.complex`` tensors directly.
        expects_real_imag_interleaved: True if the model expects 2C
            real channels representing C complex coils.
        requires_paired_data: True if training requires paired (input,
            target) examples. Cycle/SSL models set False explicitly.
        output_field_units: Physical units of a field-valued output
            (e.g. ``"Hz"`` for a B0/off-resonance map). The field-domain
            metrics + Tier-1 audit read this to enforce the parametrization
            guard (pitfall #16) so a Hz-RMSE metric refuses to grade a model
            whose declared output is an image.
        trajectory_parametrization: Coordinate system of a trajectory-valued
            output (``"spiral"`` / ``"cartesian"`` / ``"radial"``). Blocks a
            spiral-trajectory metric from grading a Cartesian-per-line estimate.
        override: Permit a same-class re-registration to REPLACE a
            capabilities-bearing entry with an empty (all-None) one. Off by
            default so a second bare ``register_model(name, mode)(Cls)`` cannot
            silently clobber a decorator's full ``ModelCapabilities`` (the
            bloch_mamba_v2 scar) — which would disable the audit's
            compatibility checks for that model without any diagnostic.
    """

    capabilities = ModelCapabilities(
        spatial_dims=spatial_dims,
        input_domain=input_domain,
        output_domain=output_domain,
        accepts_complex=accepts_complex,
        expects_real_imag_interleaved=expects_real_imag_interleaved,
        requires_paired_data=requires_paired_data,
        output_field_units=output_field_units,
        trajectory_parametrization=trajectory_parametrization,
        workflows=workflows,
        tasks=tasks,
    )

    def decorator(cls: type[Any]):
        # Per CLAUDE.md #9 and TODO/audit/09_models_registry_generators.md
        # §3.18, refuse to silently overwrite an existing registration
        # with a *different* class. Same-class re-registration (test
        # reloads / idempotent imports) is allowed.
        existing = MODEL_REGISTRY.get(name)
        if existing is not None:
            existing_cls = existing.get("class")
            if existing_cls is not cls:
                raise ValueError(
                    f"Model '{name}' already registered to "
                    f"{existing_cls.__module__}.{existing_cls.__qualname__} "
                    f"(mode={existing.get('mode')!r}); refusing to overwrite "
                    f"with {cls.__module__}.{cls.__qualname__} "
                    f"(mode={training_mode!r}). Rename one of the two registrations."
                )
            # Same class re-registration: refuse a capability DOWNGRADE. A
            # bare second registration with all-None capabilities would
            # silently replace the decorator's declared caps and disable the
            # audit's data/model compatibility checks (the bloch_mamba_v2 scar).
            existing_caps = existing.get("capabilities")
            if (
                not override
                and isinstance(existing_caps, ModelCapabilities)
                and existing_caps != ModelCapabilities()
                and capabilities == ModelCapabilities()
            ):
                raise ValueError(
                    f"Refusing to re-register model '{name}' "
                    f"({cls.__module__}.{cls.__qualname__}) with EMPTY "
                    f"capabilities: it already declares {existing_caps}. A bare "
                    f"re-registration would silently disable the audit's "
                    f"compatibility checks for this model. Remove the redundant "
                    f"registration, or pass override=True to force."
                )

        MODEL_REGISTRY[name] = {
            "class": cls,
            "mode": training_mode,
            "supports_contrast_conditioning": supports_contrast_conditioning,
            "supports_vendor_conditioning": supports_vendor_conditioning,
            "capabilities": capabilities,
        }
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Rejected aspirational model names (Phase 4 of
# TODO/deleted_model_types_reimplementation_plan.md).
#
# These 24 names appeared in the d8ccb8452 deletion ledger with NO implementing
# class anywhere. Unlike the Phase-3 set (glow, blurring_diffusion, ... — now
# implemented) and the Phase-1 aliases (e.g. reversible_network → invertible_
# network), these are rejected: each is either an under-specified umbrella name
# with no anchor paper, a misfiled non-model (operator / procedure), or a
# speculative variant already subsumed by an implemented family. Re-adding any
# of them to VALID_MODEL_TYPES re-creates the "audit-surface lie" the deletion
# fixed. The namespace-axis audit check consults this set to emit the rejection
# rationale as the fix hint instead of a generic "NOT FOUND in registry".
# ---------------------------------------------------------------------------
REJECTED_NAMES: dict[str, str] = {
    # Under-specified umbrella names — no anchor paper, so the name is
    # marketing not specification. File a real impl under the chosen paper's
    # specific name instead.
    "advanced_vae_latent": "under-specified umbrella name; no anchor paper",
    "attention_dense": "under-specified umbrella name; no anchor paper",
    "autoregressive_decoder": "under-specified umbrella name; no anchor paper",
    "contrastive_network": "under-specified umbrella name; no anchor paper",
    "dense_prediction_unet": "under-specified umbrella name; no anchor paper",
    "dynamic_unet": "under-specified umbrella name; no anchor paper",
    "hyperspherical_network": "under-specified; use hyperspherical_vae instead",
    "mesh_mri": "under-specified umbrella name; no anchor paper",
    "uncertain_pyramid": "under-specified umbrella name; no anchor paper",
    "universal_adapter": "under-specified umbrella name; no anchor paper",
    "student_unet": "under-specified distillation stub; no anchor paper",
    # Misfiled non-models — an operator or a procedure, not an architecture.
    "max_pool": "a pooling operator, not a model architecture",
    "pareto_optimization": "a multi-objective procedure, not a model",
    # Subsumed by existing primitives / losses.
    "epistemic_aleatoric": "covered by evidential + uncertainty losses on heads",
    # Speculative GAN/VAE variants — no anchor paper; implemented families
    # (glow, progressive_gan, hierarchical_vq_vae, hyperspherical_vae, moe_vae,
    # beta_vae_gan) cover the scientifically-grounded cases.
    "score_based_gan": "speculative variant; no anchor paper",
    "sn_gan": "spectral-norm is a layer flag, not a distinct model",
    "unet_gan": "speculative variant; use a discriminator + UNet generator",
    "topographic_vae": "speculative variant; no anchor paper",
    "recursive_cascade": "speculative variant; no anchor paper",
    "recursive_residual": "speculative variant; no anchor paper",
    "wasserstein_vae": "speculative variant; no anchor paper",
    # TRELLIS speculative variants — the implemented trellis_*_vae set covers
    # the structured-latent family; these have no formulation.
    "trellis_diffusion": "speculative TRELLIS variant; no formulation",
    "trellis_image_large": "speculative TRELLIS variant; no formulation",
    "trellis_volume": "speculative TRELLIS variant; no formulation",
}


def get_model_class(name: str) -> type[Any]:
    """get_model_class.

    Args:
        name (str): Description.
    Returns:
        type[Any]: Description.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{name}' not found in registry. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name]["class"]


def get_model_mode(name: str) -> str:
    """get_model_mode.

    Args:
        name (str): Description.
    Returns:
        str: Description.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in registry.")
    return MODEL_REGISTRY[name]["mode"]


def model_supports(name: str, capability: str) -> bool:
    """Return True if the registered model declares the given capability.

    Falls back to False for unknown capability flags so future flags
    don't crash callers that haven't been updated yet.
    """
    entry = MODEL_REGISTRY.get(name)
    if entry is None:
        return False
    return bool(entry.get(capability, False))


def get_model_capabilities(name: str) -> ModelCapabilities | None:
    """Return ``ModelCapabilities`` for a registered model, or None.

    Returns None for both unknown models and models whose decorator
    did not set any capability fields. Callers MUST distinguish between
    "unannotated" and "annotated as not-supported" — the latter has a
    populated dataclass, the former returns None.
    """
    entry = MODEL_REGISTRY.get(name)
    if entry is None:
        return None
    caps = entry.get("capabilities")
    if not isinstance(caps, ModelCapabilities):
        return None
    # Treat fully-default dataclass as "unannotated" so the audit skips it.
    if all(getattr(caps, f) is None for f in caps.__dataclass_fields__):
        return None
    return caps


def list_models() -> dict[str, dict[str, Any]]:
    """Return dictionary of all registered models."""
    return MODEL_REGISTRY.copy()


def list_models_with_capability(capability: str) -> list[str]:
    """Return all registered model names that declare ``capability=True``."""
    return [n for n, e in MODEL_REGISTRY.items() if e.get(capability, False)]


# Note: Model discovery is handled in src/models/init_registry.py to avoid circular imports.
