"""Loss Domain Validation - Phase 2 Runtime Domain Checking.

Validates that loss functions match the input domain at runtime (on bootstrap).
Prevents silent domain mismatches like HFEN (image-domain) receiving k-space data.

Integrates with loss_audit.py for comprehensive validation.
"""

import logging
from dataclasses import dataclass

from spectramr.config.schemas.loss import LossConfigSchema
from spectramr.infrastructure.training.builders.loss_builder import LossBuilder

logger = logging.getLogger(__name__)


# Domain Metadata Registry (Phase 3 - will be extended)
LOSS_DOMAIN_REGISTRY = {
    # K-Space native
    "complex_l1": {"domain": "kspace", "compatible_with": ["kspace", "agnostic"]},
    "weighted_kspace_l1": {
        "domain": "kspace",
        "compatible_with": ["kspace", "agnostic"],
    },
    "data_consistency": {
        "domain": "kspace",
        "compatible_with": ["kspace", "agnostic"],
    },
    "parallel_imaging_kspace": {
        "domain": "kspace",
        "compatible_with": ["kspace", "agnostic"],
    },
    "spectral_kspace": {"domain": "kspace", "compatible_with": ["kspace", "agnostic"]},
    "frequency_weighted_l1_kspace": {
        "domain": "kspace",
        "compatible_with": ["kspace", "agnostic"],
    },
    "background_suppression": {
        "domain": "kspace",
        "compatible_with": ["kspace", "agnostic"],
    },
    # Image-domain only (DANGEROUS with k-space)
    "hfen": {"domain": "image", "compatible_with": ["image", "agnostic"]},
    "ssim": {"domain": "image", "compatible_with": ["image", "agnostic"]},
    "ms_ssim": {"domain": "image", "compatible_with": ["image", "agnostic"]},
    "perceptual": {"domain": "image", "compatible_with": ["image", "agnostic"]},
    "lpips": {"domain": "image", "compatible_with": ["image", "agnostic"]},
    "dino_perceptual": {"domain": "image", "compatible_with": ["image", "agnostic"]},
    "mind_ssc": {"domain": "image", "compatible_with": ["image", "agnostic"]},
    "sobel_edge": {"domain": "image", "compatible_with": ["image", "agnostic"]},
    "edge_consistency": {"domain": "image", "compatible_with": ["image", "agnostic"]},
    "histogram_consistency": {
        "domain": "image",
        "compatible_with": ["image", "agnostic"],
    },
    "topological": {"domain": "image", "compatible_with": ["image", "agnostic"]},
    # Physics-informed
    "bloch_residual": {
        "domain": "physics",
        "compatible_with": ["kspace", "image", "agnostic"],
    },
    "non_cartesian_graph": {
        "domain": "physics",
        "compatible_with": ["kspace", "image", "agnostic"],
    },
    "graph_consistency": {
        "domain": "physics",
        "compatible_with": ["kspace", "image", "agnostic"],
    },
    # Diffusion-agnostic
    "diffusion_mse": {"domain": "agnostic", "compatible_with": ["kspace", "image"]},
    "diffusion_velocity": {
        "domain": "agnostic",
        "compatible_with": ["kspace", "image"],
    },
    "huber": {"domain": "agnostic", "compatible_with": ["kspace", "image"]},
    # Standard agnostic
    "l1": {"domain": "agnostic", "compatible_with": ["kspace", "image", "latent"]},
    "l2": {"domain": "agnostic", "compatible_with": ["kspace", "image", "latent"]},
    "smooth_l1": {
        "domain": "agnostic",
        "compatible_with": ["kspace", "image", "latent"],
    },
    "frequency": {"domain": "agnostic", "compatible_with": ["kspace", "image"]},
    "log_spectral": {"domain": "agnostic", "compatible_with": ["kspace", "image"]},
    # Latent-specific
    "kl": {"domain": "latent", "compatible_with": ["latent", "agnostic"]},
    "vq": {"domain": "latent", "compatible_with": ["latent", "agnostic"]},
    "vq_kl": {"domain": "latent", "compatible_with": ["latent", "agnostic"]},
    "vqgan": {"domain": "latent", "compatible_with": ["latent", "agnostic"]},
    "latent_consistency": {
        "domain": "latent",
        "compatible_with": ["latent", "agnostic"],
    },
    "perceptual_latent": {
        "domain": "latent",
        "compatible_with": ["latent", "agnostic"],
    },
    "latent_diffusion": {
        "domain": "latent",
        "compatible_with": ["latent", "agnostic"],
    },
}

# Normalize aliases to canonical names
LOSS_ALIASES = {
    "mae": "l1",
    "mean_absolute_error": "l1",
    "mse": "l2",
    "mean_squared_error": "l2",
    "bce": "l1",
    "cross_entropy": "l1",
    "gan_standard": "l1",
    "gan_vanilla": "l1",
    "lsgan": "l1",
    "r1_regularization": "l1",
    "gan_bce": "l1",
    "contrastive": "l1",
    "domain_adversarial": "l1",
    "deep_supervision": "l1",
    "modality_swap": "l1",
    "uncertainty": "l1",
}


@dataclass
class DomainMismatchWarning:
    """Warning about potential domain mismatch."""

    loss_name: str
    loss_domain: str
    input_domain: str
    severity: str  # "error" or "warning"
    message: str
    fix: str  # Suggested fix


@dataclass
class DomainValidationResult:
    """Result of domain validation."""

    is_valid: bool
    errors: list[str]
    warnings: list[DomainMismatchWarning]


def _normalize_loss_name(loss_name: str) -> str:
    """Normalize loss name to canonical form via aliases."""
    canonical = loss_name.lower()
    return LOSS_ALIASES.get(canonical, canonical)


def _get_loss_domain(loss_name: str) -> dict | None:
    """Get domain metadata for a loss.

    Args:
        loss_name: Loss name (canonical or alias)

    Returns:
        Domain metadata dict or None if not found
    """
    canonical = _normalize_loss_name(loss_name)
    return LOSS_DOMAIN_REGISTRY.get(canonical)


def validate_loss_domains(config: LossConfigSchema, input_domain: str) -> DomainValidationResult:
    """Validate that all configured losses match input domain.

    Args:
        config: LossConfigSchema instance
        input_domain: Model input domain (kspace, image, or latent)

    Returns:
        DomainValidationResult with validation outcome

    Raises:
        ValueError: If input_domain is invalid
    """
    if input_domain not in ("kspace", "image", "latent"):
        raise ValueError(
            f"Invalid input domain: {input_domain}. Must be 'kspace', 'image', or 'latent'"
        )

    result = DomainValidationResult(is_valid=True, errors=[], warnings=[])

    # F35 / 2026-05-22 — losses declared under ``losses.image_losses`` are
    # evaluated in IMAGE domain via a bridge that LossBuilder auto-inserts
    # (the same contract the static audit's ``loss_domain_consistency`` check
    # already honours). For a k-space model these image-domain losses are
    # therefore EXPECTED and handled — not a domain error. Without this, a
    # correctly-bridged config (e.g. experiment_12_physics_cold_diffusion_v2
    # with hfen/ssim under image_losses) was rejected at pipeline build with
    # "Loss 'hfen' requires domain 'image' but model uses 'kspace'". Collect
    # the bridged set so the critical check below skips them.
    bridged_image_losses: set[str] = set()
    image_losses_block = getattr(config, "image_losses", None) or []
    for _entry in image_losses_block:
        _name = getattr(_entry, "name", None)
        if _name is None and isinstance(_entry, dict):
            _name = _entry.get("name")
        _enabled = getattr(_entry, "enabled", True)
        if isinstance(_entry, dict):
            _enabled = _entry.get("enabled", True)
        if _name and _enabled:
            bridged_image_losses.add(_normalize_loss_name(_name))

    # Create builder to get enabled losses
    class ConfigShim:
        """ConfigShim class."""

        def __init__(self, loss_config):
            """__init__.

            Args:
                loss_config (Any): Description.
            """
            self.losses = loss_config
            self.objectives = None
            self.training = None
            self.training_mode = "reconstruction"
            self.deep_supervision_weight = 0.0

    try:
        shim = ConfigShim(config)
        builder = LossBuilder(shim, device="cpu")  # type: ignore
        enabled_losses = builder.get_enabled_losses()
    except Exception as e:
        result.errors.append(f"Failed to get enabled losses: {e!s}")
        result.is_valid = False
        return result

    # Check each enabled loss against input domain
    for loss_name, weight in enabled_losses.items():
        domain_meta = _get_loss_domain(loss_name)

        if domain_meta is None:
            # Unknown loss, warn but don't fail (may be custom loss)
            result.warnings.append(
                DomainMismatchWarning(
                    loss_name=loss_name,
                    loss_domain="unknown",
                    input_domain=input_domain,
                    severity="warning",
                    message=f"Unknown loss '{loss_name}' - domain validation skipped (custom loss?).",
                    fix="Register loss domain in LOSS_DOMAIN_REGISTRY",
                )
            )
            continue

        loss_domain = domain_meta["domain"]
        compatible = domain_meta.get("compatible_with", [])

        # Check if loss domain is compatible with input domain
        is_compatible = loss_domain == "agnostic" or input_domain in compatible

        if not is_compatible:
            error_msg = (
                f"Loss '{loss_name}' requires domain '{loss_domain}' "
                f"but model uses '{input_domain}'"
            )

            # F35 — image-domain losses declared under ``image_losses`` are
            # bridged to image domain by LossBuilder, so a k-space model is
            # fine. Treat as compatible (skip silently) rather than a critical
            # build-blocking error.
            if loss_domain == "image" and _normalize_loss_name(loss_name) in bridged_image_losses:
                continue

            is_critical = loss_domain == "image" and input_domain == "kspace"

            if is_critical:
                severity = "error"
                fix = (
                    f"Set enable_{loss_name}: false or lambda_{loss_name}: 0.0 "
                    f"in your experiment YAML, "
                    f"or use spatial_losses_use_fourier_bridge: true "
                    f"to auto-bridge image losses to k-space"
                )
                logger.error("Domain mismatch (critical): %s — %s", loss_name, error_msg)
                result.errors.append(error_msg)
                result.is_valid = False
            else:
                severity = "warning"
                fix = f"Verify compatibility or set lambda_{loss_name}=0.0 if unsure"

            result.warnings.append(
                DomainMismatchWarning(
                    loss_name=loss_name,
                    loss_domain=loss_domain,
                    input_domain=input_domain,
                    severity=severity,
                    message=error_msg,
                    fix=fix,
                )
            )

    return result


def verify_startup_loss_domains(
    config: LossConfigSchema, input_domain: str, fail_on_error: bool = True
) -> bool:
    """Verify loss domains match input domain on startup (fail-fast).

    Integrates with bootstrap.py validation pipeline.

    Args:
        config: LossConfigSchema instance
        input_domain: Model input domain (kspace, image, or latent)
        fail_on_error: If True, raise RuntimeError on domain errors

    Returns:
        True if validation passed

    Raises:
        RuntimeError: If domain errors found and fail_on_error=True
    """
    result = validate_loss_domains(config, input_domain)

    # Print summary
    status = "🟢" if result.is_valid else "🔴"
    logger.debug(
        f"\n{status} Loss Domain Validation: {input_domain} domain | "
        f"{len(result.errors)} errors, {len(result.warnings)} warnings"
    )

    # Print errors
    if result.errors:
        logger.debug("\n❌ Domain Mismatches (ERRORS):")
        for error in result.errors:
            logger.debug(f"  - {error}")

    # Print warnings
    if result.warnings:
        logger.debug("\n⚠️  Domain Compatibility Warnings:")
        for warning in result.warnings:
            logger.debug(f"  - [{warning.severity.upper()}] {warning.message}")
            logger.debug(f"    Fix: {warning.fix}")

    # Fail fast if errors
    if result.errors and fail_on_error:
        raise RuntimeError(
            f"Loss domain validation failed for '{input_domain}' domain. "
            f"Fix the {len(result.errors)} critical error(s) above and retry."
        )

    return result.is_valid


def register_loss_domain(loss_name: str, domain: str, compatible_with: list | None = None) -> None:
    """Register domain metadata for a loss (for custom losses).

    Allows users to register custom losses with domain information.

    Args:
        loss_name: Canonical loss name
        domain: Loss domain (kspace, image, latent, agnostic, physics)
        compatible_with: List of compatible input domains
    """
    if domain not in ("kspace", "image", "latent", "agnostic", "physics"):
        raise ValueError(
            f"Invalid domain: {domain}. Must be one of: kspace, image, latent, agnostic, physics"
        )

    if compatible_with is None:
        compatible_with = []

    LOSS_DOMAIN_REGISTRY[loss_name.lower()] = {
        "domain": domain,
        "compatible_with": compatible_with,
    }


def get_domain_report(config: LossConfigSchema, input_domain: str) -> str:
    """Generate a detailed domain validation report.

    Args:
        config: LossConfigSchema instance
        input_domain: Model input domain

    Returns:
        Detailed report as multi-line string
    """
    result = validate_loss_domains(config, input_domain)

    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("Loss Domain Validation Report")
    lines.append("=" * 70)
    lines.append(f"Input Domain: {input_domain}")
    lines.append(f"Status: {'🟢 VALID' if result.is_valid else '🔴 INVALID'}")
    lines.append(f"Errors: {len(result.errors)} | Warnings: {len(result.warnings)}")
    lines.append("")

    if result.errors:
        lines.append("CRITICAL ERRORS:")
        for i, error in enumerate(result.errors, 1):
            lines.append(f"  {i}. ❌ {error}")
        lines.append("")

    if result.warnings:
        lines.append("WARNINGS:")
        for warning in result.warnings:
            lines.append(
                f"  ⚠️  [{warning.severity.upper()}] {warning.loss_name}: {warning.message}"
            )
            lines.append(f"      Fix: {warning.fix}")
        lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)
