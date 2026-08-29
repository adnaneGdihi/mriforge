"""Latent Gaussian Diffusion Generator.

This module provides the LatentGaussianDiffusion class, which is an alias
for LatentDiffusionGenerator.
"""

from mriforge.models.generators.latent_diffusion_generator import LatentDiffusionGenerator
from mriforge.models.registry import register_model


# Alias for backward compatibility or specific naming convention.
#
# The capability set below is a FULL mirror of ``latent_gan_generator`` /
# ``latent_diffusion`` (the two names on ``LatentDiffusionGenerator`` itself), and it
# has to be. This block used to say "the flag is re-declared here so the audit registry
# stays consistent across aliases" while re-declaring exactly ONE of six -- so the
# registry gave two different answers for one class: ``spatial_dims`` read ``(2,)``
# under the base names and ``None`` under this one (#1067).
#
# ``None`` is not a harmless omission. It means UNDECLARED, and the audit checks that
# consume these -- ``check_workflow_spatial_rank``, the signal-domain checks, the spec
# card -- simply do not run against an undeclared value. So an arm written against this
# name got no spatial-rank pre-flight while the identical arm written against
# ``latent_diffusion`` did. PR #1073 widened the gap rather than creating it: it
# narrowed the base from ``(2, 3)`` to ``(2,)`` and this alias was not updated with it.
#
# ``training_mode`` is DELIBERATELY different (``diffusion`` here, ``gan`` on the base):
# that is the reason the alias exists, and it is a registration fact rather than a model
# capability. Everything below describes what the CLASS can do, and the class is the
# same object, so these must not diverge again.
#
# Enforced by tests/unit/models/test_registry_capability_parity.py.
@register_model(
    "latent_gaussian_diffusion",
    training_mode="diffusion",
    supports_contrast_conditioning=True,
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class LatentGaussianDiffusion(LatentDiffusionGenerator):
    """Alias for LatentDiffusionGenerator registered as latent_gaussian_diffusion."""

    pass
