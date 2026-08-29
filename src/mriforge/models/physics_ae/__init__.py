"""Physics-decoder autoencoders.

Models whose *decoder is a physical law* rather than a learned network, so the
latent is forced to carry physically-meaningful quantities. Currently the
dispersion-latent Bloch autoencoder (DL-BAE, M4 of the 2026-06-29
contrast/field-agnostic bundle design), whose decoder is the
Bloembergen-Purcell-Pound relaxation-dispersion law followed by the Bloch render.

Importing this package registers its models on the registry via
``@register_model``.
"""

from __future__ import annotations

from mriforge.models.physics_ae.disp_bloch_ae import DispersionBlochAutoencoder

__all__ = ["DispersionBlochAutoencoder"]
