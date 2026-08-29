"""MRF-DiPh: Physics-Constrained Diffusion Sampler
================================================

Magnetic Resonance Fingerprinting Diffusion Physics (MRF-DiPh) sampler
that prevents hallucinations by projecting diffusion samples onto the
Bloch manifold at each reverse step.

Key Innovation:
- Standard diffusion: denoise -> sample
- MRF-DiPh: denoise -> dictionary match -> Bloch projection -> blend -> sample

This ensures all generated images are physically realizable MRI signals.
"""

from .bloch_dictionary import BlochDictionary
from .bloch_projection import BlochProjector
from .mrf_diph_sampler import MRFDiPhSampler, create_mrf_diph_sampler

__all__ = [
    "BlochDictionary",
    "BlochProjector",
    "MRFDiPhSampler",
    "create_mrf_diph_sampler",
]
