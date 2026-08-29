"""Cortical-surface infrastructure for fMRI §3.

Three primitives:

* :class:`CorticalMesh` — minimal mesh container with vertex / face
  arrays and a stable hash for caching.
* :class:`ConformalFlattening` — wraps the precomputed disk
  flattening. The Gu-Yau algorithm itself is an external service
  (FreeSurfer / GuYau-CPM); this class consumes its output.
* :class:`SurfaceMambaBlock` — bidirectional scan over a cortex-
  adaptive SFC drawn from the disk-flattened representation.

References:
    [14] X. Gu et al., "Genus zero surface conformal mapping",
         *IEEE Trans. Med. Imag.*, 23(8), 2004.
"""

from .cortical_mesh import ConformalFlattening, CorticalMesh
from .surface_mamba import SurfaceMambaBlock

__all__ = [
    "ConformalFlattening",
    "CorticalMesh",
    "SurfaceMambaBlock",
]
