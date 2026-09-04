"""Connectivity-domain metrics (functional / structural).

Currently hosts the geodesic FC error metric used by
fMRI §4 (Riemannian dynamic-FC diffusion). Other connectivity
metrics (GFA, anisotropy) are registered elsewhere in the
``spectramr.core.metrics`` hierarchy.
"""

from .fc_metrics import GeodesicFCErrorMetric

__all__ = ["GeodesicFCErrorMetric"]
