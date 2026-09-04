"""Geometric Deep Learning models for MRI (Mesh, Slicing, Manifolds)."""

from .b0_estimator import B0Estimator
from .differentiable_slicer import DifferentiableSlicer
from .grafmri import GraFMRIGenerator
from .storm import GraphImagePrior, ManifoldLaplacian
from .template_deformation import TemplateDeformer

# ``SToRMRegularizer`` was moved to ``spectramr.models.losses.storm_loss``
# during the Phase 5 canonical-home migration (pitfall #12). Import it
# directly from there: ``from spectramr.models.losses.storm_loss import
# SToRMRegularizer``. Re-exporting here would create a circular import
# (geometric → losses → geometric.storm → geometric/__init__).

__all__ = [
    "B0Estimator",
    "DifferentiableSlicer",
    "GraFMRIGenerator",
    "GraphImagePrior",
    "ManifoldLaplacian",
    "TemplateDeformer",
]
