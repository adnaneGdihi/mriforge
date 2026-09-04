"""Navigation models — endogenous-navigator motion-correction architectures.

Currently houses the SE(3)-equivariant DC-navigator (B-3 of the VF campaign).
Importing this package registers the model via its ``@register_model``
decorator.
"""

from spectramr.models.navigation.se3_equivariant_dc_navigator import (
    SE3EquivariantDCNavigator,
)

__all__ = ["SE3EquivariantDCNavigator"]
