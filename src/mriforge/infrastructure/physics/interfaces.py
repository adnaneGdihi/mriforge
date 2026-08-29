"""Physics operator interfaces for MRI physics operations.

This module defines the unified abstract interfaces for physics operators
used in MRI reconstruction and simulation. Consolidates multiple interface
definitions into a single, comprehensive protocol.
"""

from abc import ABC, abstractmethod
from typing import Protocol

import torch


class IPhysicsOperator(Protocol):
    """Unified protocol for physics operators in MRI.

    Combines forward and adjoint operations with comprehensive
    interface for MRI physics-based reconstruction.
    Mathematical Formulation:
    .. math::

        \\mathcal{O}_{IPhysicsOperator}(x) = \text{abstract}"""

    @abstractmethod
    def forward(self, img: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply forward physics operator.

        Args:
            img: Input image tensor of shape (B, C, H, W)
            **kwargs: Additional operator-specific parameters

        Returns:
            Transformed tensor (e.g., k-space)

        """
        ...

    @abstractmethod
    def adjoint(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply adjoint physics operator.

        Args:
            data: Input data tensor (e.g., k-space)
            **kwargs: Additional operator-specific parameters

        Returns:
            Reconstructed image tensor of shape (B, C, H, W)

        """
        ...

    @property
    @abstractmethod
    def img_size(self) -> tuple[int, int]:
        """Get image size (H, W)."""
        ...

    @abstractmethod
    def get_operator_type(self) -> str:
        """Get string identifier for operator type."""
        ...


class IPhysicsOperatorABC(ABC):
    """Abstract base class version of IPhysicsOperator.

    Provides the same interface but as an ABC for inheritance.
    Mathematical Formulation:
    .. math::

        \\mathcal{O}_{IPhysicsOperatorABC}(x) = \text{abstract}"""

    @abstractmethod
    def forward(self, img: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply forward physics operator.

        Args:
            img: Input image tensor of shape (B, C, H, W)
            **kwargs: Additional operator-specific parameters

        Returns:
            Transformed tensor (e.g., k-space)

        """

    @abstractmethod
    def adjoint(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply adjoint physics operator.

        Args:
            data: Input data tensor (e.g., k-space)
            **kwargs: Additional operator-specific parameters

        Returns:
            Reconstructed image tensor of shape (B, C, H, W)

        """

    @property
    @abstractmethod
    def img_size(self) -> tuple[int, int]:
        """Get image size (H, W)."""

    @abstractmethod
    def get_operator_type(self) -> str:
        """Get string identifier for operator type."""


class IDataConsistency(ABC):
    """Abstract interface for data consistency operations.

    Defines the contract for enforcing data consistency
    in iterative reconstruction algorithms.
    """

    @abstractmethod
    def enforce_consistency(
        self,
        _current_img: torch.Tensor,
        _measured_data: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Enforce data consistency between current estimate and measurements.

        Args:
            current_img: Current image estimate
            measured_data: Measured data (e.g., k-space)
            mask: Optional sampling mask

        Returns:
            Consistency-enforced image

        """


# Backward compatibility aliases
PhysicsOperator = IPhysicsOperator
BasePhysicsOperator = IPhysicsOperatorABC


__all__ = [
    "BasePhysicsOperator",  # Backward compatibility
    "IDataConsistency",
    "IPhysicsOperator",
    "IPhysicsOperatorABC",
    "PhysicsOperator",  # Backward compatibility
]
