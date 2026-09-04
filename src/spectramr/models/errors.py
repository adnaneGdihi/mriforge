"""Model Error Classes
==================

Custom exception and error code classes for model creation and validation.
"""

from enum import Enum


class FallbackCode(Enum):
    """Error codes for model factory fallback scenarios."""

    UNKNOWN_GENERATOR_TYPE = "unknown_generator_type"
    UNKNOWN_DISCRIMINATOR_TYPE = "unknown_discriminator_type"
    PARAMETER_MAPPING_FAILED = "parameter_mapping_failed"
    MODEL_VALIDATION_FAILED = "model_validation_failed"


class FactoryWarningCode(Enum):
    """Warning codes for model factory operations."""

    FALLBACK_USED = "fallback_used"
    PARAMETER_OVERRIDE = "parameter_override"
    MODEL_TYPE_DEPRECATED = "model_type_deprecated"


class ModelCreationError(Exception):
    """Exception raised when model creation fails."""

    def __init__(self, message: str, code: FallbackCode | None = None):
        """__init__.

        Args:
            message (str): Description.
            code (FallbackCode | None): Description.
        """
        super().__init__(message)
        self.code = code
        self.message = message

    def __str__(self) -> str:
        """__str__.

        Returns:
            str: Description.
        """
        if self.code:
            return f"{self.code.value}: {self.message}"
        return self.message


class ModelValidationError(ValueError):
    """Exception raised when model validation fails."""


class ParameterMappingError(ValueError):
    """Exception raised when parameter mapping fails."""
