"""Exceptions for dataset operations.

Separate module to avoid circular imports.
"""


class DatasetValidationError(Exception):
    """Raised when dataset configuration is invalid."""

    pass


class DatasetNotFoundError(DatasetValidationError):
    """Raised when required dataset files not found."""

    pass


class DatasetTypeNotSupportedError(DatasetValidationError):
    """Raised when dataset type not registered."""

    pass
