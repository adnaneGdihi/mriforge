"""
Domain Exceptions for MRIForge
=============================

This module defines the exception hierarchy for the MRIForge project.
All custom exceptions should inherit from `MRIForgeError`.
"""


class MRIForgeError(Exception):
    """Base class for all exceptions in MRIForge."""

    pass


class ConfigurationError(MRIForgeError):
    """Raised when configuration is invalid or missing."""

    pass


class ModelDivergenceError(MRIForgeError):
    """Raised when model training diverges (loss is NaN or Inf)."""

    pass


class DimensionMismatchError(MRIForgeError):
    """Raised when tensor dimensions do not match expected physics or model constraints."""

    pass


class ResourceError(MRIForgeError):
    """Raised when a required resource (GPU, File) is unavailable."""

    pass


class DataCorruptionError(MRIForgeError):
    """Raised when input data is corrupt or unreadable."""

    pass


class WorkflowNotImplementedError(MRIForgeError):
    """Raised when a pipeline is asked to run an imaging regime the framework
    cannot honestly honour.

    A regime whose :class:`~mriforge.config.schemas.enums.Maturity` is ``STUB``
    has no forward operator, no losses, no strategy — every pipeline raises
    this. A regime that is ``EVAL_ONLY`` raises this from ``train`` (but not
    from ``evaluate``/``predict``). See
    :mod:`mriforge.domain.workflows.profiles`.
    """

    pass
