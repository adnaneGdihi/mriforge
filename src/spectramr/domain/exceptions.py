"""
Domain Exceptions for spectraMR
=============================

This module defines the exception hierarchy for the spectraMR project.
All custom exceptions should inherit from `SpectraMRError`.
"""


class SpectraMRError(Exception):
    """Base class for all exceptions in spectraMR."""

    pass


class ConfigurationError(SpectraMRError):
    """Raised when configuration is invalid or missing."""

    pass


class ModelDivergenceError(SpectraMRError):
    """Raised when model training diverges (loss is NaN or Inf)."""

    pass


class DimensionMismatchError(SpectraMRError):
    """Raised when tensor dimensions do not match expected physics or model constraints."""

    pass


class ResourceError(SpectraMRError):
    """Raised when a required resource (GPU, File) is unavailable."""

    pass


class DataCorruptionError(SpectraMRError):
    """Raised when input data is corrupt or unreadable."""

    pass


class WorkflowNotImplementedError(SpectraMRError):
    """Raised when a pipeline is asked to run an imaging regime the framework
    cannot honestly honour.

    A regime whose :class:`~spectramr.config.schemas.enums.Maturity` is ``STUB``
    has no forward operator, no losses, no strategy — every pipeline raises
    this. A regime that is ``EVAL_ONLY`` raises this from ``train`` (but not
    from ``evaluate``/``predict``). See
    :mod:`spectramr.domain.workflows.profiles`.
    """

    pass
