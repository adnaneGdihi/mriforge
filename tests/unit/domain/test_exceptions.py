"""Tests for the MRIForge domain exception hierarchy.

Targets ``mriforge.domain.exceptions``. Verifies every domain-specific
exception inherits from ``MRIForgeError`` (the documented base class) so
generic ``except MRIForgeError`` blocks catch every project-specific raise.
"""

from __future__ import annotations

import pytest

from mriforge.domain.exceptions import (
    ConfigurationError,
    DataCorruptionError,
    DimensionMismatchError,
    MRIForgeError,
    ModelDivergenceError,
    ResourceError,
)

ALL_DOMAIN_EXCEPTIONS = [
    ConfigurationError,
    DataCorruptionError,
    DimensionMismatchError,
    ModelDivergenceError,
    ResourceError,
]


def test_base_exception_is_python_exception() -> None:
    """``MRIForgeError`` extends the standard ``Exception`` class."""
    assert issubclass(MRIForgeError, Exception)


@pytest.mark.parametrize("exc_cls", ALL_DOMAIN_EXCEPTIONS)
def test_domain_exceptions_inherit_mriforge_error(exc_cls: type) -> None:
    """Every domain exception must descend from ``MRIForgeError``.

    Otherwise a generic ``except MRIForgeError:`` block in calling code
    would silently fail to catch a project-specific failure — an
    operational risk.
    """
    assert issubclass(exc_cls, MRIForgeError)


@pytest.mark.parametrize("exc_cls", ALL_DOMAIN_EXCEPTIONS)
def test_domain_exceptions_can_be_instantiated_with_message(exc_cls: type) -> None:
    """Each exception class accepts a message and exposes it via ``str()``."""
    msg = "test message"
    err = exc_cls(msg)
    assert isinstance(err, MRIForgeError)
    assert str(err) == msg


@pytest.mark.parametrize("exc_cls", ALL_DOMAIN_EXCEPTIONS)
def test_domain_exceptions_can_be_raised_and_caught_as_base(exc_cls: type) -> None:
    """Raising a subclass is caught by ``except MRIForgeError``."""
    with pytest.raises(MRIForgeError):
        raise exc_cls("oops")


def test_subclasses_remain_distinguishable() -> None:
    """Catching a more-specific class doesn't catch siblings.

    e.g. ``except ConfigurationError`` should NOT catch
    ``ModelDivergenceError``.
    """
    with pytest.raises(ModelDivergenceError):
        try:
            raise ModelDivergenceError("loss is NaN")
        except ConfigurationError:
            pytest.fail("Sibling exception should not be caught here")


def test_exception_chaining_supported() -> None:
    """``raise X from Y`` chaining works (standard Python contract)."""
    inner = ValueError("original")
    try:
        try:
            raise inner
        except ValueError as e:
            raise ConfigurationError("wrapped") from e
    except ConfigurationError as outer:
        assert outer.__cause__ is inner
