"""Tests for the back-compat ``domain.entities.exceptions`` module.

Targets ``spectramr.domain.entities.exceptions``. Distinct from
``spectramr.domain.exceptions`` (note the spelling — ``SPECTRAMRError`` vs.
``SpectraMRError``); kept for cluster-stale-code back-compat. This file
verifies the alternate hierarchy still satisfies the catch-all
contract.
"""

from __future__ import annotations

import pytest

from spectramr.domain.entities.exceptions import (
    BufferAllocationError,
    ConfigurationError,
    SPECTRAMRError,
)

ALL_EXCEPTIONS = [ConfigurationError, BufferAllocationError]


def test_base_extends_python_exception() -> None:
    assert issubclass(SPECTRAMRError, Exception)


@pytest.mark.parametrize("exc_cls", ALL_EXCEPTIONS)
def test_subclasses_inherit_base(exc_cls: type) -> None:
    """Both subclasses descend from ``SPECTRAMRError``."""
    assert issubclass(exc_cls, SPECTRAMRError)


@pytest.mark.parametrize("exc_cls", ALL_EXCEPTIONS)
def test_can_be_raised_and_caught_via_base(exc_cls: type) -> None:
    """A ``except SPECTRAMRError`` block catches every domain.entities exception."""
    with pytest.raises(SPECTRAMRError):
        raise exc_cls("test")


def test_subclasses_remain_distinct() -> None:
    """``except ConfigurationError`` does not catch ``BufferAllocationError``."""
    with pytest.raises(BufferAllocationError):
        try:
            raise BufferAllocationError("oom")
        except ConfigurationError:
            pytest.fail("Sibling exception should not be caught here")


def test_message_preserved_via_str() -> None:
    err = ConfigurationError("missing key")
    assert str(err) == "missing key"


def test_distinct_from_domain_exceptions_module() -> None:
    """This back-compat module's SPECTRAMRError is distinct from spectramr.domain.exceptions.SpectraMRError.

    Both exist by name but are not the same class — we want to make sure
    nobody accidentally collapses the two hierarchies, since cluster code
    catches the back-compat name.
    """
    from spectramr.domain.exceptions import SpectraMRError

    # They're related Python Exception subtypes but not the same class.
    assert SPECTRAMRError is not SpectraMRError
