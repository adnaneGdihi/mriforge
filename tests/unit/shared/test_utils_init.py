"""Regression tests for ``spectramr.shared.utils.__init__``.

The package's ``__all__`` previously advertised three EMA names
(``ExponentialMovingAverage``, ``apply_ema_to_model``,
``create_ema_for_model``) whose backing imports were commented out. This made
``__all__`` lie: a star-import would raise ``AttributeError`` and ``getattr``
on those names would fail. These tests pin the invariant that every name in
``__all__`` is actually importable, and that the stale EMA names are absent.
"""

import pytest

import spectramr.shared.utils as u

pytestmark = pytest.mark.unit

_STALE_EMA_NAMES = (
    "ExponentialMovingAverage",
    "apply_ema_to_model",
    "create_ema_for_model",
)


def test_all_names_are_accessible() -> None:
    """Every name advertised in ``__all__`` must resolve via getattr."""
    missing = [name for name in u.__all__ if not hasattr(u, name)]
    assert missing == [], f"__all__ advertises non-importable names: {missing}"
    # And each is actually retrievable (not just truthy on hasattr).
    for name in u.__all__:
        assert getattr(u, name) is not None or hasattr(u, name)


def test_stale_ema_names_not_in_all() -> None:
    """The three EMA names with commented-out imports must not be advertised."""
    for name in _STALE_EMA_NAMES:
        assert name not in u.__all__, f"stale EMA name still in __all__: {name}"


def test_star_import_does_not_raise() -> None:
    """A star-import binds exactly the names in ``__all__`` with no error.

    This is the concrete failure mode the bug caused: ``from ... import *``
    walks ``__all__`` and raises ``AttributeError`` for any advertised-but-
    missing name.
    """
    namespace: dict[str, object] = {}
    exec("from spectramr.shared.utils import *", namespace)
    for name in u.__all__:
        assert name in namespace, f"star-import failed to bind {name!r}"


def test_all_is_nonempty_and_unique() -> None:
    """``__all__`` is a non-empty list of unique names."""
    assert isinstance(u.__all__, list)
    assert u.__all__, "__all__ is unexpectedly empty"
    assert len(u.__all__) == len(set(u.__all__)), "__all__ contains duplicates"
