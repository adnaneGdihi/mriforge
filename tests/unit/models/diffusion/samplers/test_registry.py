"""Tests for the diffusion sampler registry.

Targets ``mriforge.models.diffusion.samplers.registry`` — the
``@register_sampler`` dispatch + lookup helpers introduced by
``CC-1`` of ``TODO/backlog_paradigm_expansion_roadmap.md``.

Categories:

- ``register_sampler`` decorator stores the class under canonical name
- Aliases dispatch case-insensitively to the same canonical class
- ``get_sampler`` raises ``KeyError`` with the available list on miss
- ``list_available`` is sorted
- Re-registering the same class is a no-op
- Constructor kwargs forwarded by ``get_sampler``
"""

from __future__ import annotations

import pytest

from mriforge.models.diffusion.samplers import (
    SAMPLER_REGISTRY,
    SamplerRegistry,
    get_sampler,
    list_available,
    register_sampler,
)


class _DummySampler:
    """Minimal sampler test double."""

    def __init__(self, eta: float = 0.0) -> None:
        self.eta = eta


@pytest.fixture(autouse=True)
def _isolated_registration():
    """Register the dummy for the test duration only.

    Because the registry is a class-level singleton, leaving entries
    behind would leak between tests. We register once, then prune.
    """
    register_sampler("__dummy__", aliases=["__dummy_alias__", "DummyAlias"])(
        _DummySampler
    )
    yield
    SamplerRegistry._samplers.pop("__dummy__", None)
    for alias in ("__dummy_alias__", "dummyalias"):
        SamplerRegistry._aliases.pop(alias, None)


def test_register_and_lookup_canonical() -> None:
    """``get_sampler`` finds the registered class."""
    sampler = get_sampler("__dummy__")
    assert isinstance(sampler, _DummySampler)


def test_lookup_via_alias() -> None:
    """An alias dispatches to the same canonical class."""
    sampler = get_sampler("__dummy_alias__")
    assert isinstance(sampler, _DummySampler)


def test_alias_is_case_insensitive() -> None:
    """Aliases are normalised to lower-case at registration."""
    sampler = get_sampler("DummyAlias")
    assert isinstance(sampler, _DummySampler)


def test_canonical_name_case_insensitive() -> None:
    """Canonical lookup is also case-insensitive."""
    sampler = get_sampler("__DUMMY__")
    assert isinstance(sampler, _DummySampler)


def test_constructor_kwargs_forwarded() -> None:
    """``**kwargs`` flow through to the sampler constructor."""
    sampler = get_sampler("__dummy__", eta=0.7)
    assert sampler.eta == 0.7


def test_unknown_sampler_raises_keyerror() -> None:
    """Unknown name → ``KeyError`` mentioning available samplers."""
    with pytest.raises(KeyError, match="Unknown sampler"):
        get_sampler("definitely_not_registered_12345")


def test_list_available_sorted_includes_dummy() -> None:
    """``list_available`` returns sorted canonical keys."""
    available = list_available()
    assert "__dummy__" in available
    assert available == sorted(available)


def test_is_registered_handles_alias_and_canonical() -> None:
    """``is_registered`` covers both surfaces."""
    assert SamplerRegistry.is_registered("__dummy__") is True
    assert SamplerRegistry.is_registered("__dummy_alias__") is True
    assert SamplerRegistry.is_registered("__missing__") is False


def test_get_class_does_not_instantiate() -> None:
    """``get_class`` returns the type without running ``__init__``."""
    cls = SamplerRegistry.get_class("__dummy__")
    assert cls is _DummySampler


def test_module_level_alias() -> None:
    """``SAMPLER_REGISTRY`` is the public alias for ``SamplerRegistry``."""
    assert SAMPLER_REGISTRY is SamplerRegistry
