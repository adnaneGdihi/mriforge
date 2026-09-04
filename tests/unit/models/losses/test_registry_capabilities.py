"""Tests for the LossCapabilities retrieval bridge (cached-cascade WS-X).

``get_loss_capabilities`` is the single retrieval surface: it returns an
explicit ``capabilities=`` contract when one was registered, otherwise
synthesizes a ``LossCapabilities`` from the legacy ``domain=`` plumbing, so the
two surfaces present as one.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch.nn as nn

from spectramr.models.capabilities import LossCapabilities
from spectramr.models.losses.registry import LossRegistry, get_loss_capabilities


@pytest.fixture
def _isolated_registry() -> Iterator[None]:
    """Snapshot/restore the LossRegistry class dicts so temp registrations
    don't leak into other tests (or trip the duplicate-registration guard)."""
    snaps = {
        "_custom_losses": dict(LossRegistry._custom_losses),
        "_aliases": dict(LossRegistry._aliases),
        "_loss_domains": dict(LossRegistry._loss_domains),
        "_loss_capabilities": dict(LossRegistry._loss_capabilities),
    }
    try:
        yield
    finally:
        for name, snap in snaps.items():
            d = getattr(LossRegistry, name)
            d.clear()
            d.update(snap)


def test_unknown_loss_returns_none() -> None:
    assert get_loss_capabilities("__definitely_not_a_loss__") is None


def test_explicit_capabilities_take_precedence(_isolated_registry: None) -> None:
    caps = LossCapabilities(domain="kspace", requires_paired_target=True)

    @LossRegistry.register("wsx_explicit_loss", capabilities=caps)
    class _Explicit(nn.Module):
        pass

    assert get_loss_capabilities("wsx_explicit_loss") == caps
    assert get_loss_capabilities("wsx_explicit_loss").requires_paired_target is True


def test_legacy_domain_is_synthesized(_isolated_registry: None) -> None:
    @LossRegistry.register("wsx_legacy_image_loss", domain="image")
    class _LegacyImage(nn.Module):
        pass

    resolved = get_loss_capabilities("wsx_legacy_image_loss")
    assert resolved == LossCapabilities(domain="image")


def test_complex_domain_maps_to_complex_image(_isolated_registry: None) -> None:
    @LossRegistry.register("wsx_complex_loss", domain="complex")
    class _Complex(nn.Module):
        pass

    assert get_loss_capabilities("wsx_complex_loss").domain == "complex_image"


def test_unmappable_domain_yields_none_domain(_isolated_registry: None) -> None:
    # "physics"/"agnostic" have no Domain literal — synthesize with domain=None
    # rather than guessing (no silent fallback).
    @LossRegistry.register("wsx_physics_loss", domain="physics")
    class _Physics(nn.Module):
        pass

    resolved = get_loss_capabilities("wsx_physics_loss")
    assert resolved is not None
    assert resolved.domain is None


def test_loss_without_domain_or_caps_returns_none(_isolated_registry: None) -> None:
    @LossRegistry.register("wsx_bare_loss")
    class _Bare(nn.Module):
        pass

    assert get_loss_capabilities("wsx_bare_loss") is None


def test_classmethod_and_module_fn_agree(_isolated_registry: None) -> None:
    @LossRegistry.register("wsx_parity_loss", domain="kspace")
    class _Parity(nn.Module):
        pass

    assert get_loss_capabilities(
        "wsx_parity_loss"
    ) == LossRegistry.get_loss_capabilities("wsx_parity_loss")


class TestCanonicalName:
    """``canonical_name`` is the public alias-resolution seam.

    Before it existed, ``_aliases.get(n, n)`` was copy-pasted at four call sites and
    every consumer that keyed on the RAW config name treated ``mae`` and ``l1`` as two
    distinct losses — which is how the fold seam's ``{l1, l2, mse}`` skip-list could be
    bypassed by spelling the same loss ``mae`` (a double-count).
    """

    def test_alias_resolves_to_canonical(self) -> None:
        assert LossRegistry.canonical_name("mse") == "l2"
        assert LossRegistry.canonical_name("mae") == "l1"

    def test_canonical_name_is_idempotent(self) -> None:
        assert LossRegistry.canonical_name("l2") == "l2"

    def test_case_insensitive(self) -> None:
        assert LossRegistry.canonical_name("MSE") == "l2"

    def test_unregistered_name_passes_through(self) -> None:
        # Strategy-inline lambdas (e.g. pre_dc_kspace) have no registry entry, so
        # callers must be able to canonicalise unconditionally.
        assert LossRegistry.canonical_name("pre_dc_kspace") == "pre_dc_kspace"

    def test_is_registered_agrees_with_canonical_name(self) -> None:
        assert LossRegistry.is_registered("mse") is True
        assert LossRegistry.is_registered("definitely_not_a_loss") is False
