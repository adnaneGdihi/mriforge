"""Absence-path tests for CubicalPHWassersteinLoss's optional dependencies.

These live apart from ``test_cubical_ph_w2_loss.py`` on purpose. That module
calls ``pytest.importorskip("gudhi")`` at import time, so it skips *entirely*
exactly where this behaviour matters -- on a host where gudhi cannot be
installed. A test of the missing-dependency message must run when the
dependency is missing, so it cannot share that module.

What is pinned here is the NN18 contract: gudhi and POT are imported in
separate ``try`` blocks, so which one is absent is *known*, and the error must
state it rather than infer a union. Plus the linux/aarch64 case, where
``pip install -e ".[topology]"`` now succeeds while installing no gudhi (it is
marked off aarch64 in pyproject.toml because upstream ships neither a wheel nor
an sdist there) -- so pointing the user at that command would loop them
straight back to this same error.
"""

from __future__ import annotations

import platform

import pytest

# NO importorskip: the module under test is designed to import with gudhi and
# POT both absent, and that property is itself part of what is being pinned.
from spectramr.models.losses import cubical_ph_w2_loss as mod

LINUX_ARM = ("Linux", "aarch64")
LINUX_X86 = ("Linux", "x86_64")
MAC_ARM = ("Darwin", "arm64")


@pytest.fixture
def env(monkeypatch):
    """Set the two module-level dep flags and the reported platform."""

    def _apply(*, gudhi: bool, ot: bool, host: tuple[str, str]) -> str:
        monkeypatch.setattr(mod, "_gudhi", object() if gudhi else None)
        monkeypatch.setattr(mod, "_ot", object() if ot else None)
        monkeypatch.setattr(platform, "system", lambda: host[0])
        monkeypatch.setattr(platform, "machine", lambda: host[1])
        return mod._install_hint()

    return _apply


def test_module_imports_with_both_deps_absent() -> None:
    """The guarded imports must never make the module itself unimportable."""
    assert mod._install_hint is not None
    assert hasattr(mod, "_gudhi")
    assert hasattr(mod, "_ot")


@pytest.mark.parametrize(
    ("gudhi", "ot", "named", "not_named"),
    [
        (False, True, "gudhi", "POT"),
        (True, False, "POT", "gudhi"),
    ],
)
def test_hint_names_the_package_that_is_actually_missing(
    env, gudhi: bool, ot: bool, named: str, not_named: str
) -> None:
    """Reporting the union would infer absence; the flags already state it."""
    msg = env(gudhi=gudhi, ot=ot, host=LINUX_X86)
    assert named in msg
    assert not_named not in msg


def test_hint_names_both_when_both_are_missing(env) -> None:
    msg = env(gudhi=False, ot=False, host=LINUX_X86)
    assert "gudhi" in msg
    assert "POT" in msg


def test_linux_aarch64_reports_unavailable_and_does_not_suggest_the_extra(env) -> None:
    """On linux/aarch64 the extra installs no gudhi, so it must not be suggested."""
    msg = env(gudhi=False, ot=True, host=LINUX_ARM)
    assert "aarch64" in msg
    assert "no aarch64 wheel and no sdist" in msg
    assert "pip install" not in msg
    assert "make install-topology" not in msg


def test_macos_arm64_still_gets_the_normal_install_hint(env) -> None:
    """The machine string alone is not the discriminator.

    gudhi *does* publish macOS arm64 wheels; only linux/aarch64 is the gap. A
    check written against ``platform.machine()`` alone would match "arm64" here
    and wrongly tell a Mac user the loss is unavailable.
    """
    msg = env(gudhi=False, ot=True, host=MAC_ARM)
    assert 'pip install -e ".[topology]"' in msg
    assert "aarch64" not in msg


def test_linux_x86_64_gets_the_normal_install_hint(env) -> None:
    msg = env(gudhi=False, ot=True, host=LINUX_X86)
    assert 'pip install -e ".[topology]"' in msg
    assert "aarch64" not in msg


def test_constructor_raises_importerror_carrying_the_hint(env, monkeypatch) -> None:
    """The hint is only useful if the raise actually carries it."""
    monkeypatch.setattr(mod, "_gudhi", None)
    monkeypatch.setattr(mod, "_ot", object())
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    with pytest.raises(ImportError, match="no aarch64 wheel and no sdist"):
        mod.CubicalPHWassersteinLoss(homology_dims=(0,))
