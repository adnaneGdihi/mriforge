"""Tests for ``scripts/maintenance/prove_reachable.py``.

The load-bearing test is ``test_a_decorator_that_only_fires_on_a_walk_is_reported``:
it builds a package shaped exactly like the six incidents recorded in
``models/init_registry.py:73-190`` — a decorator whose module the curated entry point
does not import — and asserts the tool calls it unreachable. Without that case the
suite would only prove the tool runs, not that it detects anything.

These spawn real subprocesses, which is the point: the defect is invisible to any
same-process check, because the checking process has already imported the module.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "maintenance" / "prove_reachable.py"


def _load():
    spec = importlib.util.spec_from_file_location("prove_reachable", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["prove_reachable"] = module
    spec.loader.exec_module(module)
    return module


mod = _load()


@pytest.fixture
def fake_pkg(tmp_path, monkeypatch):
    """A package with a registry, one curated import, and one uncurated module."""
    pkg = tmp_path / "fakereg"
    pkg.mkdir()
    (pkg / "registry.py").write_text(
        textwrap.dedent("""
            REGISTRY = {}

            def register(name):
                def deco(fn):
                    REGISTRY[name] = fn
                    return fn
                return deco
            """)
    )
    (pkg / "curated.py").write_text(
        "from fakereg.registry import register\n\n@register('curated_one')\ndef f(): ...\n"
    )
    # The defect: decorated, correct-looking, and imported by nothing.
    (pkg / "orphan.py").write_text(
        "from fakereg.registry import register\n\n@register('orphan_one')\ndef f(): ...\n"
    )
    (pkg / "__init__.py").write_text("from . import curated as _curated  # noqa: F401\n")
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    return mod.Registry(
        kind="fake",
        cold="import fakereg",
        home="fakereg",
        names="list(__import__('fakereg.registry', fromlist=['x']).REGISTRY)",
    )


class TestDetection:
    def test_a_decorator_that_only_fires_on_a_walk_is_reported(self, fake_pkg):
        """The whole reason the script exists: registered != reachable."""
        result = mod.probe(fake_pkg, timeout=60)
        assert result["error"] is None
        assert result["cold"] == ["curated_one"]
        assert set(result["warm"]) == {"curated_one", "orphan_one"}
        assert result["unreachable"] == ["orphan_one"]

    def test_curating_the_import_clears_the_finding(self, fake_pkg, tmp_path):
        """Adding the missing import is the fix, and the tool must agree."""
        init = tmp_path / "fakereg" / "__init__.py"
        init.write_text(init.read_text() + "from . import orphan as _orphan  # noqa: F401\n")
        result = mod.probe(fake_pkg, timeout=60)
        assert result["unreachable"] == []
        assert set(result["cold"]) == {"curated_one", "orphan_one"}


class TestProbeRobustness:
    def test_an_unreadable_registry_reports_an_error_not_an_empty_set(self, monkeypatch, tmp_path):
        """An empty registry and an unimportable one must not look identical.

        This is the soft-skip seam that `config_health_checker.py` gets wrong: it
        returns passed=True when the registry cannot be imported, so the guard passes
        exactly when the chain is broken.
        """
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        broken = mod.Registry(
            kind="broken",
            cold="import a_module_that_does_not_exist",
            home="a_module_that_does_not_exist",
            names="[]",
        )
        result = mod.probe(broken, timeout=60)
        assert result["error"], "an unimportable registry must surface as an error"
        assert "ModuleNotFoundError" in str(result["error"])

    def test_timeout_is_reported_rather_than_hanging(self, monkeypatch, tmp_path):
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        slow = mod.Registry(kind="slow", cold="import time; time.sleep(30)", home="sys", names="[]")
        names, err = mod._run(slow.cold, slow.names, timeout=1)
        assert names == [] and err is not None and "timed out" in err


class TestNameQuery:
    """`--name` had no test at all; its three outcomes must be distinguishable."""

    def _run_name(self, monkeypatch, capsys, reg, name):
        monkeypatch.setattr(mod, "REGISTRIES", (reg,))
        monkeypatch.setattr(sys, "argv", ["prove_reachable.py", "--name", name])
        code = mod.main()
        return code, capsys.readouterr().out

    def test_a_reachable_name_reports_reachable_and_exits_zero(self, fake_pkg, monkeypatch, capsys):
        code, out = self._run_name(monkeypatch, capsys, fake_pkg, "curated_one")
        assert code == 0
        assert "REACHABLE" in out and "UNREACHABLE" not in out

    def test_a_registered_but_uncurated_name_reports_unreachable_and_exits_one(
        self, fake_pkg, monkeypatch, capsys
    ):
        code, out = self._run_name(monkeypatch, capsys, fake_pkg, "orphan_one")
        assert code == 1
        assert "UNREACHABLE" in out
        assert "UNKNOWN" not in out, "a registered-but-uncurated name is not 'unknown'"

    def test_an_absent_name_reports_unknown_and_exits_one(self, fake_pkg, monkeypatch, capsys):
        code, out = self._run_name(monkeypatch, capsys, fake_pkg, "never_registered")
        assert code == 1
        assert "UNKNOWN" in out
        assert "REACHABLE" not in out


class TestSummaryAccounting:
    """A probe that could not RUN must never be summarised as names being unreachable."""

    def test_failed_probes_are_not_counted_as_unreachable_names(
        self, monkeypatch, capsys, tmp_path
    ):
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(
            mod,
            "REGISTRIES",
            (
                mod.Registry(
                    kind="broken", cold="import nope_not_a_module", home="nope", names="[]"
                ),
            ),
        )
        monkeypatch.setattr(sys, "argv", ["prove_reachable.py", "--audit"])
        assert mod.main() == 1
        out = capsys.readouterr().out
        assert "PROBE FAILED" in out
        assert "were NOT checked" in out
        # The old wording said "1 name(s) are registered but not reachable" for this.
        assert "are registered but not reachable" not in out
        assert "Every registered name is reachable" not in out


class TestRegistryTable:
    def test_every_declared_registry_names_a_real_spectramr_package(self):
        """A typo in the table would silently probe nothing and report clean."""
        for reg in mod.REGISTRIES:
            assert reg.home.startswith("spectramr."), reg.kind
            assert reg.cold and reg.names, reg.kind

    def test_kinds_are_unique(self):
        kinds = [r.kind for r in mod.REGISTRIES]
        assert len(kinds) == len(set(kinds))
