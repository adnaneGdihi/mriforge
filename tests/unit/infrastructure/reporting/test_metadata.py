"""Tests for ``mriforge.infrastructure.reporting.metadata``.

The paired test file this module never had — which is part of why ``_REPO_ROOT``
could be off by one level for months (#721).

Focus: the repo-root anchor and the vendored-baseline SHA lookup, i.e. the bits
that answer "which upstream revision produced these published numbers?".
``baseline_provenance``'s declarative half is covered by
``test_baseline_provenance.py``.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest

from mriforge.infrastructure.reporting import metadata as meta_mod


class TestRepoRootAnchor:
    """``_REPO_ROOT`` must be the repository root, not ``src/``.

    It was ``parents[3]``, correct while the package lived at ``src/`` and
    silently wrong after the 2026-05 ``src -> src/mriforge`` refactor added a
    level. Nothing caught it because the only consumer returns ``None`` on a
    miss, so every baseline recorded ``upstream_commit: None`` (#721).
    """

    def test_it_points_at_the_repo_root_not_the_package_dir(self):
        assert meta_mod._REPO_ROOT.name != "src"
        assert (meta_mod._REPO_ROOT / "pyproject.toml").exists()

    def test_the_external_baselines_tree_is_reachable_from_it(self):
        """The anchor's whole job. A wrong root makes every lookup miss."""
        assert (meta_mod._REPO_ROOT / "external" / "baselines").is_dir()

    def test_it_is_derived_from_this_files_location_not_the_cwd(self):
        """Runs are launched from many directories; a cwd-relative root would
        resolve differently under SLURM than in a dev shell."""
        expected = Path(meta_mod.__file__).resolve().parents[4]
        assert meta_mod._REPO_ROOT == expected


class TestVendorCommitSha:
    def test_an_absent_vendor_dir_is_a_quiet_none(self, caplog):
        """Not-vendored is a legitimate state (submodule uninitialised)."""
        with caplog.at_level(logging.WARNING):
            assert meta_mod._vendor_commit_sha("definitely_not_vendored") is None
        assert not caplog.records, "an absent submodule must not warn"

    def test_a_present_but_unreadable_vendor_dir_warns(self, monkeypatch, caplog):
        """The distinction the bare ``except Exception`` erased.

        A directory that EXISTS but cannot be identified is a real problem: the
        run replicates an upstream revision it cannot name. Sharing an answer
        with "not vendored" is what let #721 hide.
        """
        monkeypatch.setattr(Path, "exists", lambda self: True)

        def _boom(*a, **kw):
            raise subprocess.CalledProcessError(128, "git")

        monkeypatch.setattr(subprocess, "check_output", _boom)

        with caplog.at_level(logging.WARNING):
            assert meta_mod._vendor_commit_sha("cdiffmr") is None

        assert any("rev-parse" in r.getMessage() for r in caplog.records)

    def test_a_real_vendored_baseline_resolves_to_a_sha(self):
        """End-to-end on the tree as checked out, when the submodule is present."""
        vendor = meta_mod._REPO_ROOT / "external" / "baselines" / "cdiffmr"
        if not vendor.exists():
            pytest.skip("cdiffmr submodule not initialised in this checkout")

        sha = meta_mod._vendor_commit_sha("cdiffmr")

        assert sha is not None, "vendored baseline present but unidentified (#721)"
        assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)

    def test_an_empty_sha_is_reported_as_none(self, monkeypatch):
        """A blank stdout is not a commit; it must not reach provenance as one."""
        monkeypatch.setattr(Path, "exists", lambda self: True)
        monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: b"  \n")

        assert meta_mod._vendor_commit_sha("cdiffmr") is None
