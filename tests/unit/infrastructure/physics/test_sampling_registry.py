"""One registry owns every sampling-pattern name.

Before this, four vocabularies named the same patterns and disagreed: two arms
could declare the same physics under different spellings, and a third could
declare a spelling nothing resolved (issues #953, #954).
"""

from __future__ import annotations

import pytest

from spectramr.infrastructure.physics.sampling_registry import SamplingPatternRegistry


class TestResolution:
    def test_canonical_names_resolve_to_themselves(self) -> None:
        for name in SamplingPatternRegistry.list_canonical():
            assert SamplingPatternRegistry.resolve(name) == name

    @pytest.mark.parametrize(
        ("alias", "canonical"),
        [
            ("random", "random_cartesian"),
            ("gaussian", "variable_density_2d_gaussian"),
            ("uniform", "uniform_cartesian"),
            ("poisson", "poisson_disk"),
            ("fractional_vd", "fractional_variable_density"),
            ("cartesian_vd", "variable_density_1d"),
        ],
    )
    def test_documented_aliases_resolve(self, alias: str, canonical: str) -> None:
        assert SamplingPatternRegistry.resolve(alias) == canonical

    def test_unknown_name_raises_and_lists_the_options(self) -> None:
        """Non-negotiable #3: an unknown option raises, never degrades."""
        with pytest.raises(ValueError) as excinfo:
            SamplingPatternRegistry.resolve("definitely_not_a_pattern")
        message = str(excinfo.value)
        assert "definitely_not_a_pattern" in message
        assert "random_cartesian" in message, "the error must list what IS accepted"

    def test_resolution_is_case_insensitive(self) -> None:
        assert SamplingPatternRegistry.resolve("Random") == "random_cartesian"

    def test_every_alias_points_at_a_canonical_name(self) -> None:
        """An alias to a missing key is a landmine that fires at training time."""
        canonical = set(SamplingPatternRegistry.list_canonical())
        for alias, target in SamplingPatternRegistry.ALIASES.items():
            assert target in canonical, f"alias {alias!r} points at unknown {target!r}"

    def test_accepted_is_canonical_plus_aliases(self) -> None:
        accepted = set(SamplingPatternRegistry.list_accepted())
        assert accepted == set(SamplingPatternRegistry.list_canonical()) | set(
            SamplingPatternRegistry.ALIASES
        )

    def test_every_inprogress_arm_declares_an_accepted_name(self) -> None:
        """The registry is only the owner if it accepts what the corpus declares."""
        import yaml

        from tests.utils.corpus import repo_root, tracked_yamls

        accepted = set(SamplingPatternRegistry.list_accepted())
        unknown: dict[str, int] = {}
        # ``repo_root()`` rather than ``parents[4]``: a hardcoded index is the
        # defect tests.utils.corpus exists to remove, and it is silent here --
        # a wrong index makes ``root`` a directory that does not exist, which
        # takes the skip below and reports "corpus not present" for a corpus
        # that is present.
        root = repo_root() / "experiments/inprogress"
        if not root.is_dir():
            # ``rglob`` on a missing directory yields nothing and raises nothing,
            # so without this the loop below never runs, ``unknown`` stays empty,
            # and the final assert passes -- a corpus test reporting green having
            # read zero arms. The public export ships no experiments/ tree, which
            # makes that reachable rather than hypothetical.
            pytest.skip(f"corpus not present: {root}")
        scanned = 0
        # Tracked files only. An rglob here has a different subject on every
        # machine -- the incident tests.utils.corpus records cost a cluster job
        # 24 failures against two arms that exist in no git history.
        for path in tracked_yamls(root):
            try:
                doc = yaml.safe_load(path.read_text())
            except Exception:
                continue
            if not isinstance(doc, dict):
                continue
            block = doc.get("undersampling")
            if isinstance(block, dict) and block.get("acceleration_type"):
                name = str(block["acceleration_type"]).lower()
                if name not in accepted:
                    unknown[name] = unknown.get(name, 0) + 1
            scanned += 1
        # A present-but-empty corpus is the same vacuity with a different cause,
        # so the count is asserted rather than the emptiness of ``unknown`` alone.
        assert scanned, f"no YAML read under {root} -- nothing was checked"
        assert not unknown, f"inprogress arms declare unresolvable patterns: {unknown}"


class TestAccelerator:
    def test_accelerator_for_returns_the_registry_entry(self) -> None:
        from spectramr.infrastructure.physics.sampling import (
            UniformCartesianKSpaceAccelerator,
        )

        assert (
            SamplingPatternRegistry.accelerator_for("cartesian_equispaced")
            is UniformCartesianKSpaceAccelerator
        )
