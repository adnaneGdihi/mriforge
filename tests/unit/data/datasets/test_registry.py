"""Contract tests for the dataset-type registry (CLAUDE.md non-negotiable #6).

``DatasetInstantiator.create_datasets`` resolved ``dataset_type`` through a
21-branch ``if/elif`` chain -- the one component family without a registry. The
chain was not merely inelegant: because its labels were hand-written and the
schema folds aliases BEFORE dispatch, ten labels were unreachable and one
canonical type had no branch at all. These tests pin the properties that make
that state impossible to re-enter.
"""

from __future__ import annotations

import pytest

from spectramr.config.schemas.data import CANONICAL_DATASET_TYPES
from spectramr.data.builders.dataset_instantiator import DatasetInstantiator
from spectramr.data.datasets.registry import (
    DATASET_REGISTRY,
    get_dataset_creator,
    list_registered_dataset_types,
    register_dataset,
)

#: The two types resolved by conditional routes rather than a name lookup:
#: ``manifest_roles`` is a predicate, and ``image`` depends on whether
#: ``source.index_path`` is set. Both are documented in the registry module.
RESIDUAL = {"kspace", "image"}


# Importing ``dataset_instantiator`` must be enough -- see the import-time
# population call at the bottom of that module. If it ever regresses to lazy
# population, the parametrised tests below silently collapse to an EMPTY
# parameter set and report a green skip instead of failing. The explicit call
# names that dependency instead of leaving it an invisible import side effect
# (which read to ruff as an unused import, one autofix away from deleting the
# line that makes every parametrised test below cover anything at all).
DatasetInstantiator._ensure_registered()
assert DATASET_REGISTRY, (
    "dataset registry is empty at import time; parametrised tests would "
    "silently cover nothing"
)


class TestRegistryCoversTheVocabulary:
    def test_every_canonical_type_is_registered_or_residual(self):
        missing = (
            set(CANONICAL_DATASET_TYPES)
            - set(list_registered_dataset_types())
            - RESIDUAL
        )
        assert not missing, f"canonical dataset_type(s) with no route: {missing}"

    def test_no_registered_name_is_outside_the_canonical_vocabulary(self):
        """A route for a type the schema rejects is unreachable by construction."""
        stray = set(list_registered_dataset_types()) - set(CANONICAL_DATASET_TYPES)
        assert not stray, f"registered but not canonical: {stray}"

    def test_registry_and_residual_partition_the_vocabulary_exactly(self):
        assert set(list_registered_dataset_types()) | RESIDUAL == set(
            CANONICAL_DATASET_TYPES
        )


class TestRouteShape:
    @pytest.mark.parametrize("name", sorted(DATASET_REGISTRY))
    def test_every_route_is_callable_and_declares_its_arity(self, name):
        entry = get_dataset_creator(name)
        assert callable(entry.fn)
        assert isinstance(entry.indexed, bool)

    def test_indexed_and_self_indexed_both_present(self):
        """The two real signature families; collapsing them would break one."""
        kinds = {get_dataset_creator(n).indexed for n in DATASET_REGISTRY}
        assert kinds == {True, False}

    def test_self_indexed_routes_match_the_directors_skip_set(self):
        """A self-indexed dataset must also skip the ManifestLoader pre-split.

        These are two encodings of one fact, and they drifted in BOTH
        directions: ``graph_mri`` sat in the skip-set with no construction route
        at all, and seven routes with ``indexed=False`` were missing from the
        skip-set, so the director ran a fastMRI H5 pre-split for them.

        This assertion was one-directional when the skip-set was hand-written --
        which is why it stayed green over the second, larger half of the drift.
        Now that the set is derived, assert equality.
        """
        from spectramr.infrastructure.builders.directors.data_pipeline_director import (
            _self_indexed_dataset_types,
        )

        skip_set = _self_indexed_dataset_types()
        for name in skip_set:
            entry = get_dataset_creator(name)
            assert entry is not None, f"{name} is in the skip-set with no route"
            assert (
                not entry.indexed
            ), f"{name} skips the manifest pre-split but its route expects an index"

        missing = {n for n, e in DATASET_REGISTRY.items() if not e.indexed} - skip_set
        assert not missing, (
            f"self-indexed routes absent from the skip-set: {sorted(missing)} -- "
            "each one runs a fastMRI H5 pre-split that raises before its own "
            "creator is called"
        )


class TestDuplicateRegistration:
    def test_two_creators_under_one_name_raises(self):
        """Import-order-dependent resolution is a defect, not a warning."""

        def _a(*args, **kwargs):
            return ("t", "v")

        def _b(*args, **kwargs):
            return ("t", "v")

        try:
            register_dataset("_probe_dupe", _a, indexed=False)
            with pytest.raises(ValueError, match="already registered"):
                register_dataset("_probe_dupe", _b, indexed=False)
        finally:
            DATASET_REGISTRY.pop("_probe_dupe", None)

    def test_re_registering_the_same_creator_is_idempotent(self):
        """``_ensure_registered`` may run from several entry points."""

        def _a(*args, **kwargs):
            return ("t", "v")

        try:
            register_dataset("_probe_idem", _a, indexed=False)
            register_dataset("_probe_idem", _a, indexed=False)
            assert get_dataset_creator("_probe_idem").fn is _a
        finally:
            DATASET_REGISTRY.pop("_probe_idem", None)


def test_unknown_name_returns_none_not_a_raise():
    """``None`` is a legitimate answer -- the residual routes handle it."""
    assert get_dataset_creator("definitely_not_a_type") is None
