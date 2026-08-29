"""Tests for DatasetRegistry — duplicate-registration guard (WS-3).

Regression for the silent-overwrite fallback (`# warning or overwrite? pass`):
a conflicting duplicate must now raise, idempotent re-registration is allowed.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from mriforge.domain.entities.data import DatasetEntity
from mriforge.domain.entities.data.dataset_registry import (
    DatasetRegistry,
    get_dataset_registry,
)


@pytest.fixture
def _clean_registry() -> Iterator[None]:
    snap = dict(DatasetRegistry._registry)
    DatasetRegistry._registry.clear()
    try:
        yield
    finally:
        DatasetRegistry._registry.clear()
        DatasetRegistry._registry.update(snap)


def _entity(dataset_id: str = "d1", **over: object) -> DatasetEntity:
    base: dict[str, object] = dict(
        dataset_id=dataset_id,
        name="n",
        data_type="mri",
        path="/p",
        size=1,
        dimensions=(2, 2),
    )
    base.update(over)
    return DatasetEntity(**base)  # type: ignore[arg-type]


def test_register_and_get(_clean_registry: None) -> None:
    e = _entity()
    DatasetRegistry.register(e)
    assert DatasetRegistry.get("d1") is e


def test_idempotent_reregister_same_object(_clean_registry: None) -> None:
    e = _entity()
    DatasetRegistry.register(e)
    DatasetRegistry.register(e)  # same object -> allowed (idempotent)
    assert DatasetRegistry.get("d1") is e


def test_conflicting_duplicate_raises(_clean_registry: None) -> None:
    DatasetRegistry.register(_entity(name="a"))
    with pytest.raises(ValueError, match="already registered"):
        DatasetRegistry.register(_entity(name="b"))  # same id, different value


def test_get_missing_raises(_clean_registry: None) -> None:
    with pytest.raises(KeyError):
        DatasetRegistry.get("__missing__")


def test_list_datasets_returns_copy(_clean_registry: None) -> None:
    DatasetRegistry.register(_entity())
    snapshot = DatasetRegistry.list_datasets()
    snapshot.clear()
    assert "d1" in DatasetRegistry.list_datasets()  # original unaffected


def test_global_instance_accessor(_clean_registry: None) -> None:
    assert isinstance(get_dataset_registry(), DatasetRegistry)


def test_data_package_all_is_minimal() -> None:
    # __all__ was over-broad (exported stdlib names Any/dataclass/datetime/field).
    #
    # `MRIBatch` left the set with D23: it was a fourth batch declaration
    # competing with `data.batch_types.TrainingBatch` (309 references), and it
    # had ZERO importers. Narrowing here is the same direction this test already
    # pushes -- minimal means minimal.
    import mriforge.domain.entities.data as data_pkg

    assert set(data_pkg.__all__) == {"DatasetEntity"}
