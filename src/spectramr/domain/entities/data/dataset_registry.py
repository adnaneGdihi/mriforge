from spectramr.domain.entities.data import DatasetEntity


class DatasetRegistry:
    """Registry for managing known datasets and their metadata."""

    _registry: dict[str, DatasetEntity] = {}

    @classmethod
    def register(cls, entity: DatasetEntity) -> None:
        """Register a dataset entity.

        Raises on a *conflicting* duplicate — a different entity already
        registered under the same ``dataset_id``. A silent overwrite (the
        previous behaviour) would flip which dataset an id resolves to depending
        on registration/import order, the silent-fallback class CLAUDE.md
        pitfall #9 forbids (and which ``LossRegistry`` already guards against).
        Re-registering an equal entity is idempotent.
        """
        existing = cls._registry.get(entity.dataset_id)
        if existing is not None and existing != entity:
            raise ValueError(
                f"Dataset '{entity.dataset_id}' is already registered to a "
                f"different entity; refusing to silently overwrite it. "
                f"Use a distinct dataset_id."
            )
        cls._registry[entity.dataset_id] = entity

    @classmethod
    def get(cls, dataset_id: str) -> DatasetEntity:
        """Get a dataset entity by ID."""
        if dataset_id not in cls._registry:
            raise KeyError(f"Dataset {dataset_id} not found in registry")
        return cls._registry[dataset_id]

    @classmethod
    def list_datasets(cls) -> dict[str, DatasetEntity]:
        """list_datasets.

        Returns:
            dict[str, DatasetEntity]: Description.
        """
        return cls._registry.copy()


# Global registry instance
_instance = DatasetRegistry()


def get_dataset_registry() -> DatasetRegistry:
    """Get the global dataset registry instance."""
    return _instance
