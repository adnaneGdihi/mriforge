__all__ = ["DatasetEntity"]

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class DatasetEntity:
    """Domain entity representing a dataset in the MRIForge system.

    ### Dataset Hierarchy
    ```mermaid
    classDiagram
        class DatasetEntity {
            +str dataset_id
            +str name
            +str data_type
            +str path
            +int size
            +tuple dimensions
            +dict normalization_params
            +datetime created_at
            +to_dict() dict
        }
    ```
    """

    dataset_id: str
    name: str
    data_type: str  # 'mri', 'synthetic', etc.
    path: str
    size: int  # number of samples
    dimensions: tuple[int, ...]
    normalization_params: dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Converts the entity to a dictionary representation.

        Returns:
            dict[str, Any]: Dictionary containing all entity attributes
                with datetime objects converted to ISO format strings.

        """
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "data_type": self.data_type,
            "path": self.path,
            "size": self.size,
            "dimensions": self.dimensions,
            "normalization_params": self.normalization_params,
            "created_at": self.created_at.isoformat(),
        }


# ``MRIBatch`` (a dataclass, ~65 lines with its own ``to(device)``) lived here and
# had ZERO importers -- a fourth batch declaration competing with
# ``data.batch_types.TrainingBatch`` (309 references), ``data.structures.MRIBatch``
# and ``domain.entities.data.types.MRIBatchDict``. Deleted with D23; TrainingBatch
# is the elected container. ``MRIBatchDict`` stays: it is the domain-layer protocol
# type ``domain/training.py`` annotates against, which is a different job.
