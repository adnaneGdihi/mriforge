from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ModelEntity:
    """Domain entity representing a neural network model in the spectraMR system.

    ### Model Entity Hierarchy
    ```mermaid
    classDiagram
        class ModelEntity {
            +str model_id
            +str model_type
            +str architecture
            +dict parameters
            +datetime created_at
            +str version
            +name str
            +to_dict() dict
        }
    ```
    """

    model_id: str
    model_type: str
    architecture: str
    parameters: dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"

    @property
    def name(self) -> str:
        """Returns the formatted model name combining type and ID.

        Returns:
            str: Formatted name in the format '{model_type}_{model_id}'.

        """
        return f"{self.model_type}_{self.model_id}"

    def to_dict(self) -> dict[str, Any]:
        """Converts the entity to a dictionary representation.

        This method is useful for serialization, logging, and data persistence.
        All datetime objects are converted to ISO format strings.

        Returns:
            dict[str, Any]: Dictionary containing all entity attributes.

        """
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "architecture": self.architecture,
            "parameters": self.parameters,
            "created_at": self.created_at.isoformat(),
            "version": self.version,
        }
