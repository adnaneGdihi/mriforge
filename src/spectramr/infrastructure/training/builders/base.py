"""Abstract Builder Base Class

Defines the interface for all builders following the fluent API pattern.
"""

from abc import ABC, abstractmethod
from typing import Any


class Builder(ABC):
    """Abstract base class for all builders.

    Builders follow the fluent interface pattern, returning self
    from each build method to enable method chaining.

    Example:
        >>> builder = ModelBuilder(config, device)
        >>> models = (builder
        ...     .build_generator()
        ...     .build_discriminator()
        ...     .validate()
        ...     .build())
    """

    @abstractmethod
    def build(self) -> Any:
        """Return the built product.

        This method must be implemented by all concrete builders.
        It should return the final constructed object(s).

        Raises:
            ValueError: If required components are missing

        Returns:
            Any: The built product (dict, tuple, or single object)
        """
        pass

    def validate(self) -> "Builder":
        """Validate the current builder state.

        Override in subclasses to add specific validation logic.
        This method should check that all required components are present
        and properly configured before build() is called.

        Returns:
            self: To enable method chaining

        Raises:
            ValueError: If validation fails
        """
        return self
