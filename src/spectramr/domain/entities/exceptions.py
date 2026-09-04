"""Domain Entities Exceptions (Back-compat)
Restored for backward compatibility with stale cluster code.
"""


class SPECTRAMRError(Exception):
    """Base exception for all spectraMR errors.

    ### Exception Hierarchy
    ```mermaid
    classDiagram
        Exception <|-- SPECTRAMRError
        SPECTRAMRError <|-- ConfigurationError
        SPECTRAMRError <|-- BufferAllocationError
    ```
    """


class ConfigurationError(SPECTRAMRError):
    """Raised when configuration is invalid or missing."""

    pass


class BufferAllocationError(SPECTRAMRError):
    """Raised when tensor buffer allocation fails."""

    pass
