"""Domain Entities Exceptions (Back-compat)
Restored for backward compatibility with stale cluster code.
"""


class MRIFORGEError(Exception):
    """Base exception for all MRIForge errors.

    ### Exception Hierarchy
    ```mermaid
    classDiagram
        Exception <|-- MRIFORGEError
        MRIFORGEError <|-- ConfigurationError
        MRIFORGEError <|-- BufferAllocationError
    ```
    """


class ConfigurationError(MRIFORGEError):
    """Raised when configuration is invalid or missing."""

    pass


class BufferAllocationError(MRIFORGEError):
    """Raised when tensor buffer allocation fails."""

    pass
