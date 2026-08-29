"""Optional Imports Module

Provides lazy import helpers for optional dependencies (nibabel, pydicom, h5py).
These dependencies may not be available in all environments and are only
loaded when explicitly needed.
"""


def get_nibabel():
    """Get nibabel module with lazy loading."""
    try:
        import nibabel

        return nibabel
    except ImportError as e:
        raise ImportError(
            "nibabel is required for NIfTI file support. Install it with: pip install nibabel"
        ) from e


def get_pydicom():
    """Get pydicom module with lazy loading."""
    try:
        import pydicom

        return pydicom
    except ImportError as e:
        raise ImportError(
            "pydicom is required for DICOM file support. Install it with: pip install pydicom"
        ) from e


def get_h5py():
    """Get h5py module with lazy loading."""
    try:
        import h5py

        return h5py
    except ImportError as e:
        raise ImportError(
            "h5py is required for HDF5 file support. Install it with: pip install h5py"
        ) from e


__all__ = [
    "OptionalDependencies",
    "get_h5py",
    "get_nibabel",
    "get_pydicom",
    "get_safetensors",
    "get_torch_fidelity",
    "has_h5py",
    "has_safetensors",
    "has_torch_fidelity",
]


class OptionalDependencies:
    """Unified interface for optional dependencies.

    Provides lazy loading and availability checks for optional packages.

    Usage:
        OPTIONAL = OptionalDependencies()
        if OPTIONAL.has_h5py():
            h5py = OPTIONAL.get_h5py()
            with h5py.File(path, 'r') as f:
                ...
    """

    @staticmethod
    def get_nibabel():
        """Get nibabel module with lazy loading."""
        return get_nibabel()

    @staticmethod
    def get_pydicom():
        """Get pydicom module with lazy loading."""
        return get_pydicom()

    @staticmethod
    def get_h5py():
        """Get h5py module with lazy loading."""
        return get_h5py()

    @staticmethod
    def has_h5py() -> bool:
        """Check if h5py is available."""
        try:
            import h5py  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def has_nibabel() -> bool:
        """Check if nibabel is available."""
        try:
            import nibabel  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def has_pydicom() -> bool:
        """Check if pydicom is available."""
        try:
            import pydicom  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def has_torch_fidelity() -> bool:
        """Check if torch_fidelity is available."""
        return has_torch_fidelity()

    @staticmethod
    def get_torch_fidelity():
        """Get torch_fidelity module with lazy loading."""
        return get_torch_fidelity()

    @staticmethod
    def has_sigpy() -> bool:
        """Check if sigpy is available (used for ESPIRiT coil sensitivity)."""
        try:
            import sigpy  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def get_sigpy():
        """Get sigpy module with lazy loading."""
        try:
            import sigpy

            return sigpy
        except ImportError as e:
            raise ImportError(
                "sigpy is required for ESPIRiT coil sensitivity estimation. "
                "Install it with: pip install sigpy"
            ) from e

    @staticmethod
    def get_safetensors():
        """Get safetensors module with lazy loading."""
        return get_safetensors()

    @staticmethod
    def has_safetensors() -> bool:
        """Check if safetensors is available."""
        return has_safetensors()


def has_h5py() -> bool:
    """Check if h5py is available."""
    try:
        import h5py  # noqa: F401

        return True
    except ImportError:
        return False


def has_torch_fidelity() -> bool:
    """Check if torch_fidelity is available."""
    try:
        import torch_fidelity  # noqa: F401

        return True
    except ImportError:
        return False


def get_torch_fidelity():
    """Get torch_fidelity module with lazy loading."""
    try:
        import torch_fidelity

        return torch_fidelity
    except ImportError as e:
        raise ImportError(
            "torch_fidelity is required for FID/IS metrics. "
            "Install it with: pip install torch-fidelity"
        ) from e


def get_safetensors():
    """Get safetensors module with lazy loading."""
    try:
        import safetensors

        return safetensors
    except ImportError as e:
        raise ImportError(
            "safetensors is required for .safetensors file support. "
            "Install it with: pip install safetensors"
        ) from e


def has_safetensors() -> bool:
    """Check if safetensors is available."""
    try:
        import safetensors  # noqa: F401

        return True
    except ImportError:
        return False
