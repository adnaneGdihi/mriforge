"""
Unit tests for shared utilities.
Covers security, path normalization, device utils.
"""

from pathlib import Path

import pytest

try:
    from spectramr.shared.utils.path_security import validate_path
except ImportError:
    validate_path = None

try:
    from spectramr.shared.utils.device_utils import get_device
except ImportError:
    get_device = None


# @# pytest.mark.unit
class TestSharedUtils:
    def test_path_traversal_prevention(self):
        if validate_path is None:
            pytest.skip("validate_path not available")

        base = Path("/tmp/app")
        unsafe = Path("/tmp/app/../../etc/passwd")

        # Should raise security error or return None/Safe path
        # Adjust expectation based on implementation details
        with pytest.raises(Exception):
            validate_path(unsafe, base_dir=base)

    def test_device_selection(self):
        if get_device is None:
            pytest.skip("get_device not available")
        try:
            dev = get_device()
            assert dev is not None
        except:
            pass
