"""Auto-mark every test in tests/smoke/exotic/ with @pytest.mark.smoke.

These are smoke tests for exotic / experimental models (SOTA 2.0 candidates,
Mamba variants, KAN variants, 3D Gaussian splatting, etc.). They were
relocated here from the tests/ root in the 2026-05 test-suite reorganisation.
"""

import pytest


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "gpu" not in item.keywords and "slow" not in item.keywords:
            item.add_marker(pytest.mark.smoke)
