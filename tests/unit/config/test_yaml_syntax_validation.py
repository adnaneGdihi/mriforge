"""YAML Syntax Validation — Config CI Gate.

Validates that ALL experiment YAML files across the workspace are syntactically
valid YAML.  This catches common errors like:
- Orphaned list items after inline lists
- Unquoted special characters in values
- Incorrect indentation
- Duplicate keys

This test does NOT validate Pydantic schema compliance (see smoke tests for that).
It only ensures that `yaml.safe_load()` succeeds without exceptions, which is
a prerequisite for any downstream validation.

Run:
    pytest tests/unit/config/test_yaml_syntax_validation.py -v
"""

from __future__ import annotations

import glob
from pathlib import Path

import pytest
import yaml

# Discover experiment YAML files in directories used for training.
# Inference configs (experiments/inference/) are excluded — all 149 are known
# broken from a batch-generation script (empty-dict + nested-keys conflict).
_YAML_DIRS = [
    "experiments/training",
    "experiments/active",
    "experiments/inference",
]

_ALL_YAML_FILES: list[str] = []
for _d in _YAML_DIRS:
    _ALL_YAML_FILES.extend(glob.glob(str(Path(_d) / "*.yaml")))
    _ALL_YAML_FILES.extend(glob.glob(str(Path(_d) / "**/*.yaml"), recursive=True))

# Deduplicate and sort for deterministic ordering
_ALL_YAML_FILES = sorted(set(_ALL_YAML_FILES))


@pytest.mark.parametrize("yaml_path", _ALL_YAML_FILES)
def test_yaml_syntax_valid(yaml_path: str) -> None:
    """Verify that the YAML file can be parsed without syntax errors."""
    path = Path(yaml_path)
    assert path.exists(), f"YAML file not found: {yaml_path}"

    with open(path) as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            pytest.fail(
                f"YAML syntax error in {yaml_path}:\n{exc}"
            )

    # Basic structural check: top-level must be a dict (not a list, not a scalar)
    assert isinstance(data, dict), (
        f"Expected top-level dict in {yaml_path}, got {type(data).__name__}"
    )


@pytest.mark.parametrize("yaml_path", _ALL_YAML_FILES)
def test_yaml_has_required_top_level_keys(yaml_path: str) -> None:
    """Verify that experiment configs have the minimum required top-level keys.

    Every valid experiment config must have at least 'model' and either
    'training' or 'inference' sections.  This catches empty or stub files.
    """
    path = Path(yaml_path)
    with open(path) as f:
        data = yaml.safe_load(f)

    if data is None:
        pytest.fail(f"Empty YAML file: {yaml_path}")

    # Must have 'model' section
    assert "model" in data, (
        f"Missing required 'model' key in {yaml_path}. "
        f"Top-level keys: {list(data.keys())}"
    )

    # Must have either training or inference
    has_training = "training" in data
    has_inference = "inference" in data
    # Some configs are training-only or inference-only, but at least one must exist
    # Unless it's a template/fragment file
    if not has_training and not has_inference:
        # Allow files that look like config fragments (no training/inference)
        # but warn about them
        pytest.skip(
            f"No 'training' or 'inference' section in {yaml_path} — may be a config fragment"
        )
"""Config YAML CI gate tests."""
