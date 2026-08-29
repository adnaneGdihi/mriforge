"""Unit tests for models/parameter_normalization.py.

Covers: ParameterNormalizer, ModelParameterSchema, ParameterMapping,
        normalize_model_parameters, validate_model_parameters,
        get_parameter_normalizer.
"""

from __future__ import annotations

import pytest

from mriforge.models.parameter_normalization import (
    ModelParameterSchema,
    ParameterCategory,
    ParameterMapping,
    ParameterNormalizer,
    normalize_model_parameters,
    validate_model_parameters,
)


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_normalizer_canary():
    """Default schemas are registered on construction."""
    normalizer = ParameterNormalizer()
    schemas = normalizer.get_available_schemas()
    assert len(schemas) > 0
    assert "standard_unet" in schemas


# ---------------------------------------------------------------------------
# Parametrized: normalize_parameters
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_type,kwargs,expected_key",
    [
        ("standard_unet", {"in_channels": 1, "out_channels": 1}, "in_channels"),
        ("patchgan", {"in_channels": 2}, "in_channels"),
        ("edsr", {"in_channels": 1, "out_channels": 1}, "in_channels"),
    ],
)
def test_normalize_parameters_identity(model_type, kwargs, expected_key):
    """Schema identity mappings preserve the key name."""
    normalizer = ParameterNormalizer()
    result = normalizer.normalize_parameters(model_type, kwargs)
    assert expected_key in result


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_type",
    ["standard_unet", "enhanced_unet", "edsr", "patchgan", "restormer"],
)
def test_validate_ok_with_required_params(model_type):
    """Supplying all required params yields empty error list."""
    normalizer = ParameterNormalizer()
    schema = normalizer.get_schema(model_type)
    if schema is None:
        pytest.skip(f"No schema for {model_type}")
    kwargs = {p: 1 for p in schema.required_params}
    errors = normalizer.validate_parameters(model_type, kwargs)
    assert errors == [], f"Unexpected errors: {errors}"


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_register_custom_schema():
    """Custom schemas can be registered and looked up."""
    normalizer = ParameterNormalizer()
    schema = ModelParameterSchema(
        model_type="my_custom_model",
        mappings=[
            ParameterMapping("old_key", "new_key", ParameterCategory.OTHER)
        ],
        required_params={"new_key"},
    )
    normalizer.register_schema(schema)
    assert "my_custom_model" in normalizer.get_available_schemas()


@pytest.mark.unit
def test_global_mapping_applied():
    """A global mapping is applied before model-specific mappings."""
    normalizer = ParameterNormalizer()
    normalizer.add_global_mapping(
        ParameterMapping("alias", "real_name", ParameterCategory.OTHER)
    )
    result = normalizer.normalize_parameters("standard_unet", {"alias": 99})
    assert result.get("real_name") == 99
    assert "alias" not in result


@pytest.mark.unit
def test_convenience_normalize_returns_dict():
    result = normalize_model_parameters("standard_unet", {"in_channels": 1, "out_channels": 1})
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Expected failure
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validate_missing_required_param_returns_error():
    """Missing required param must produce a non-empty error list."""
    errors = validate_model_parameters("standard_unet", {})
    assert len(errors) > 0
    assert any("Missing required" in e for e in errors)


@pytest.mark.unit
def test_validate_unknown_model_returns_error():
    """Validating an unregistered model type returns an error message."""
    errors = validate_model_parameters("totally_unknown_xyz", {"in_channels": 1})
    assert len(errors) > 0
    assert any("No parameter schema" in e for e in errors)
