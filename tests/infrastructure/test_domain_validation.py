from unittest.mock import patch

import pytest

from spectramr.infrastructure.domain_validation import (
    DomainMismatchWarning,
    DomainValidationResult,
    _get_loss_domain,
    _normalize_loss_name,
    get_domain_report,
    register_loss_domain,
    validate_loss_domains,
    verify_startup_loss_domains,
)


@pytest.mark.parametrize(
    "input_name,expected",
    [("mae", "l1"), ("MSE", "l2"), ("unknown_loss", "unknown_loss")],
)
def test_normalize_loss_name(input_name, expected):
    assert _normalize_loss_name(input_name) == expected


@pytest.mark.parametrize(
    "loss_name,expected_domain",
    [("complex_l1", "kspace"), ("mae", "agnostic"), ("non_existent_loss", None)],
)
def test_get_loss_domain(loss_name, expected_domain):
    meta = _get_loss_domain(loss_name)
    if expected_domain is None:
        assert meta is None
    else:
        assert meta is not None
        assert meta["domain"] == expected_domain


@patch("spectramr.infrastructure.domain_validation.LossBuilder")
def test_validate_loss_domains_valid(mock_builder):
    mock_instance = mock_builder.return_value
    mock_instance.get_enabled_losses.return_value = {"complex_l1": 1.0, "l1": 0.5}

    result = validate_loss_domains({}, "kspace")
    assert result.is_valid is True
    assert len(result.errors) == 0
    assert len(result.warnings) == 0


@patch("spectramr.infrastructure.domain_validation.LossBuilder")
def test_validate_loss_domains_warning(mock_builder):
    mock_instance = mock_builder.return_value
    mock_instance.get_enabled_losses.return_value = {"hfen": 1.0}

    # kspace input with hfen loss is a critical error (image domain)
    result = validate_loss_domains({}, "kspace")
    assert result.is_valid is False
    assert len(result.errors) == 1
    assert len(result.warnings) == 1
    assert result.warnings[0].severity == "error"

    # latent input with hfen loss is a warning (image domain but not critical kspace mismatch)
    result = validate_loss_domains({}, "latent")
    assert result.is_valid is True
    assert len(result.errors) == 0
    assert len(result.warnings) == 1
    assert result.warnings[0].severity == "warning"


@patch("spectramr.infrastructure.domain_validation.LossBuilder")
def test_validate_loss_domains_unknown_loss(mock_builder):
    mock_instance = mock_builder.return_value
    mock_instance.get_enabled_losses.return_value = {"custom_unknown_loss": 1.0}

    result = validate_loss_domains({}, "kspace")
    assert result.is_valid is True
    assert len(result.errors) == 0
    assert len(result.warnings) == 1
    assert result.warnings[0].loss_domain == "unknown"


def test_validate_loss_domains_invalid_input_domain():
    with pytest.raises(ValueError, match="Invalid input domain"):
        validate_loss_domains({}, "invalid_domain")


@patch("spectramr.infrastructure.domain_validation.LossBuilder")
def test_validate_loss_domains_builder_error(mock_builder):
    mock_builder.side_effect = Exception("Builder failed")
    result = validate_loss_domains({}, "kspace")
    assert result.is_valid is False
    assert len(result.errors) == 1
    assert "Builder failed" in result.errors[0]


@patch("spectramr.infrastructure.domain_validation.validate_loss_domains")
def test_verify_startup_loss_domains_success(mock_validate):
    mock_validate.return_value = DomainValidationResult(
        is_valid=True, errors=[], warnings=[]
    )
    assert verify_startup_loss_domains({}, "kspace") is True


@patch("spectramr.infrastructure.domain_validation.validate_loss_domains")
def test_verify_startup_loss_domains_failure(mock_validate):
    mock_validate.return_value = DomainValidationResult(
        is_valid=False,
        errors=["Critical error"],
        warnings=[DomainMismatchWarning("l", "d", "i", "warning", "msg", "fix")],
    )
    with pytest.raises(RuntimeError, match="Loss domain validation failed"):
        verify_startup_loss_domains({}, "kspace", fail_on_error=True)

    # Should not raise if fail_on_error is False
    assert verify_startup_loss_domains({}, "kspace", fail_on_error=False) is False


def test_register_loss_domain():
    register_loss_domain("new_loss", "kspace", ["agnostic"])
    meta = _get_loss_domain("new_loss")
    assert meta is not None
    assert meta["domain"] == "kspace"
    assert meta["compatible_with"] == ["agnostic"]

    with pytest.raises(ValueError, match="Invalid domain"):
        register_loss_domain("bad_loss", "invalid_domain")


@patch("spectramr.infrastructure.domain_validation.validate_loss_domains")
def test_get_domain_report(mock_validate):
    mock_validate.return_value = DomainValidationResult(
        is_valid=False,
        errors=["Error 1"],
        warnings=[
            DomainMismatchWarning("loss1", "kspace", "image", "warning", "msg", "fix")
        ],
    )
    report = get_domain_report({}, "image")
    assert "Loss Domain Validation Report" in report
    assert "Input Domain: image" in report
    assert "Status: 🔴 INVALID" in report
    assert "CRITICAL ERRORS:" in report
    assert "Error 1" in report
    assert "WARNINGS:" in report
    assert "loss1: msg" in report


@patch("spectramr.infrastructure.domain_validation.LossBuilder")
def test_image_losses_block_is_bridged_not_critical(mock_builder):
    """F35 — hfen/ssim declared under losses.image_losses are auto-bridged to
    image domain by LossBuilder, so they are NOT a critical error on a k-space
    model (regression for experiment_12_physics_cold_diffusion_v2, which was
    rejected at pipeline build with 'Loss hfen requires domain image but model
    uses kspace' despite being correctly placed under image_losses).
    """
    from types import SimpleNamespace

    mock_builder.return_value.get_enabled_losses.return_value = {
        "hfen": 1.0,
        "ssim": 0.5,
    }
    config = SimpleNamespace(
        image_losses=[
            SimpleNamespace(name="hfen", enabled=True),
            SimpleNamespace(name="ssim", enabled=True),
        ]
    )
    result = validate_loss_domains(config, "kspace")
    assert result.is_valid is True
    assert result.errors == []


@patch("spectramr.infrastructure.domain_validation.LossBuilder")
def test_image_loss_outside_image_block_still_critical(mock_builder):
    """An image-domain loss NOT declared under image_losses (no bridge) on a
    k-space model is still a critical error — the F35 demotion is scoped to
    bridged losses only.
    """
    from types import SimpleNamespace

    mock_builder.return_value.get_enabled_losses.return_value = {"hfen": 1.0}
    config = SimpleNamespace(image_losses=[])  # hfen NOT bridged
    result = validate_loss_domains(config, "kspace")
    assert result.is_valid is False
