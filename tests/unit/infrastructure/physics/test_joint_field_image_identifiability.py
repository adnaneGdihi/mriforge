"""Oracle tests for joint field-image identifiability certs (A-3.2 JIFI)."""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.infrastructure.physics.joint_field_image_identifiability import (
    JointFieldImageIdentifiabilityCert,
    field_image_confound_angle,
    joint_condition_number,
    joint_identifiability_rank,
)

# --------------------------------------------------------------------------- #
# Confound angle: orthogonal -> pi/2, shared direction -> 0
# --------------------------------------------------------------------------- #


def test_orthogonal_subspaces_give_right_angle() -> None:
    # Image Jacobian spans the first 2 axes, field the next 2 -> orthogonal -> pi/2.
    j_img = torch.zeros(6, 2)
    j_img[0, 0] = j_img[1, 1] = 1.0
    j_field = torch.zeros(6, 2)
    j_field[2, 0] = j_field[3, 1] = 1.0
    assert field_image_confound_angle(j_img, j_field) == pytest.approx(math.pi / 2, abs=1e-5)


def test_shared_direction_gives_zero_angle() -> None:
    # A field column lies in the image span -> fully confounded -> angle 0.
    j_img = torch.eye(6, 3)
    j_field = j_img[:, 0:1].clone()  # field == an image column
    assert field_image_confound_angle(j_img, j_field) == pytest.approx(0.0, abs=1e-5)


def test_confound_angle_in_valid_range() -> None:
    torch.manual_seed(0)
    j_img = torch.randn(20, 3)
    j_field = torch.randn(20, 2)
    angle = field_image_confound_angle(j_img, j_field)
    assert 0.0 <= angle <= math.pi / 2 + 1e-6


# --------------------------------------------------------------------------- #
# Rank certificate: full rank vs confounded
# --------------------------------------------------------------------------- #


def test_full_rank_when_separable() -> None:
    torch.manual_seed(1)
    j_img = torch.randn(30, 3)
    j_field = torch.randn(30, 2)
    info = joint_identifiability_rank(j_img, j_field)
    assert info["full_rank"] is True
    assert info["rank"] == 5 and info["n_params"] == 5
    assert info["min_singular_value"] > 0.0


def test_rank_deficient_when_field_confounded_with_image() -> None:
    # field column = an image column -> stacked Jacobian is rank-deficient.
    j_img = torch.eye(8, 3)
    j_field = j_img[:, 1:2].clone()
    info = joint_identifiability_rank(j_img, j_field)
    assert info["full_rank"] is False
    assert info["rank"] == 3 and info["n_params"] == 4
    assert info["min_singular_value"] == pytest.approx(0.0, abs=1e-6)


def test_condition_number_infinite_when_confounded() -> None:
    j_img = torch.eye(8, 3)
    j_field = j_img[:, 0:1].clone()
    assert math.isinf(joint_condition_number(j_img, j_field))


def test_condition_number_finite_when_separable() -> None:
    torch.manual_seed(2)
    j_img = torch.randn(40, 2)
    j_field = torch.randn(40, 2)
    assert math.isfinite(joint_condition_number(j_img, j_field))


# --------------------------------------------------------------------------- #
# The multi-echo story: extra echoes raise the confound angle (identifiability)
# --------------------------------------------------------------------------- #


def test_multi_echo_raises_confound_angle() -> None:
    # Single echo: field (a phase ramp) nearly aligns with an image column.
    # Two echoes with distinct TE give the field a different phase evolution,
    # rotating its column away from the image span -> larger confound angle.
    base = torch.randn(8, 1)
    # image column repeated across echoes (no echo-dependent change);
    # field column scaled differently per echo (TE-dependent phase accrual).
    img_single = base.clone()
    field_single = base + 0.02 * torch.randn(8, 1)  # nearly collinear -> small angle
    angle_single = field_image_confound_angle(img_single, field_single)

    img_multi = torch.cat([base, base], dim=0)  # same image across 2 echoes
    field_multi = torch.cat([base, 3.0 * base], dim=0) + 0.02 * torch.randn(16, 1)
    angle_multi = field_image_confound_angle(img_multi, field_multi)
    assert angle_multi > angle_single


# --------------------------------------------------------------------------- #
# Cert bundle + complex Jacobian
# --------------------------------------------------------------------------- #


def test_cert_bundle_identifiable_vs_confounded() -> None:
    torch.manual_seed(3)
    cert = JointFieldImageIdentifiabilityCert()
    sep = cert.certify(torch.randn(20, 3), torch.randn(20, 2))
    assert sep["identifiable"] is True and sep["full_rank"] is True

    j_img = torch.eye(8, 3)
    conf = cert.certify(j_img, j_img[:, 0:1].clone())
    assert conf["identifiable"] is False
    assert conf["min_principal_angle"] == pytest.approx(0.0, abs=1e-5)


def test_complex_jacobian_supported() -> None:
    j_img = torch.randn(12, 2, dtype=torch.cfloat)
    j_field = torch.randn(12, 1, dtype=torch.cfloat)
    info = joint_identifiability_rank(j_img, j_field)
    assert info["n_params"] == 3 and isinstance(info["rank"], int)
