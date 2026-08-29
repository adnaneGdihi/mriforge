from unittest.mock import MagicMock

import torch
import torchio as tio

from mriforge.data.builders.torchio_subject_builder import FastMRISubjectBuilder


class TestFastMRIPropagation:
    def test_mask_propagation(self):
        """Test that mask is propagated from primary_io to subject."""

        # Mock IO Strategy
        mock_io = MagicMock()

        # Scenario 1: Mask in top-level dict
        mask_tensor = torch.ones(1, 320, 320, 1)
        mock_io.load.return_value = {
            "image": torch.randn(1, 320, 320, 1),
            "mask": mask_tensor,  # Explicit mask
            "affine": torch.eye(4),
        }

        builder = FastMRISubjectBuilder(primary_io=mock_io)
        record = {"primary_path": "dummy.h5"}

        subject = builder.build(record)

        assert "mask" in subject
        assert isinstance(subject["mask"], tio.LabelMap)
        assert torch.allclose(subject["mask"].data, mask_tensor)

    def test_mask_propagation_from_physics_dict(self):
        """Test that mask is propagated from physics dict."""

        mock_io = MagicMock()
        mask_tensor = torch.zeros(1, 320, 320, 1)  # Zeros mask

        mock_io.load.return_value = {
            "image": torch.randn(1, 320, 320, 1),
            "physics": {"mask": mask_tensor},
            "affine": torch.eye(4),
        }

        builder = FastMRISubjectBuilder(primary_io=mock_io)
        record = {"primary_path": "dummy.h5"}

        subject = builder.build(record)

        assert "mask" in subject
        assert torch.allclose(subject["mask"].data, mask_tensor)

    def test_sampling_mask_alias(self):
        """Test that sampling_mask alias works."""

        mock_io = MagicMock()
        mask_tensor = torch.ones(1, 320, 320, 1)

        mock_io.load.return_value = {
            "image": torch.randn(1, 320, 320, 1),
            "sampling_mask": mask_tensor,
            "affine": torch.eye(4),
        }

        builder = FastMRISubjectBuilder(primary_io=mock_io)
        record = {"primary_path": "dummy.h5"}

        subject = builder.build(record)

        assert "mask" in subject
        assert torch.allclose(subject["mask"].data, mask_tensor)
