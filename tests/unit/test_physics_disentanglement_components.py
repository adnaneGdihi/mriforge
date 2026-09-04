from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from spectramr.data.builders.torchio_subject_builder import FastMRISubjectBuilder
from spectramr.data.metadata.physics_registry import get_physics_vector
from spectramr.models.generators.disentangled_mri import DisentangledMRI


class TestPhysicsRegistry:
    def test_registry_values(self):
        # Check standard 64mT T1w
        # get_physics_vector returns [TR, TE, TI, B0, b_value, flip] — 6 elements
        vec = get_physics_vector("sub-01_acq-64mT_T1w.nii.gz")
        assert vec.shape == (6,)
        # TR, TE, TI, B0 should be non-zero for T1w
        assert torch.all(vec[:4] > 0)
        # b_value=0 for non-DWI sequences
        assert vec[4] == 0.0
        # flip angle is non-zero (90 deg -> normalized 1.0)
        assert vec[5] > 0
        # Check TR normalization (880 / 11000 approx 0.08)
        assert 0.0 < vec[0] < 0.2

    def test_registry_dwi(self):
        # Check 64mT DWI (heuristic: '64mT' + 'dwi')
        vec = get_physics_vector("sub-01_acq-64mT_dwi.nii.gz")
        assert vec.shape == (6,)
        # For DWI all 6 params are non-zero (b_value=900 s/mm² -> normalized 0.9)
        assert torch.all(vec > 0)
        # TR=1000, TE=76, TI=100, B0=0.064
        # Just check it's not zero and looks different from T1
        assert vec[0] > 0
        assert vec[3] < 0.1  # Low field

    def test_unknown_file(self):
        vec = get_physics_vector("unknown_file.nii.gz")
        assert vec.shape == (6,)
        assert torch.all(vec == 0)


class TestDisentangledMRI:
    def test_bloch_regressor_init(self):
        model = DisentangledMRI(in_channels=1, out_channels=1, dim=8, style_dim=4)
        assert hasattr(model, "bloch_regressor")
        assert isinstance(model.bloch_regressor, nn.Module)

    def test_predict_physics(self):
        model = DisentangledMRI(in_channels=1, out_channels=1, dim=8, style_dim=4)
        style = torch.randn(2, 4)  # [B, style_dim]
        physics = model.predict_physics(style)
        assert physics.shape == (2, 4)
        assert torch.all(physics >= 0) and torch.all(physics <= 1)  # Sigmoid output


class TestSubjectBuilderPhysics:
    @patch("spectramr.data.builders.torchio_subject_builder.get_physics_vector")
    def test_physics_injection(self, mock_get_physics):
        # Mock physics vector
        mock_vec = torch.tensor([0.1, 0.2, 0.3, 0.4])
        mock_get_physics.return_value = mock_vec

        # Mock IO
        mock_io = MagicMock()
        mock_io.load.return_value = {"image": torch.randn(1, 32, 32)}

        builder = FastMRISubjectBuilder(primary_io=mock_io)
        record = {"primary_path": "/data/sub-01_T1w.nii.gz"}

        subject = builder.build(record)

        assert "physics" in subject
        assert torch.equal(subject["physics"], mock_vec)
        # `target_physics` used to be asserted here ("should fallback to same
        # physics"). It is gone: tree-wide the input_physics/target_physics pair
        # had 9 writes and 0 reads, so the fallback maintained a value nothing
        # consumed (audit D21). `physics` itself stays and IS read.
        assert "target_physics" not in subject
