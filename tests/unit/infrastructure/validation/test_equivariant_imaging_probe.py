r"""Tests for the Equivariant Imaging Tier-2 mechanism-fires probe (Phase A step 3).

``ei_forward_probe`` runs the EI loop on the configured model and asserts, on a
synthetic batch, that (i) the arm is numerically identifiable
(``sensing_margin > 0``) and (ii) the equivariance term is non-zero with a
gradient reaching the network — i.e. the previously-inert facade now fires
(pitfall #16). A fully-sampled arm must fail the identifiability leg.
"""

from __future__ import annotations

from mriforge.infrastructure.validation.equivariant_imaging_probe import (
    ei_forward_probe,
)


class _EI:
    def __init__(self, group="dihedral", robust_correction=False):
        self.group = group
        self.alpha_equivariance = 1.0
        self.n_group_samples = 1
        self.robust_correction = robust_correction
        self.noise_std_estimate = 0.01 if robust_correction else None
        self.noise_model = "gaussian_kspace"
        self.n_coils = 1
        self.max_angle_deg = 10.0
        self.num_angles = 4
        self.split_seed = 0


class _Model:
    def __init__(self):
        self.model_type = "standard_unet"
        self.in_channels = 2
        self.out_channels = 2
        self.model_kwargs = {"features": 16}
        self.denoising_model = None
        self.spatial_dims = 2


class _Data:
    def __init__(self):
        self.patch_size = [32, 32, 1]


class _CS:
    def __init__(self, acceleration_factor=4.0):
        self.acceleration_factor = acceleration_factor


class _Physics:
    def __init__(self, acceleration_factor=4.0):
        self.compressed_sensing = _CS(acceleration_factor)


class _Training:
    def __init__(self, **ei_kw):
        self.equivariant_imaging = _EI(**ei_kw)
        self.training_mode = "equivariant_imaging"


class _Config:
    def __init__(self, acceleration_factor=4.0, **ei_kw):
        self.training = _Training(**ei_kw)
        self.model = _Model()
        self.data = _Data()
        self.physics = _Physics(acceleration_factor)

        class _A:
            base_acceleration = acceleration_factor

        self.acceleration = _A()


def test_probe_passes_and_mechanism_fires():
    probe = ei_forward_probe(_Config(acceleration_factor=4.0), device="cpu")
    assert probe.passed, probe.message
    assert probe.category == "ei_mechanism_fires"


def test_probe_fails_when_fully_sampled_unidentifiable():
    probe = ei_forward_probe(_Config(acceleration_factor=1.0), device="cpu")
    assert not probe.passed
    assert "margin" in probe.message.lower() or "identif" in probe.message.lower()


def test_cli_audit_routes_ei_arm_to_ei_probe():
    """The audit --probe dispatch must route EI arms to ei_forward_probe."""
    import inspect

    from mriforge.cli import app

    src = inspect.getsource(app._audit_one)
    assert "ei_forward_probe" in src
    assert "equivariant_imaging" in src
