"""Regression: strategies must thread the live ``iteration`` into their loss
computer so the iteration-based warm-up / KL-anneal schedulers actually advance.

The ``Unified*LossComputer._get_loss_weight`` warm-up gate defaults ``iteration``
to **0**, so a spatial loss (``l1``/``perceptual``/``adversarial``) is held at
weight 0.0 while ``iteration < warmup_iterations`` (default 1000). If the
strategy calls ``loss_computer.compute(..., epoch=...)`` WITHOUT ``iteration``,
the gate never opens and the headline reconstruction terms contribute zero
gradient for the entire run — a silent facade (CLAUDE.md pitfall #16). The same
hole freezes VAE KL annealing at ``beta_start`` (default 0.0 → unregularised
autoencoder).

The live counter is available in ``_compute_losses_impl`` via ``kwargs``
(``pipelines/train.py`` calls ``strategy.train_step(..., iteration=iteration)``
→ base ``g_closure`` → ``_compute_losses(**kwargs)`` → ``_compute_losses_impl``).
These tests assert the seam forwards it.
"""

from __future__ import annotations

import types

import torch

from mriforge.infrastructure.training.strategies.disentangled_vae_strategy import (
    DisentangledVAETrainingStrategy,
)
from mriforge.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)
from mriforge.infrastructure.training.strategies.vae import VAETrainingStrategy
from mriforge.models.losses.computers.base import LossOutput

_ITER = 5000  # well past the default warmup_iterations=1000


class _RecordingComputer:
    """Records every ``compute(**kwargs)`` call so the test can assert the
    ``iteration`` kwarg was threaded through."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def compute(self, **kwargs):
        self.calls.append(kwargs)
        return LossOutput(
            total=torch.zeros((), requires_grad=True),
            components={},
            metrics={},
        )


def _new(cls):
    return object.__new__(cls)


def test_reconstruction_threads_iteration_to_loss_computer():
    s = _new(ReconstructionTrainingStrategy)
    rec = _RecordingComputer()
    target = torch.randn(1, 2, 4, 4)
    pred = torch.randn(1, 2, 4, 4, requires_grad=True)

    s.loss_computer = rec
    s.pinn_module = None
    s.logging_service = None
    s.device = torch.device("cpu")
    s._loss_dict_reuse = {}
    s.env = types.SimpleNamespace(losses={}, step=0)
    s.config = types.SimpleNamespace(model=types.SimpleNamespace(model_type="unet"))
    # Stub the forward / context helpers — we only exercise the loss seam.
    s._prepare_batch_context = lambda *a, **k: {"hr": target, "use_dc": False}
    s._prepare_generator_inputs = lambda *a, **k: (target, {})
    s._generate_predictions = lambda *a, **k: (pred, {})
    s._compute_training_metrics = lambda *a, **k: {}

    s._compute_losses_impl(target, target, epoch=0, iteration=_ITER)

    assert rec.calls, "loss_computer.compute was never called"
    assert rec.calls[0].get("iteration") == _ITER, (
        "ReconstructionTrainingStrategy dropped 'iteration' — spatial-loss "
        "warm-up gate stays closed forever (l1/perceptual/adversarial = 0)."
    )


def test_vae_threads_iteration_to_loss_computer():
    s = _new(VAETrainingStrategy)
    rec = _RecordingComputer()
    x = torch.randn(1, 2, 4, 4)
    recon = torch.randn(1, 2, 4, 4, requires_grad=True)
    mu = torch.zeros(1, 1)
    logvar = torch.zeros(1, 1)

    s.loss_computer = rec
    s.device = torch.device("cpu")
    s._loss_dict_reuse = {}
    s.env = types.SimpleNamespace(
        generator=lambda _x: (recon, mu, logvar), losses={}, step=0
    )
    s.config = types.SimpleNamespace(model=types.SimpleNamespace(model_type="vae"))
    s.precision_manager = types.SimpleNamespace(
        prepare_latent_for_kl_computation=lambda m, lv: (m, lv)
    )
    s._get_loss_weight = lambda *a, **k: 0.0
    s._compute_training_metrics = lambda *a, **k: {}

    s._compute_losses_impl(x, recon, epoch=0, iteration=_ITER)

    assert rec.calls, "loss_computer.compute was never called"
    assert rec.calls[0].get("iteration") == _ITER, (
        "VAETrainingStrategy dropped 'iteration' — KL annealing stays at "
        "beta_start (default 0) forever (unregularised autoencoder)."
    )


def test_disentangled_vae_threads_iteration_to_loss_computer():
    s = _new(DisentangledVAETrainingStrategy)
    rec = _RecordingComputer()
    img = torch.randn(1, 2, 8, 8)
    recon = torch.randn(1, 2, 8, 8, requires_grad=True)
    mu = torch.zeros(1, 4)
    logvar = torch.zeros(1, 4)
    z = torch.zeros(1, 4, 2, 2)

    class _Gen:
        def __call__(self, _img, _phys):
            return recon, mu, logvar

        def encode_content(self, _img):
            return z

        def decode(self, _z, _phys):
            return recon

    s.loss_computer = rec
    s.device = torch.device("cpu")
    s._loss_dict_reuse = {}
    s.env = types.SimpleNamespace(generator=_Gen(), losses={}, step=0)
    s.config = types.SimpleNamespace(
        training=types.SimpleNamespace(kl_anneal_end=0.0001)
    )
    s._get_loss_weight = lambda *a, **k: 0.0
    s._anatomy_criterion = lambda a, b: torch.zeros((), requires_grad=True)

    s._compute_losses_impl(img, img, epoch=0, iteration=_ITER)

    assert rec.calls, "loss_computer.compute was never called"
    assert all(c.get("iteration") == _ITER for c in rec.calls), (
        "DisentangledVAETrainingStrategy dropped 'iteration' on one or more "
        "of its three compute() calls — spatial-loss warm-up never opens."
    )
