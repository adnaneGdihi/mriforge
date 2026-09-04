"""VAETrainingStrategy autoencoder-target guard (2026-07 ldm_two_stage triage).

The stage-1 VAE reconstruction target MUST equal the input (an autoencoder).
The former ``target_batch = input_batch`` fallback silently rescued a
shape-mismatched target — which is exactly how a ``hf_to_ulf`` *translation*
config masqueraded as an HF autoencoder and corrupted the frozen stage-2 latent.
The guard now RAISES on a missing / empty / shape-mismatched target. These tests
pin that a matching-shape target proceeds and a mismatched / None target raises.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from spectramr.infrastructure.training.strategies.vae import VAETrainingStrategy


class _SentinelError(Exception):
    """Raised by the mocked loss computer to prove control reached past the guard."""


def _mock_self(recon: torch.Tensor):
    """A MagicMock strategy whose generator returns (recon, mu, logvar) and whose
    loss computer raises _SentinelError — so a passing guard surfaces as _SentinelError and
    a failing guard surfaces as ValueError, cleanly distinguishing the two."""
    mock = MagicMock()
    mu = torch.zeros(recon.shape[0], 4, 1, 1)
    logvar = torch.zeros(recon.shape[0], 4, 1, 1)
    mock.env.generator = MagicMock(return_value=(recon, mu, logvar))
    mock.precision_manager.prepare_latent_for_kl_computation = MagicMock(
        return_value=(mu, logvar)
    )
    mock.env.losses = {}
    mock.loss_computer.compute = MagicMock(side_effect=_SentinelError)
    return mock


def test_compute_losses_matching_target_passes_guard() -> None:
    inp = torch.randn(1, 1, 32, 32)
    recon = torch.randn(1, 1, 32, 32)  # same shape as target below
    mock = _mock_self(recon)
    # target matches recon → guard must NOT raise ValueError; control reaches the
    # loss computer (our sentinel), proving the autoencoder objective was accepted.
    with pytest.raises(_SentinelError):
        VAETrainingStrategy._compute_losses_impl(
            mock, inp, torch.randn(1, 1, 32, 32), epoch=0, iteration=0
        )


def test_compute_losses_mismatched_target_raises() -> None:
    inp = torch.randn(1, 1, 32, 32)
    recon = torch.randn(1, 1, 32, 32)
    mock = _mock_self(recon)
    with pytest.raises(ValueError, match="target ≡ input"):
        VAETrainingStrategy._compute_losses_impl(
            mock, inp, torch.randn(1, 1, 16, 16), epoch=0, iteration=0
        )


def test_compute_losses_none_target_raises() -> None:
    inp = torch.randn(1, 1, 32, 32)
    recon = torch.randn(1, 1, 32, 32)
    mock = _mock_self(recon)
    with pytest.raises(ValueError, match="missing or shape-mismatched"):
        VAETrainingStrategy._compute_losses_impl(mock, inp, None, epoch=0, iteration=0)


class TestBareTensorPosteriorRecovery:
    """A VAE that returns a BARE TENSOR but caches its posterior must still get a KL.

    ``slat_vae_slab_to_volume`` returns a plain tensor from ``forward`` (its
    structured return is behind an opt-in ``return_structured=True`` that this path
    never passes) and exposes the posterior via ``last_aux()``. The strategy used to
    fall straight through to dummy ZERO mu/logvar, making KL identically 0: the beta
    schedule was routed correctly and then multiplied a constant zero, so the "VAE"
    trained as a plain autoencoder with an unregularised latent (pitfalls #9/#16).
    """

    @staticmethod
    def _loss_output():
        """A real-enough LossOutput: the strategy reads .total.device downstream."""
        return SimpleNamespace(
            total=torch.tensor(1.0, requires_grad=True), components={}, metrics={}
        )

    @classmethod
    def _mock_with_last_aux(cls, recon, mu, logvar):
        mock = MagicMock()
        mock.env.generator = MagicMock(return_value=recon)  # BARE tensor
        mock.env.generator.last_aux = MagicMock(
            return_value={"mu": mu, "logvar": logvar, "s_coarse": None}
        )
        # Identity precision manager so we can assert on the values that arrive.
        mock.precision_manager.prepare_latent_for_kl_computation = MagicMock(
            side_effect=lambda m, lv: (m, lv)
        )
        mock.env.losses = {}
        mock.loss_computer.compute = MagicMock(return_value=cls._loss_output())
        return mock

    def test_posterior_is_pulled_from_last_aux(self) -> None:
        inp = torch.randn(1, 1, 32, 32)
        recon = torch.randn(1, 1, 32, 32)
        mu = torch.full((1, 8, 4, 4), 0.5)
        logvar = torch.full((1, 8, 4, 4), -0.25)

        mock = self._mock_with_last_aux(recon, mu, logvar)
        VAETrainingStrategy._compute_losses_impl(
            mock, inp, recon.clone(), epoch=0, iteration=0
        )

        posterior = mock.loss_computer.compute.call_args.kwargs["posterior"]
        got_mu, got_logvar = posterior

        # The REAL posterior arrived — not the 1-element zero dummy.
        assert torch.equal(got_mu, mu)
        assert torch.equal(got_logvar, logvar)
        # The pre-fix bug in one assertion: a zero posterior yields KL == 0 exactly.
        assert got_mu.abs().sum() > 0

    def test_plain_non_vae_model_still_gets_a_zero_posterior(self) -> None:
        """A model with no ``last_aux`` (e.g. a bare UNet) keeps the old behaviour:
        reconstruction-only with a zero KL contribution, not a crash."""
        inp = torch.randn(1, 1, 32, 32)
        recon = torch.randn(1, 1, 32, 32)

        mock = MagicMock()
        mock.env.generator = MagicMock(return_value=recon)
        # A MagicMock auto-creates attributes, so last_aux must be explicitly absent.
        del mock.env.generator.last_aux
        mock.precision_manager.prepare_latent_for_kl_computation = MagicMock(
            side_effect=lambda m, lv: (m, lv)
        )
        mock.env.losses = {}
        mock.loss_computer.compute = MagicMock(return_value=self._loss_output())

        VAETrainingStrategy._compute_losses_impl(
            mock, inp, recon.clone(), epoch=0, iteration=0
        )

        got_mu, got_logvar = mock.loss_computer.compute.call_args.kwargs["posterior"]
        assert torch.count_nonzero(got_mu) == 0
        assert torch.count_nonzero(got_logvar) == 0
