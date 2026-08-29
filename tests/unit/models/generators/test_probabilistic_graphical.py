"""Unit tests for :mod:`mriforge.models.generators.probabilistic_graphical`.

This module had no sibling test at all, which is part of why its registration
went five capability fields short for so long: nothing asserted the contract, so
nothing noticed its absence. The audit did not fail on it either -- it
*fail-softs*, reporting "model_type='probabilistic_graphical' is unannotated;
audit cross-check skipped (legacy escape hatch)" with a green tick, which reads
as coverage (#1106).

The tests below pair each declared field with the behaviour that makes it true,
because a declaration is only worth having if it is checkable: in this repo a
WRONG declaration is a hard audit error while an absent one is merely advisory,
so an unverified declaration is a downgrade rather than an improvement.

device='cpu', tiny tensors, fixed seed.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from mriforge.models.generators.probabilistic_graphical import (
    ProbabilisticGraphicalGenerator,
)
from mriforge.models.init_registry import populate_model_registry
from mriforge.models.registry import MODEL_REGISTRY, get_model_capabilities

populate_model_registry()


def _model(in_channels: int = 2, out_channels: int = 2, **kwargs):
    torch.manual_seed(0)
    return ProbabilisticGraphicalGenerator(
        in_channels=in_channels, out_channels=out_channels, **kwargs
    )


def _code_of(obj) -> str:
    """Source of ``obj`` with comment lines stripped.

    Needed because ``inspect.getsource`` on a decorated class includes the
    decorator -- and the decorator's own comment explains the domain reasoning
    in prose that mentions IFFT, so a naive substring scan would match the
    explanation rather than the code and always fail.
    """
    return "\n".join(
        line
        for line in inspect.getsource(obj).splitlines()
        if not line.lstrip().startswith("#")
    )


class TestRegistration:
    def test_registered_under_its_name(self) -> None:
        assert "probabilistic_graphical" in MODEL_REGISTRY
        assert MODEL_REGISTRY["probabilistic_graphical"]["class"] is (
            ProbabilisticGraphicalGenerator
        )

    def test_the_name_property_agrees_with_the_registry_key(self) -> None:
        """Two spellings of the same identity; a drift between them is silent."""
        assert _model().name == "probabilistic_graphical"


class TestForward:
    def test_forward_preserves_spatial_shape(self) -> None:
        model = _model()
        model.eval()
        with torch.no_grad():
            y = model(torch.randn(1, 2, 32, 32))
        assert y.shape == (1, 2, 32, 32)

    def test_gradients_reach_the_message_passing_layers(self) -> None:
        """Belief propagation is the point of the model; a residual+gate stack
        can look trained while the messages contribute nothing."""
        model = _model()
        model.train()
        model(torch.randn(1, 2, 16, 16)).mean().backward()
        bp_grads = [
            p.grad
            for n, p in model.named_parameters()
            if "bp_layers" in n and p.grad is not None
        ]
        assert bp_grads, "no gradient reached any bp layer"
        assert any(g.abs().sum() > 0 for g in bp_grads)

    def test_output_is_finite(self) -> None:
        model = _model()
        model.eval()
        with torch.no_grad():
            y = model(torch.randn(2, 2, 16, 16))
        assert torch.isfinite(y).all()


class TestCapabilityContract:
    """The declared contract, each field against the behaviour it claims."""

    def test_the_contract_is_declared_at_all(self) -> None:
        caps = get_model_capabilities("probabilistic_graphical")
        assert caps is not None
        assert caps.spatial_dims == (2,)
        assert caps.input_domain == "image"
        assert caps.output_domain == "image"
        assert caps.accepts_complex is False
        assert caps.requires_paired_data is True

    def test_rank_2_is_a_contract_not_a_preference(self) -> None:
        """``spatial_dims=(2,)`` is honest only if rank 3 genuinely fails."""
        with pytest.raises(RuntimeError, match="conv2d"):
            _model()(torch.randn(1, 2, 8, 16, 16))

    def test_accepts_complex_false_is_honest(self) -> None:
        """Declared False, so complex input must actually fail.

        A wrong False is the expensive direction: it is what made the audit
        reject valid pre_model iFFT chains for ``graph_unet`` until 2026-05-12.
        Pin that the refusal comes from the model, not from the declaration.
        """
        with pytest.raises(RuntimeError):
            _model(in_channels=1)(torch.randn(1, 1, 16, 16, dtype=torch.complex64))

    def test_the_image_domain_claim_rests_on_there_being_no_transform(self) -> None:
        """The module docstring hedges -- "k-space / image domain" -- so the
        declaration rests on the forward being domain-agnostic (no FFT on any
        path), which makes what the ARM feeds decisive. Pin that absence: adding
        an internal transform later would silently falsify the declaration."""
        assert "fft" not in _code_of(ProbabilisticGraphicalGenerator).lower()
