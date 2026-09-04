"""Regression tests for ``RegularizationBlock`` activation dispatch.

Pitfall #9/#10: ``RegularizationBlock`` previously appended a nonlinearity only
for ``activation in {'relu', 'leaky_relu'}`` with no else branch, so any other
value (typo 'gelu'/'silu') silently produced a purely-linear Conv2d+BatchNorm2d
stack with ZERO nonlinearity — a catastrophic silent degradation that still
passes smoke. Post-fix: an unknown activation RAISES, and the two valid values
still insert exactly one nonlinearity per layer (byte-identical numerics).
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from spectramr.models.generators.unrolled_reconstruction_generator import (
    RegularizationBlock,
    UnrolledReconstructionGenerator,
)


def test_regularization_block_rejects_unknown_activation() -> None:
    with pytest.raises(ValueError, match="Unknown activation"):
        RegularizationBlock(activation="gelu")


def test_unrolled_generator_rejects_unknown_activation() -> None:
    # The bad activation propagates through to RegularizationBlock construction.
    with pytest.raises(ValueError, match="Unknown activation"):
        UnrolledReconstructionGenerator(activation="silu")


@pytest.mark.parametrize(
    "activation,expected",
    [("relu", nn.ReLU), ("leaky_relu", nn.LeakyReLU)],
)
def test_valid_activation_inserts_nonlinearity(activation, expected) -> None:
    block = RegularizationBlock(
        in_channels=1, out_channels=1, features=8, num_layers=3, activation=activation
    )
    modules = list(block.model)
    # Regression guard: a valid activation must yield a non-empty nonlinearity set
    # (the bug produced a purely-linear Conv+BN stack).
    acts = [m for m in modules if isinstance(m, (nn.ReLU, nn.LeakyReLU))]
    assert acts, "valid activation must insert a nonlinearity"
    assert all(isinstance(m, expected) for m in acts)


class TestVariationalNetworkOutputActivation:
    """Pitfall #9 for ``VariationalNetworkGenerator.output_activation``: the
    documented None/'none' -> Identity path stays, but an unrecognized STRING
    must raise instead of silently becoming Identity (a typo'd 'tanh' would
    otherwise ship an unbounded output while the YAML advertises a bounded
    one)."""

    @staticmethod
    def _build(output_activation):
        from spectramr.models.generators.unrolled_reconstruction_generator import (
            VariationalNetworkGenerator,
        )

        return VariationalNetworkGenerator(
            in_channels=2,
            out_channels=2,
            img_size=(16, 16),
            num_stages=1,
            features=8,
            output_activation=output_activation,
        )

    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, nn.Identity),
            ("none", nn.Identity),
            ("NONE", nn.Identity),
            ("tanh", nn.Tanh),
            ("Tanh", nn.Tanh),
            ("sigmoid", nn.Sigmoid),
            ("relu", nn.ReLU),
        ],
    )
    def test_valid_output_activation_resolves(self, value, expected):
        gen = self._build(value)
        assert isinstance(gen.output_activation, expected)

    @pytest.mark.parametrize("value", ["swish", "identity", "leaky_relu"])
    def test_unknown_output_activation_raises(self, value):
        with pytest.raises(
            ValueError, match="'none', 'tanh', 'sigmoid', 'relu'"
        ):
            self._build(value)
