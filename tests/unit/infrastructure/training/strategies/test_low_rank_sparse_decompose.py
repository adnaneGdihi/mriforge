"""Regression: the low-rank + sparse (RPCA) decomposition must actually run.

2026-05-31 audit — :class:`LowRankSparseStrategy._compute_losses_impl` keyed
the entire RPCA splitting on ``batch.get("prediction")``, a key the parent
:class:`ReconstructionTrainingStrategy` never writes. The lookup therefore
always returned ``None`` and the strategy silently degraded to a plain
reconstruction loss — the L+S decomposition (the whole paradigm) never ran.

The fix re-runs the generator (``self.env.generator``) on the resolved input
to obtain the temporal stack X, then performs ``X = L + S`` and adds the
nuclear / sparse / consistency terms onto the base recon loss.

These tests construct the strategy via ``object.__new__`` (bypassing the heavy
``ReconstructionTrainingStrategy.__init__``) and set only the attributes the
tested method touches. ``super()._compute_losses_impl`` is patched to return a
controlled base dict so the test isolates the decomposition behavior.
"""

from __future__ import annotations

import types
from unittest.mock import patch

import pytest
import torch

from mriforge.infrastructure.training.strategies.low_rank_sparse_strategy import (
    LowRankSparseStrategy,
)
from mriforge.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)


class _StubGenerator(torch.nn.Module):
    """Maps a (B, T, H, W) input to a (B, T, H, W) temporal stack via one
    learnable scale, so we can assert gradient flow from the L+S terms."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Non-trivial, full-rank-ish transform keeps the SVT/soft-threshold
        # paths exercised rather than collapsing to zero.
        return self.scale * x + 0.01


def _make_strategy(generator):
    """A strategy with only the attrs ``_compute_losses_impl`` touches."""
    s = object.__new__(LowRankSparseStrategy)
    s.env = types.SimpleNamespace(generator=generator)
    s.device = torch.device("cpu")
    # Class-level loss weights are inherited; make them visibly non-zero.
    s.tau_low_rank = 0.05
    s.tau_sparse = 0.05
    s.lambda_nuclear = 1e-3
    s.lambda_sparse = 1e-2
    s.lambda_consistency = 1e-3
    return s


def _base_dict():
    # Mimic the parent's return: a recon loss bound to its own graph.
    total = (torch.zeros(1, requires_grad=True) + 1.0).squeeze(0)
    return {"loss_total": total, "g_total_loss": total}


class TestLowRankSparseDecompose:
    def test_decomposition_runs_and_terms_in_loss_dict(self):
        gen = _StubGenerator()
        s = _make_strategy(gen)
        frames = torch.randn(2, 4, 8, 8)  # (B, T, H, W) temporal stack

        with patch.object(
            ReconstructionTrainingStrategy,
            "_compute_losses_impl",
            return_value=_base_dict(),
        ):
            out = s._compute_losses_impl(input_batch={"input": frames}, epoch=0)

        # The fix: the RPCA terms are present (decomposition actually executed).
        assert "loss_nuclear" in out
        assert "loss_sparse" in out
        assert "loss_ls_consistency" in out
        # Nuclear term is the reliable witness that the decomposition ran (the
        # sparse term can legitimately be 0 when the residual is fully below the
        # soft-threshold tau, so it is not a robust "ran" signal).
        assert out["loss_nuclear"].item() > 0.0
        # The total absorbed the extra terms.
        assert out["loss_total"].item() > 1.0

    def test_nuclear_sparse_terms_flow_gradient_to_generator(self):
        gen = _StubGenerator()
        s = _make_strategy(gen)
        frames = torch.randn(2, 4, 8, 8)

        with patch.object(
            ReconstructionTrainingStrategy,
            "_compute_losses_impl",
            return_value=_base_dict(),
        ):
            out = s._compute_losses_impl(input_batch={"input": frames}, epoch=0)

        # The decomposition was driven by gen(input), so backprop through the
        # L+S terms must reach the generator's learnable scale.
        rpca = out["loss_nuclear"] + out["loss_sparse"] + out["loss_ls_consistency"]
        rpca.backward()
        assert gen.scale.grad is not None
        assert torch.any(gen.scale.grad != 0)

    def test_missing_generator_raises(self):
        # Paradigm contract: no generator -> fail loudly, never silent no-op.
        s = _make_strategy(generator=None)
        with patch.object(
            ReconstructionTrainingStrategy,
            "_compute_losses_impl",
            return_value=_base_dict(),
        ):
            try:
                s._compute_losses_impl(
                    input_batch={"input": torch.randn(2, 4, 8, 8)}, epoch=0
                )
            except RuntimeError as exc:
                assert "generator" in str(exc)
            else:
                raise AssertionError("expected RuntimeError when generator is absent")

    def test_tensor_input_path_decomposes(self):
        # When the batch is a plain tensor (not a dict), the generator runs on
        # input_batch directly.
        gen = _StubGenerator()
        s = _make_strategy(gen)
        frames = torch.randn(2, 4, 8, 8)

        with patch.object(
            ReconstructionTrainingStrategy,
            "_compute_losses_impl",
            return_value=_base_dict(),
        ):
            out = s._compute_losses_impl(input_batch=frames, epoch=0)

        assert "loss_nuclear" in out
        assert "loss_sparse" in out


def _config_with_block(block):
    """A frozen-ish config stub exposing ``config.training.low_rank_sparse``."""
    return types.SimpleNamespace(training=types.SimpleNamespace(low_rank_sparse=block))


def _init_strategy_with_config(config):
    """Run ``LowRankSparseStrategy.__init__`` with the heavy parent ``__init__``
    stubbed to a no-op that only sets ``self.config``.

    This isolates the knob read/validate/stamp logic added to the strategy's
    ``__init__`` without constructing a full TrainingEnvironment.
    """

    def fake_parent_init(self, env=None, device=None, **kwargs):
        self.config = config
        self.logging_service = None

    s = object.__new__(LowRankSparseStrategy)
    with patch.object(
        ReconstructionTrainingStrategy, "__init__", fake_parent_init
    ):
        LowRankSparseStrategy.__init__(s, env=None, device=None)
    return s


class TestLowRankSparseKnobWiring:
    """Regression: the RPCA knobs must be read from the frozen config, not
    silently fall back to the hard-coded class-attribute defaults.

    Pre-fix the strategy had no ``__init__`` reading config, so
    ``strategy.lambda_nuclear`` was always the class literal ``1e-3``
    regardless of YAML — these tests fail on that code and pass now.
    """

    def test_yaml_value_flows_through_to_instance(self):
        block = types.SimpleNamespace(
            tau_low_rank=0.05,
            tau_sparse=0.05,
            lambda_nuclear=0.4242,  # distinctive, != default 1e-3
            lambda_sparse=1e-2,
            lambda_consistency=1e-3,
        )
        s = _init_strategy_with_config(_config_with_block(block))
        assert s.lambda_nuclear == 0.4242
        # Distinct from the class-attribute default — proves the YAML value
        # flowed through rather than the literal being used.
        assert s.lambda_nuclear != LowRankSparseStrategy.lambda_nuclear
        # Stamped into provenance for audit.
        assert s._resolved_rpca_knobs["lambda_nuclear"] == 0.4242

    def test_missing_block_uses_documented_defaults(self):
        s = _init_strategy_with_config(_config_with_block(None))
        assert s.tau_low_rank == 0.05
        assert s.lambda_nuclear == 1e-3
        assert s._resolved_rpca_knobs["lambda_sparse"] == 1e-2

    def test_negative_weight_raises(self):
        block = types.SimpleNamespace(
            tau_low_rank=0.05,
            tau_sparse=0.05,
            lambda_nuclear=-1.0,  # illegal
            lambda_sparse=1e-2,
            lambda_consistency=1e-3,
        )
        with pytest.raises(ValueError):
            _init_strategy_with_config(_config_with_block(block))

    def test_all_knobs_resolved_and_stamped(self):
        block = types.SimpleNamespace(
            tau_low_rank=0.11,
            tau_sparse=0.22,
            lambda_nuclear=0.33,
            lambda_sparse=0.44,
            lambda_consistency=0.55,
        )
        s = _init_strategy_with_config(_config_with_block(block))
        assert s._resolved_rpca_knobs == {
            "tau_low_rank": 0.11,
            "tau_sparse": 0.22,
            "lambda_nuclear": 0.33,
            "lambda_sparse": 0.44,
            "lambda_consistency": 0.55,
        }
        # The decomposition path reads these instance attrs, so they must match.
        assert s.tau_low_rank == 0.11
        assert s.lambda_consistency == 0.55
