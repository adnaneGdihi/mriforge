import ast
import inspect
import textwrap

import pytest
import torch
import torch.nn as nn

from spectramr.infrastructure.training.distributed.dogs import DistributedVolumeTrainer


class SimpleConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Input channel 1, output channel 1.
        self.conv = nn.Conv3d(1, 1, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


@pytest.fixture
def trainer():
    model = SimpleConvModel()
    return DistributedVolumeTrainer(model, num_nodes=2)


def test_dogs_block_split(trainer):
    # Volume: [C, D, H, W] -> [1, 10, 16, 16]
    volume = torch.randn(1, 10, 16, 16)
    chunks = trainer.block_split(volume)

    # Should get 2 chunks of size [1, 5, 16, 16]
    assert len(chunks) == 2
    assert chunks[0].shape == (1, 5, 16, 16)
    assert chunks[1].shape == (1, 5, 16, 16)


def test_dogs_epoch_execution(trainer):
    # Setup
    volume = torch.randn(1, 10, 8, 8)
    criterion = nn.MSELoss()

    # Capture initial weights
    initial_weight = trainer.global_model.conv.weight.detach().clone()

    # Run epoch
    avg_loss = trainer.run_epoch(volume, criterion)

    # Assertions
    assert avg_loss >= 0

    # Weights should have changed
    final_weight = trainer.global_model.conv.weight
    assert not torch.allclose(initial_weight, final_weight)


def test_dogs_aggregation_logic(trainer):
    # Manually modify local models to test averaging
    trainer.broadcast_global_weights()

    # Model 0: weights = 1
    with torch.no_grad():
        trainer.local_models[0].conv.weight.fill_(1.0)

    # Model 1: weights = 3
    with torch.no_grad():
        trainer.local_models[1].conv.weight.fill_(3.0)

    trainer.aggregate_weights()

    # Global should be (1+3)/2 = 2
    assert torch.allclose(trainer.global_model.conv.weight, torch.tensor(2.0))


def test_dogs_train_local_step_returns_python_float(trainer):
    """TE-002: per-step loss must be deferred; return value is a Python float."""
    volume = torch.randn(1, 10, 8, 8)
    criterion = nn.MSELoss()
    chunks = trainer.block_split(volume)
    trainer.broadcast_global_weights()

    avg_loss = trainer.train_local_step(chunks, criterion)

    assert isinstance(avg_loss, float)
    assert avg_loss >= 0


def _item_calls_in_loop(func) -> int:
    """Count ``X.item()`` attribute-calls that sit inside a for-loop body."""
    src = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(src)
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "item"
            ):
                count += 1
    return count


def test_dogs_train_local_step_no_item_inside_loop():
    """TE-002: NN#9 — no per-step ``.item()`` D2H sync inside the loop.

    The pre-fix code called ``loss.item()`` once per local model (one D2H
    sync per node). The fix defers via ``loss.detach()`` and resolves with a
    single ``torch.stack(...).mean().item()`` AFTER the loop. A global mock
    on ``torch.Tensor.item`` would also catch PyTorch-internal calls, so we
    assert the source contract directly.
    """
    assert _item_calls_in_loop(DistributedVolumeTrainer.train_local_step) == 0, (
        "train_local_step calls .item() inside the per-node loop — NN#9 "
        "requires deferring to a single post-loop host sync."
    )


def test_dogs_train_local_step_post_loop_stack_mean(trainer):
    """TE-002: the deferred losses are reduced with a single stack().mean()."""
    src = inspect.getsource(DistributedVolumeTrainer.train_local_step)
    assert "loss.detach()" in src, "per-step loss must be deferred via detach()"
    assert "torch.stack(" in src, "post-loop reduction must use torch.stack"
