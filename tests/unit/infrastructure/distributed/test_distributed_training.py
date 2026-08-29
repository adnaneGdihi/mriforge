import unittest.mock as mock

import pytest
import torch
import torch.nn as nn

from mriforge.infrastructure.distributed.distributed_training import (
    DistributedLossAudit,
    DistributedMetricsCollector,
    DistributedTrainingContext,
    RankUtility,
    synchronize_across_ranks,
    wrap_model_distributed,
)


@pytest.fixture
def mock_dist():
    """Mock torch.distributed"""
    with mock.patch("mriforge.infrastructure.distributed.distributed_training.dist") as m:
        m.is_available.return_value = True
        m.is_initialized.return_value = True
        yield m


def test_rank_utility_single_process():
    """Test RankUtility without distributed environment."""
    with mock.patch("mriforge.infrastructure.distributed.distributed_training.dist") as m:
        m.is_available.return_value = False
        m.is_initialized.return_value = False

        assert RankUtility.get_rank() == 0
        assert RankUtility.get_world_size() == 1
        assert RankUtility.is_main_rank()

        # broadcast should return original tensor
        t = torch.tensor([1.0])
        assert RankUtility.broadcast_tensor(t) is t

        # all_gather should return list with original tensor
        assert RankUtility.all_gather_tensor(t) == [t]


def test_rank_utility_distributed(mock_dist):
    """Test RankUtility with mocked distributed environment."""
    mock_dist.get_rank.return_value = 1
    mock_dist.get_world_size.return_value = 4

    assert RankUtility.get_rank() == 1
    assert RankUtility.get_world_size() == 4
    assert not RankUtility.is_main_rank()

    # all_gather on non-main rank returns empty list
    t = torch.tensor([1.0])
    assert RankUtility.all_gather_tensor(t) == []


def test_rank_utility_all_gather_main(mock_dist):
    mock_dist.get_rank.return_value = 0
    mock_dist.get_world_size.return_value = 2

    t = torch.tensor([1.0])
    gathered = RankUtility.all_gather_tensor(t)
    assert len(gathered) == 2
    mock_dist.all_gather.assert_called_once()


def test_synchronize_across_ranks_single_process():
    """Test synchronization fallback when dist is not initialized."""
    val = synchronize_across_ranks(5.0)
    assert val == 5.0


def test_synchronize_across_ranks_dist(mock_dist):
    """Test synchronization when dist is available."""

    # We mock all_reduce to simulate modifying the tensor
    def mock_all_reduce(tensor, op):
        tensor.data = torch.tensor([10.0])

    mock_dist.all_reduce = mock_all_reduce

    val = synchronize_across_ranks(5.0, op="mean")
    assert val == 10.0

    with pytest.raises(ValueError, match="Unknown reduction operation"):
        synchronize_across_ranks(5.0, op="unknown")


def test_distributed_metrics_collector():
    """Test local metrics collection and single-process synchronization."""
    collector = DistributedMetricsCollector()
    collector.update_metrics({"loss": 1.0, "acc": 0.5})
    collector.update_metrics({"loss": 3.0, "acc": 0.7})

    assert collector.local_metrics["loss"] == [1.0, 3.0]

    with mock.patch(
        "mriforge.infrastructure.distributed.distributed_training.dist.is_initialized",
        return_value=False,
    ):
        snapshot = collector.synchronize_metrics()

    assert snapshot.metrics["loss"] == 2.0
    assert snapshot.metrics["acc"] == 0.6
    assert snapshot.synchronized

    collector.reset_metrics()
    assert collector.local_metrics == {}


def test_distributed_metrics_collector_dist(mock_dist):
    """Test metrics synchronization with mocked dist."""
    collector = DistributedMetricsCollector()
    collector.update_metrics({"loss": 1.0})

    def mock_all_reduce(tensor, op):
        tensor.data = torch.tensor([2.0])

    mock_dist.all_reduce = mock_all_reduce

    snapshot = collector.synchronize_metrics()
    assert snapshot.metrics["loss"] == 2.0
    assert snapshot.synchronized


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
        self.frozen_fc = nn.Linear(10, 1)
        for param in self.frozen_fc.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.fc(x) + self.frozen_fc(x)


class DummyModelLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.mse = nn.MSELoss()

    def forward(self, x, target):
        return self.mse(self.model(x), target)


def test_distributed_loss_audit_single_process():
    """Gradient Flow Audit and general Loss Audit test on CPU."""
    model = DummyModel()
    # The DistributedLossAudit expects a loss_fn that has .parameters() if it computes grad_norm
    loss_fn = DummyModelLoss(model)

    audit = DistributedLossAudit(loss_fn)

    batch = {"input": torch.randn(2, 10)}
    target = torch.randn(2, 1)

    # Note that DistributedLossAudit computes loss = loss_fn(model_output, target)
    # where it passes model_output directly. So we can bypass the model in audit and just give it input.
    # We will pass x as "model_output"
    result = audit.audit_loss_computation(batch, batch["input"], target, backward=True)

    assert result.loss_valid
    assert result.backward_successful
    assert result.gradient_norm > 0

    # GRADS CHECK
    assert model.fc.weight.grad is not None
    assert not torch.isnan(model.fc.weight.grad).any()

    # FROZEN LAYER CHECK
    assert model.frozen_fc.weight.grad is None


def test_distributed_loss_audit_invalid_loss():
    """Test loss audit handling non-finite loss."""
    model = DummyModel()

    # Force inf loss
    def bad_loss(out, target):
        return torch.tensor(float("inf"), requires_grad=True)

    audit = DistributedLossAudit(bad_loss)
    result = audit.audit_loss_computation(
        {}, torch.randn(1), torch.randn(1), backward=True
    )

    assert not result.loss_valid
    assert "not finite" in result.error_message


def test_distributed_training_context():
    """Test DistributedTrainingContext context manager."""
    with mock.patch("mriforge.infrastructure.distributed.distributed_training.dist") as m:
        m.is_available.return_value = True
        m.is_initialized.return_value = False

        with DistributedTrainingContext(backend="gloo") as ctx:
            m.init_process_group.assert_called_once_with(backend="gloo")
            assert ctx.is_initialized

        m.destroy_process_group.assert_called_once()


def test_wrap_model_distributed_no_dist():
    """Test wrap_model_distributed when dist is not active."""
    model = DummyModel()
    wrapped = wrap_model_distributed(model)
    assert wrapped is model


@mock.patch("mriforge.infrastructure.distributed.distributed_training.DDP")
def test_wrap_model_distributed_with_dist(mock_ddp, mock_dist):
    """Test wrap_model_distributed when dist is active."""
    model = DummyModel()
    wrap_model_distributed(model)
    mock_ddp.assert_called_once()


def test_get_local_rank_reads_local_rank_env(monkeypatch):
    """LOCAL_RANK env (torchrun/launcher) is the authoritative local index."""
    monkeypatch.setenv("LOCAL_RANK", "2")
    assert RankUtility.get_local_rank() == 2


def test_get_local_rank_falls_back_to_modulo_device_count(monkeypatch, mock_dist):
    """Without LOCAL_RANK, local rank = global_rank % device_count.

    Regression: DDP used the GLOBAL rank as the CUDA device index, which on a
    multi-node run exceeds the per-node GPU count (invalid index).
    """
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    mock_dist.get_rank.return_value = 5  # global rank on node 1 (4 GPUs/node)
    with mock.patch(
        "mriforge.infrastructure.distributed.distributed_training.torch.cuda"
    ) as mock_cuda:
        mock_cuda.is_available.return_value = True
        mock_cuda.device_count.return_value = 4
        assert RankUtility.get_local_rank() == 1  # 5 % 4, NOT 5


@mock.patch("mriforge.infrastructure.distributed.distributed_training.DDP")
def test_wrap_model_uses_local_rank_for_device_ids(mock_ddp, mock_dist, monkeypatch):
    """DDP must receive the node-local device index, not the global rank."""
    monkeypatch.setenv("LOCAL_RANK", "3")
    wrap_model_distributed(DummyModel())
    _, kwargs = mock_ddp.call_args
    assert kwargs["device_ids"] == [3]


def test_broadcast_object_noop_single_process():
    """Outside a process group, broadcast_object returns the object unchanged
    (so the single-process training path is unaffected)."""
    with mock.patch(
        "mriforge.infrastructure.distributed.distributed_training.dist"
    ) as m:
        m.is_available.return_value = False
        m.is_initialized.return_value = False
        obj = {"overrides": {"l1": 0.5}, "checkpoint": True}
        assert RankUtility.broadcast_object(obj) is obj


def test_broadcast_object_uses_collective_when_distributed(mock_dist):
    """Under a live process group it goes through dist.broadcast_object_list so
    every rank ends up with rank 0's decision."""
    obj = {"overrides": {"perceptual": 0.1}, "checkpoint": False}
    out = RankUtility.broadcast_object(obj)
    mock_dist.broadcast_object_list.assert_called_once()
    # The mock does not mutate the holder, so the src-rank value is returned.
    assert out == obj
