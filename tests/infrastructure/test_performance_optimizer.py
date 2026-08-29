from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from mriforge.infrastructure.performance_optimizer import (
    MemoryEfficientTrainer,
    PerformanceOptimizer,
    get_performance_optimizer,
    performance_monitor,
)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)
        self.seq = nn.Sequential(nn.Conv2d(1, 1, 3, padding=1), nn.ReLU())

    def forward(self, x):
        return self.seq(self.conv(x))


@patch("mriforge.infrastructure.performance_optimizer.resolve_service")
def test_performance_optimizer_init(mock_resolve):
    config = {
        "amp": {"enabled": True},
        "gradient_checkpointing": {"enabled": True},
        "memory_optimization": {"enabled": True},
    }
    opt = PerformanceOptimizer(config)
    assert opt._amp_enabled() is True
    assert opt._gradient_checkpointing_enabled() is True
    assert opt._memory_optimization_enabled() is True
    assert opt.scaler is not None


@patch("mriforge.infrastructure.performance_optimizer.resolve_service")
def test_auto_tune_batch_size_returns_max_without_cuda(mock_resolve):
    """Without CUDA, auto_tune short-circuits to max_batch_size. Regression guard
    for the single-guard cleanup (the duplicate ``if not cuda: return`` removed)."""
    if torch.cuda.is_available():
        pytest.skip("CUDA present; this guards the no-CUDA short-circuit path")
    opt = PerformanceOptimizer({})
    assert opt.auto_tune_batch_size(DummyModel(), max_batch_size=17) == 17


@patch("mriforge.infrastructure.performance_optimizer.resolve_service")
def test_apply_gradient_checkpointing(mock_resolve):
    config = {"gradient_checkpointing": {"enabled": True}}
    opt = PerformanceOptimizer(config)

    model = DummyModel()

    # Check if a custom model applies it correctly if it has apply_checkpointing
    class CustomModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.applied = False

        def apply_checkpointing(self):
            self.applied = True

    cm = CustomModel()
    opt.apply_gradient_checkpointing(cm)
    assert cm.applied is True


@patch("mriforge.infrastructure.performance_optimizer.resolve_service")
def test_optimize_memory_usage(mock_resolve):
    config = {"memory_optimization": {"enabled": True}}
    opt = PerformanceOptimizer(config)

    model = DummyModel()

    # We can't strictly assert channels_last is set perfectly across all layers without inspecting internal flags,
    # but we can call it and make sure it doesn't crash.
    opt.optimize_memory_usage(model)


@patch("mriforge.infrastructure.performance_optimizer.resolve_service")
@patch(
    "mriforge.infrastructure.performance_optimizer.torch.cuda.is_available",
    return_value=False,
)
@patch("mriforge.infrastructure.performance_optimizer.psutil.virtual_memory")
def test_get_memory_stats_cpu(mock_vm, mock_cuda, mock_resolve):
    mock_mem = MagicMock()
    mock_mem.percent = 50.0
    mock_vm.return_value = mock_mem

    opt = PerformanceOptimizer()
    stats = opt.get_memory_stats()

    assert stats.allocated_mb == 0.0
    assert stats.reserved_mb == 0.0
    assert stats.system_memory_percent == 50.0
    assert stats.gpu_memory_percent == 0.0
    assert len(opt.memory_history) == 1


@patch("mriforge.infrastructure.performance_optimizer.resolve_service")
@patch(
    "mriforge.infrastructure.performance_optimizer.torch.cuda.is_available",
    return_value=False,
)
def test_profile_model(mock_cuda, mock_resolve):
    opt = PerformanceOptimizer()
    model = DummyModel()
    input_shape = (2, 1, 8, 8)

    metrics = opt.profile_model(model, input_shape, num_runs=2)
    assert metrics.forward_time > 0
    assert metrics.backward_time > 0
    assert metrics.throughput > 0
    assert len(opt.metrics_history) == 1


@patch("mriforge.infrastructure.performance_optimizer.resolve_service")
@patch(
    "mriforge.infrastructure.performance_optimizer.torch.cuda.is_available",
    return_value=False,
)
def test_auto_tune_batch_size_cpu(mock_cuda, mock_resolve):
    # On CPU, returns max_batch_size immediately
    opt = PerformanceOptimizer()
    model = DummyModel()
    res = opt.auto_tune_batch_size(model, (1, 1, 8, 8), max_batch_size=16)
    assert res == 16


@patch("mriforge.infrastructure.performance_optimizer.resolve_service")
def test_optimize_model_for_inference(mock_resolve):
    opt = PerformanceOptimizer()
    model = DummyModel()
    # It attempts to compile
    model = opt.optimize_model_for_inference(model)
    assert not model.training


@patch("mriforge.infrastructure.performance_optimizer.resolve_service")
def test_memory_efficient_trainer_gradient_flow(mock_resolve):
    # CRITICAL AUDIT: Check gradient flow through the training step
    config = {
        "amp": {"enabled": False}
    }  # Disable AMP to ensure simple backward pass works predictably on CPU
    opt = PerformanceOptimizer(config)

    model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 10))

    # Freeze the first linear layer
    for param in model[0].parameters():
        param.requires_grad = False

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.1
    )

    trainer = MemoryEfficientTrainer(optimizer, opt)

    batch = {"lr": torch.randn(2, 10), "hr": torch.randn(2, 10)}

    loss_fn = nn.MSELoss()

    metrics = trainer.training_step(model, batch, loss_fn)

    assert "loss" in metrics
    assert "lr" in metrics

    # Assert gradient flows properly to the active layer
    assert model[2].weight.grad is not None
    assert not torch.isnan(model[2].weight.grad).any()
    assert not torch.isinf(model[2].weight.grad).any()

    # Assert gradient strictly DOES NOT flow to the detached/frozen layer
    assert model[0].weight.grad is None


@patch("mriforge.infrastructure.performance_optimizer.resolve_service")
def test_autocast_context_uses_device_type_no_deprecation(mock_resolve):
    """INFRA-MISC-001: autocast_context must call torch.amp.autocast with a
    device_type so it does not emit the deprecated no-arg DeprecationWarning
    (promoted to an error for mriforge.*).
    """
    import warnings

    config = {"amp": {"enabled": True}}
    opt = PerformanceOptimizer(config)
    assert opt.scaler is not None  # AMP path active so autocast is exercised

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with opt.autocast_context():
            pass  # must not raise a DeprecationWarning


@patch("mriforge.infrastructure.performance_optimizer.resolve_service")
def test_autocast_context_passes_resolved_device_type(mock_resolve):
    """INFRA-MISC-001: the resolved device.type is forwarded to torch.amp.autocast."""
    config = {"amp": {"enabled": True}}
    opt = PerformanceOptimizer(config)

    with patch(
        "mriforge.infrastructure.performance_optimizer.torch.amp.autocast"
    ) as mock_autocast:
        with opt.autocast_context():
            pass

    mock_autocast.assert_called_once_with(opt.device.type)


@patch("mriforge.infrastructure.performance_optimizer.resolve_service")
def test_memory_efficient_trainer_zero_grad_set_to_none(mock_resolve):
    """INFRA-MISC-002: training_step must call zero_grad(set_to_none=True)."""
    config = {"amp": {"enabled": False}}
    opt = PerformanceOptimizer(config)

    model = nn.Sequential(nn.Linear(10, 10))
    real_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    mock_optimizer = MagicMock(wraps=real_optimizer)
    # MagicMock(wraps=...) does not forward param_groups attribute access; expose it.
    mock_optimizer.param_groups = real_optimizer.param_groups

    trainer = MemoryEfficientTrainer(mock_optimizer, opt)
    batch = {"lr": torch.randn(2, 10), "hr": torch.randn(2, 10)}

    trainer.training_step(model, batch, nn.MSELoss())

    mock_optimizer.zero_grad.assert_called_once_with(set_to_none=True)


def test_performance_monitor():
    @performance_monitor
    def dummy_task():
        return "done"

    assert dummy_task() == "done"


@patch("mriforge.infrastructure.performance_optimizer.resolve_service")
def test_get_performance_optimizer_singleton(mock_resolve):
    # test caching behavior. The getter is deprecated (dormant module,
    # 2026-07-01) and the repo's filterwarnings promotes mriforge
    # DeprecationWarnings to errors — consume it explicitly.
    with pytest.warns(DeprecationWarning, match="dormant"):
        opt1 = get_performance_optimizer()
    with pytest.warns(DeprecationWarning, match="dormant"):
        opt2 = get_performance_optimizer()
    assert opt1 is opt2


@patch("mriforge.infrastructure.performance_optimizer.resolve_service")
def test_get_performance_optimizer_is_deprecated(mock_resolve):
    """2026-07-01: the module is dormant (no training-path consumer); the
    getter must warn and point adopters at the live ModelBuilder.compile()
    path so nobody adopts the silent-fallback compile in here (pitfall #9)."""
    with pytest.warns(DeprecationWarning, match="ModelBuilder.compile"):
        get_performance_optimizer()


def test_create_optimized_data_loader_is_deleted():
    """D22 (2026-08-05): the fourth ``DataLoader`` construction site is gone.

    It had zero callers anywhere — the one apparent caller,
    ``scripts/hpo/performance_optimization_experiment.py``, defines its OWN
    method of the same name and does not inherit this class. It was deleted
    rather than rerouted because it was a DIVERGENT sibling, not unfinished
    capability: a plain-``dict`` config instead of the frozen SSOT, a hardcoded
    ``shuffle=True``, and neither a collate strategy nor ``worker_init_fn``
    seeding. Re-adding it forks loader construction a fourth way.
    """
    assert not hasattr(PerformanceOptimizer, "create_optimized_data_loader")


def test_module_no_longer_imports_dataloader():
    """The import went with the method; a lingering one invites a re-add and
    would be flagged by scripts/ci/check_dataloader_construction_ssot.py."""
    from mriforge.infrastructure import performance_optimizer

    assert not hasattr(performance_optimizer, "DataLoader")
