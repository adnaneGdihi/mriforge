import torch
import torch.nn as nn

from mriforge.infrastructure.training.distributed.dogs import DistributedVolumeTrainer


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
        # Add a buffer
        self.register_buffer("running_count", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        self.running_count += 1
        return self.fc(x)


def test_dogs_refactor_functionality():
    # Setup
    global_model = SimpleModel()

    # Just instantiate and check if optimizers are created
    trainer = DistributedVolumeTrainer(global_model, num_nodes=2, lr=0.01)
    assert hasattr(trainer, "optimizers")
    assert len(trainer.optimizers) == 2


def test_aggregate_weights_logic():
    # Test the in-place aggregation logic specifically
    global_model = SimpleModel()
    trainer = DistributedVolumeTrainer(global_model, num_nodes=2)

    # Modify local models
    with torch.no_grad():
        trainer.local_models[0].fc.weight.fill_(1.0)
        trainer.local_models[1].fc.weight.fill_(3.0)

    trainer.aggregate_weights()

    # Expected: (1+3)/2 = 2
    assert torch.allclose(trainer.global_model.fc.weight, torch.tensor([[2.0] * 10]))

    # Verify optimizers exist (if we refactor)
    assert hasattr(trainer, "optimizers")
    assert len(trainer.optimizers) == 2
