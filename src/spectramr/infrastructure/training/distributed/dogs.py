import copy

import torch
import torch.nn as nn


class DistributedVolumeTrainer:
    """
    DOGS: Distributed Optimization of Gaussian Splatting / Diffusion.
    Simulates distributed training by splitting a large volume into blocks,
    training local models on each block, and aggregating weights.
    """

    def __init__(self, global_model: nn.Module, num_nodes: int = 4, lr: float = 0.001):
        """__init__.

        Args:
            global_model (nn.Module): Description.
            num_nodes (int): Description.
            lr (float): Description.
        """
        self.global_model = global_model
        self.num_nodes = num_nodes
        self.lr = lr

        # Initialize local models (clones)
        # Note: copy.deepcopy is used here to create independent model instances for simulation.
        # This duplicates the model N times in memory. For large models or real distributed setups,
        # parameter sharing or offloading strategies should be considered.
        self.local_models = [copy.deepcopy(global_model) for _ in range(num_nodes)]

        # Initialize optimizers once (to preserve momentum state across steps)
        from spectramr.infrastructure.training.builders import OptimizationBuilder

        self.optimizers = []
        for model in self.local_models:
            optimizer = OptimizationBuilder.create_single_optimizer(
                model.parameters(), learning_rate=lr, optimizer_type="adam"
            )
            self.optimizers.append(optimizer)

    def block_split(self, volume: torch.Tensor) -> list[torch.Tensor]:
        """
        Splits a 3D volume [C, D, H, W] into 'num_nodes' blocks along the depth dimension.
        (For simplicity, we split along Depth (D)).
        """
        C, D, H, W = volume.shape
        # Ensure divisible or handle remainder
        chunk_size = D // self.num_nodes if self.num_nodes > 0 else D
        if chunk_size == 0:
            chunk_size = 1  # Fallback for tiny D

        chunks = list(torch.split(volume, chunk_size, dim=1))

        # If D is not perfectly divisible, last chunk handles remainder.
        # We need exactly num_nodes chunks if possible, but torch.split might give more.
        if len(chunks) > self.num_nodes:
            # Merge excessive chunks into the last one
            last_chunk = torch.cat(chunks[self.num_nodes - 1 :], dim=1)
            chunks = chunks[: self.num_nodes - 1] + [last_chunk]

        return chunks

    def broadcast_global_weights(self) -> None:
        """
        Copy global weights to all local models.
        """
        global_state = self.global_model.state_dict()
        for model in self.local_models:
            model.load_state_dict(global_state)

    def train_local_step(self, chunks: list[torch.Tensor], criterion: nn.Module) -> float:
        """
        Simulate one training step on all nodes.
        In a real system, this runs in parallel (MPI/NCCL).
        Here we loop sequentially.
        """
        local_losses = []
        # Optimizers are pre-created in __init__
        for i, (model, chunk, optimizer) in enumerate(
            zip(self.local_models, chunks, self.optimizers, strict=False)
        ):
            optimizer.zero_grad(set_to_none=True)

            # Forward pass (assuming model reconstructs chunk)
            # CAUTION: If model expects full shape, this fails.
            # DOGS assumes the model is coordinate-based (INR) or patch-based (Diffusion).

            # Here we assume model(chunk) -> passed_output
            output = model(chunk)

            # Loss (Self-supervised or Reconstruction)
            loss = criterion(output, chunk)
            loss.backward()
            optimizer.step()
            local_losses.append(loss.detach())

        if not local_losses:
            return 0.0
        # Single host-sync after the loop instead of one per local step (NN#9)
        return float(torch.stack(local_losses).mean().item())

    def aggregate_weights(self) -> None:
        """
        Federated Averaging: Average weights from all local models into global model.
        Optimized to perform in-place accumulation on global model buffers.
        """
        with torch.no_grad():
            global_state = self.global_model.state_dict()
            local_states = [m.state_dict() for m in self.local_models]

            for key in global_state:
                param = global_state[key]

                # Check if we should average this parameter/buffer
                # Generally we average floating point parameters and buffers
                if param.is_floating_point():
                    # Accumulate: global = sum(local) / N
                    # We reuse 'param' (which references global model memory) as accumulator
                    param.zero_()

                    for local_state in local_states:
                        # Accumulate in-place
                        param.add_(local_state[key])

                    # Average
                    param.div_(self.num_nodes)
                else:
                    # For non-floating point (e.g. LongTensor buffers), usually we don't average.
                    # We keep the global value (which effectively ignores local updates for these buffers).
                    # Alternatively, we could take the first model's value or majority vote.
                    # Given 'copy.deepcopy' usage, assuming homogeneous updates or static buffers.
                    pass

    def run_epoch(self, volume: torch.Tensor, criterion: nn.Module) -> float:
        """
        Full distributed epoch.
        """
        chunks = self.block_split(volume)
        # 1. Broadcast
        self.broadcast_global_weights()
        # 2. Local Train
        avg_loss = self.train_local_step(chunks, criterion)
        # 3. Aggregate
        self.aggregate_weights()

        return avg_loss
