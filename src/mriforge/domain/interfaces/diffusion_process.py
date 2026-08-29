from abc import ABC, abstractmethod

import torch
from torch import nn


class IDiffusionProcess(ABC):
    """Interface for diffusion processes (schedulers, sampling, losses).

    ### Diffusion process sequence
    ```mermaid
    sequenceDiagram
        participant Train as Training Loop
        participant Diff as IDiffusionProcess
        participant Model as nn.Module

        Train->>Diff: q_sample(x_start, t)
        Diff-->>Train: x_t (noisy/degraded)
        Train->>Diff: p_losses(model, x_start, t)
        Diff->>Model: forward(x_t, t)
        Model-->>Diff: pred_noise
        Diff-->>Train: loss
    ```

    ### Reverse Process (Sampling)
    ```mermaid
    graph LR
        A[x_T: Pure Noise] --> B[p_sample]
        B --> C[x_{T-1}]
        C --> D[...]
        D --> E[x_0: Clean Image]
    ```
    """

    @abstractmethod
    def p_sample(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        t_index: int,
        cond: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Denoise a single step."""

    @abstractmethod
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        cond: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Full reverse diffusion process."""

    @abstractmethod
    def p_losses(
        self,
        denoise_model: nn.Module,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
        loss_type: str = "l1",
    ) -> torch.Tensor:
        """Calculate training losses."""

    @abstractmethod
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward diffusion process (add noise/degradation)."""
