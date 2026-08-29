"""Model-Agnostic Meta-Learning (MAML) Engine.

Implements the meta-learning training loop using PyTorch's functional API
(torch.func) for efficient gradients-through-gradients.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.func import functional_call, grad


class MAML:
    """MAML Trainer for MRI Reconstruction tasks."""

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        inner_steps: int = 3,
        first_order: bool = False,
    ):
        """__init__.

        Args:
            model (nn.Module): Description.
            inner_lr (float): Description.
            inner_steps (int): Description.
            first_order (bool): Description.
        """
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        # Extract initial parameters
        self.initial_params = dict(model.named_parameters())
        self.initial_buffers = dict(model.named_buffers())

    def adaptation_step(
        self,
        params: dict[str, Tensor],
        buffers: dict[str, Tensor],
        support_kspace: Tensor,
        support_mask: Tensor,
        support_target: Tensor,
    ) -> dict[str, Tensor]:
        """Perform one inner loop step (gradient descent)."""

        # Define loss function wrapper for func.grad
        def compute_loss(p, b, k, m, t):
            # Functional call to model using p and b
            """compute_loss.

            Args:
                p (Any): Description.
                b (Any): Description.
                k (Any): Description.
                m (Any): Description.
                t (Any): Description.
            Returns:
                Any: Description.
            """
            pred = functional_call(self.model, (p, b), (k, m))
            # Magnitude loss (or complex)
            pred_abs = (
                torch.abs(torch.view_as_complex(torch.view_as_real(pred)))
                if pred.is_complex()
                else pred
            )
            t_abs = torch.abs(torch.view_as_complex(torch.view_as_real(t))) if t.is_complex() else t
            return torch.nn.functional.mse_loss(pred_abs, t_abs)

        # Compute gradient w.r.t params
        grads = grad(compute_loss)(params, buffers, support_kspace, support_mask, support_target)

        # Update params: theta' = theta - alpha * grad
        updated_params = {
            name: p - self.inner_lr * g
            for (name, p), g in zip(params.items(), grads.values(), strict=False)
        }

        return updated_params

    def inner_loop(
        self,
        support_batch: tuple[Tensor, Tensor, Tensor],
    ) -> dict[str, Tensor]:
        """Run full inner loop adaptation on support set."""
        kspace, mask, target = support_batch

        current_params = self.initial_params
        current_buffers = self.initial_buffers  # Buffers usually fixed or updated differently?
        # In BN/IN, we might update running stats. Here assume fixed for simplicity or functional update.

        for _ in range(self.inner_steps):
            current_params = self.adaptation_step(
                current_params, current_buffers, kspace, mask, target
            )

        return current_params

    def outer_loss(
        self,
        query_batch: tuple[Tensor, Tensor, Tensor],
        adapted_params: dict[str, Tensor],
    ) -> Tensor:
        """Compute loss on query set using adapted parameters."""
        kspace, mask, target = query_batch

        # Note: We use adapted_params here.
        # Gradients propagate through specific path of adaptation if not detached.
        # functional_call supports autograd tracking.

        pred = functional_call(self.model, (adapted_params, self.initial_buffers), (kspace, mask))

        pred_abs = (
            torch.abs(torch.view_as_complex(torch.view_as_real(pred)))
            if pred.is_complex()
            else pred
        )
        t_abs = (
            torch.abs(torch.view_as_complex(torch.view_as_real(target)))
            if target.is_complex()
            else target
        )

        return torch.nn.functional.mse_loss(pred_abs, t_abs)

    def train_step(
        self,
        task_batch: tuple[
            tuple[Tensor, Tensor, Tensor],  # Support
            tuple[Tensor, Tensor, Tensor],  # Query
        ],
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Perform one meta-training step."""
        optimizer.zero_grad()

        support, query = task_batch

        # 1. Outer Loop: We need gradients w.r.t self.initial_params (which are self.model.parameters)
        # However, adaptation creates new tensors.

        # Warning: 'dict(model.named_parameters())' returns dict of Tensors.
        # If we use them in functional_call, operations are recorded.
        # BUT 'adaptation_step' creates NEW tensors.
        # If we want to optimize 'initial_params', we need to ensure the computation graph
        # connects 'adapted_params' back to 'initial_params'.

        # MAML logic:
        # p0 = model.params
        # p1 = p0 - lr * grad(L_sup, p0)
        # L_qry(p1) backward -> grad w.r.t p0

        # Standard PyTorch nn.Parameters are leaves.
        # functional_call uses values.

        # To ensure gradients flow, we should NOT extract 'initial_params' as detached dict in __init__.
        # We should extract them *at step time* from the model so they are part of the graph.

        # Re-extract params for this step to ensure graph connectivity
        params = dict(self.model.named_parameters())
        buffers = dict(self.model.named_buffers())

        # Inner Loop
        adapted_params = params
        for _ in range(self.inner_steps):
            # We need to manually call grad to allow higher-order derivatives
            # self.adaptation_step uses torch.func.grad which handles higher order if inputs require grad?
            # torch.func.grad returns gradients.
            # If params requires_grad=True, does torch.func.grad allow resulting gradients to be differentiable?
            # Yes, if we don't detach.

            # Re-implement adaptation loop here to be safe about graph
            def loss_fn(p, b):
                """loss_fn.

                Args:
                    p (Any): Description.
                    b (Any): Description.
                Returns:
                    Any: Description.
                """
                kw, mw, tw = support  # kspace, mask, target
                pred_w = functional_call(self.model, (p, b), (kw, mw))
                pred_w_abs = (
                    torch.abs(torch.view_as_complex(torch.view_as_real(pred_w)))
                    if pred_w.is_complex()
                    else pred_w
                )
                t_w_abs = (
                    torch.abs(torch.view_as_complex(torch.view_as_real(tw)))
                    if tw.is_complex()
                    else tw
                )
                return torch.nn.functional.mse_loss(pred_w_abs, t_w_abs)

            # Compute grads
            # torch.func.grad computes grad of output w.r.t inputs (params)
            # This is functional.
            grads = grad(loss_fn)(adapted_params, buffers)

            # GD
            adapted_params = {
                k: adapted_params[k] - self.inner_lr * grads[k] for k in adapted_params.keys()
            }

        # Outer Loss
        qk, qm, qt = query
        pred_q = functional_call(self.model, (adapted_params, buffers), (qk, qm))
        pred_q_abs = (
            torch.abs(torch.view_as_complex(torch.view_as_real(pred_q)))
            if pred_q.is_complex()
            else pred_q
        )
        qt_abs = torch.abs(torch.view_as_complex(torch.view_as_real(qt))) if qt.is_complex() else qt

        loss_outer = torch.nn.functional.mse_loss(pred_q_abs, qt_abs)

        # Optimization
        loss_outer.backward()
        optimizer.step()

        return loss_outer.item()
