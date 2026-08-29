"""Meta-Learning Training Strategy
===============================

Training strategy for Experiment 5.1: Anatomy-Agnostic Meta-Learning.
Implements the outer loop for meta-learning (e.g. MAML/Reptile style).
"""

import logging
from typing import Any

import torch

from mriforge.infrastructure.training.strategies.base import BaseTrainingStrategy
from mriforge.models.losses.computers import UnifiedReconstructionLossComputer

logger = logging.getLogger(__name__)


class MetaLearningTrainingStrategy(BaseTrainingStrategy):
    """Meta-Learning training strategy for anatomy-agnostic model adaptation.

    Implements outer-loop meta-learning (MAML/Reptile style) enabling models to
    quickly adapt to new anatomies or imaging parameters with minimal fine-tuning.
    \n    ## Core Philosophy

    **Objective**: Learn initial weight initialization such that few gradient steps
    on new task data yield optimal performance. Contrasts with standard supervised
    learning which optimizes for a single distribution.

    **Key Insight**: Instead of learning weights W directly, learn weights W* that
    serve as good starting points for rapid task-specific adaptation.

    ## Algorithm: MAML (Model-Agnostic Meta-Learning)

    **Outer Loop** (this strategy):
    1. Sample task i (new anatomy/contrast)
    2. **Inner Loop**: Perform k gradient steps on task i with support set
       - Compute task-specific loss: L_i(θ - α·∇L_i(θ))\n    3. **Outer Update**: Meta-gradient on query set
       - Update meta-weights: θ ← θ - β·∇L_query(θ_i')

    **Inner Loop** (task adaptation, typically offline):
    1. Load pre-trained meta-weights θ*\n    2. Compute inner-loop loss on support data
    3. Inner-loop gradient update: θ_task ← θ* - α·∇L_support
    4. Evaluate on query data with θ_task

    ## Training Process

    1. **Task Sampling**: Batch of diverse anatomies/contrasts/parameters
    2. **Inner Loop**: Few-step adaptation on support set per task
    3. **Meta Loss**: Evaluate adapted model on query set
    4. **Outer Update**: Accumulate meta-gradients across tasks
    5. **Meta Weights Update**: Aggregate task gradients

    ## Configuration

    - `training.training_mode`: 'meta_learning' or 'reconstruction'
    - Inner-loop hyperparameters (``_meta_hyperparams``) are read from the FIRST
      present of: ``config.meta_learning`` / ``training.meta_learning`` /
      ``training.meta`` / ``model.adaptation_config`` / ``model.model_kwargs``,
      under any of the aliases:
      ``inner_lr`` | ``meta_lr_inner`` | ``adaptation_lr`` (default 0.01),
      ``num_inner_steps`` | ``inner_steps`` | ``adaptation_steps`` (default 5),
      and ``first_order`` (default False). The outer/meta LR is the base
      optimizer's ``optimization.optimizer.learning_rate``.

    ## Loss Components

    - **Inner Loss**: Task-specific L1/L2 on support set
    - **Meta Loss**: Generalization on query set (different from support)
    - **Reconstruction Loss**: Full pipeline (SSIM, perceptual, k-space)
    - **Regularization**: Optional L2 on meta-weights

    ## Key Features

    - **Task Distribution**: Samples from diverse anatomies/parameters
    - **Few-Shot Capability**: Adapts with 5-10 support samples
    - **Gradient Stability**: Inner/outer learning rates carefully balanced
    - **Memory Efficient**: Reuses task data in query/support split
    - **Compatible**: Works with any generator architecture

    ## Advantages for MRI

    - ✅ **Cross-anatomy generalization**: Single model for brain/cardiac/knee
    - ✅ **Parameter adaptation**: Handles variant echo times, flip angles
    - ✅ **Few-shot learning**: Adapts in 5-10 batches of new anatomy
    - ✅ **Interpretable**: Tracks adaptation speed as convergence metric\n
    ## Disadvantages & Limitations

    - ❌ Expensive: Requires backward through backward (Hessian) computation
    - ❌ Slow meta-update: Cannot use large batches effectively
    - ❌ Sensitive: Requires careful tuning of inner/outer learning rates
    - ❌ Limited: Assumes smooth loss landscape for task generalization

    Attributes:
        state: TrainingState with support/query split capability
        loss_computer: UnifiedReconstructionLossComputer for task losses
        inner_lr: Inner-loop learning rate (task-specific adaptation)
        outer_lr: Outer-loop learning rate (meta-weight update)
        device: Computation device (CUDA/CPU)

    References:
        - Finn et al. (2017): Model-Agnostic Meta-Learning for Fast Adaptation
        - Nichol et al. (2018): On First-Order Meta-Learning Algorithms
    """

    def __init__(
        self,
        env=None,
        **kwargs: Any,
    ) -> None:
        """__init__.

        Args:
            env (Any): Description.
        """
        super().__init__(env=env, **kwargs)

        # Initialize loss computer for reconstruction using unified pattern
        self.loss_computer = UnifiedReconstructionLossComputer(
            config=self.env.config, device=self.device
        )

        # We need a separate optimizer for the inner loop if not provided by model
        # But MetaLearningWrapper.adapt_to_domain creates one if needed.
        pass

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute meta-learning losses.

        Args:
            input_batch: Input images [B, C, H, W]
            target_batch: Target images [B, C, H, W]
            kwargs: May contain 'domain_id'

        We assume the batch is mixed from different domains or we simulate tasks.
        For simplicity in this integration, we split the batch into support and query.
        """
        batch_size = input_batch.shape[0]
        if batch_size < 2:
            # Cannot do meta-learning with batch size 1 without explicit
            # support/query. Per CLAUDE.md #9, this is a configuration
            # error, not a fallback: silently degrading to standard
            # training would mask a misconfigured YAML and produce a
            # passes-with-warnings outcome. See
            # TODO/audit/06_strategies_specialised.md F8.
            raise ValueError(
                f"meta_learning strategy requires batch_size >= 2 to split "
                f"into support/query halves; got batch_size={batch_size}. "
                "Increase data.batch_size in the YAML."
            )

        # Split into support and query
        mid = batch_size // 2
        support_x, query_x = input_batch[:mid], input_batch[mid:]
        support_y, query_y = target_batch[:mid], target_batch[mid:]

        del epoch  # meta loss is the inner objective; no epoch-warmup schedule
        model = self.env.generator

        # MAML inner/outer loop via ``torch.func.functional_call`` against the
        # model's REAL forward signature. There is NO ``adapt_to_domain``
        # contract — grep proves no source model implements it, including the
        # ``meta_learning``-registered ``meta_varnet`` — so requiring it made
        # the whole paradigm raise TypeError and never train (2026-05-31
        # audit). Second-order by default so the query (meta) loss flows
        # gradient back to the ORIGINAL params that opt_g steps; flip
        # ``training.meta_learning.first_order=true`` for the cheaper FOMAML.
        from torch.func import functional_call

        inner_lr, inner_steps, first_order = self._meta_hyperparams()
        params = {name: p for name, p in model.named_parameters()}

        def _forward(p: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
            try:
                out = functional_call(model, p, (x,))
            except TypeError:
                # Models that also need an explicit sampling mask (e.g.
                # meta_varnet ``forward(masked_kspace, mask, ...)``).
                mask = kwargs.get("mask")
                if mask is None:
                    mask = torch.ones_like(x)
                out = functional_call(model, p, (x, mask))
            if isinstance(out, dict):
                out = out.get("reconstruction", next(iter(out.values())))
            if isinstance(out, (tuple, list)):
                out = out[0]
            return out

        # Meta-learning differentiates THROUGH the forward pass, so grad must be
        # enabled even if the caller is in an eval / no_grad context (a
        # validation pass, or grad left disabled by a prior step). Force it on.
        with torch.enable_grad():
            # --- Inner loop: SGD adaptation on the support set ---
            adapted = params
            support_loss = support_x.new_zeros(())
            for _ in range(inner_steps):
                support_loss = self._inner_loss(_forward(adapted, support_x), support_y)
                grads = torch.autograd.grad(
                    support_loss,
                    list(adapted.values()),
                    create_graph=not first_order,
                    allow_unused=True,
                )
                adapted = {
                    name: (p if g is None else p - inner_lr * g)
                    for (name, p), g in zip(adapted.items(), grads)
                }

            # --- Outer loop: meta loss on the query set with adapted params ---
            meta_loss = self._inner_loss(_forward(adapted, query_x), query_y)

        return {
            "g_total_loss": meta_loss,
            "meta_loss": meta_loss.detach(),
            "adaptation_loss": support_loss.detach(),
        }

    def _inner_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Inner/meta objective. Prefer a configured loss from ``env.losses``;
        fall back to the registry L1 (SSOT — never a hardcoded ``F.l1_loss``).
        Guards ``isinstance(dict)`` so a MagicMock ``env.losses`` (unit tests)
        falls through to the registry instead of iterating garbage."""
        losses = getattr(self.env, "losses", None)
        if isinstance(losses, dict) and losses:
            # Deterministic selection: prefer a stable, documented key rather
            # than ``next(iter(...))`` (insertion-order arbitrary). Fall back to
            # the lexicographically-first key so the inner objective is
            # reproducible across runs and dict orderings.
            for preferred in ("l1", "reconstruction", "mse", "l2"):
                if preferred in losses:
                    return losses[preferred](pred, target)
            return losses[sorted(losses)[0]](pred, target)
        from mriforge.models.losses.registry import create_loss

        return create_loss("l1")(pred, target)

    def _meta_hyperparams(self) -> tuple[float, int, bool]:
        """``(inner_lr, inner_steps, first_order)`` scanned across EVERY known
        config home so the knob is never a silent no-op (CLAUDE.md pitfall #15):
        a dedicated ``meta_learning`` block (root or under ``training``), the
        legacy ``training.meta`` block, and ``model.model_kwargs`` (where
        ``exp_meta_varnet.yaml`` actually puts ``meta_lr_inner`` / ``inner_steps``).
        Robust to MockConfig (its ``__float__``/``__int__`` collapse-to-0 is
        treated as "unset") and to dict- or attribute-style blocks."""
        cfg = getattr(self, "config", None) or getattr(self.env, "config", None)
        training = getattr(cfg, "training", None) if cfg is not None else None
        model = getattr(cfg, "model", None) if cfg is not None else None
        model_kwargs = getattr(model, "model_kwargs", None) or {}

        blocks: list[Any] = [
            getattr(cfg, "meta_learning", None) if cfg is not None else None,
            getattr(training, "meta_learning", None) if training is not None else None,
            getattr(training, "meta", None) if training is not None else None,
            # v6 home: MetaLearningModelConfig.adaptation_config (AdaptationConfig
            # → adaptation_lr / adaptation_steps), see schemas/training/meta_learning.py.
            getattr(model, "adaptation_config", None) if model is not None else None,
            model_kwargs,
        ]

        def _attr(obj: Any, name: str) -> Any:
            if isinstance(obj, dict):
                return obj.get(name)
            return getattr(obj, name, None)

        def _coerce(value: Any, default: float | int) -> float | int:
            try:
                coerced = type(default)(value)
            except (TypeError, ValueError):
                return default
            return coerced if coerced else default

        def _read(names: tuple[str, ...], default: float | int) -> float | int:
            for block in blocks:
                if block is None:
                    continue
                for name in names:
                    coerced = _coerce(_attr(block, name), default)
                    if coerced != default:
                        return coerced
            return default

        inner_lr = _read(("inner_lr", "meta_lr_inner", "adaptation_lr"), 0.01)
        inner_steps = _read(("num_inner_steps", "inner_steps", "adaptation_steps"), 1)
        first_order = False
        for block in blocks:
            value = _attr(block, "first_order") if block is not None else None
            if isinstance(value, bool):
                first_order = value
                break
        return float(inner_lr), max(1, int(inner_steps)), first_order
