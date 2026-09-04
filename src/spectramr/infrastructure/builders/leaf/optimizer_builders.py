"""Phase 1: Optimizer Builders for Training Components

Implements fluent builders for creating optimizers and learning rate schedulers:
- OptimizerBuilder: Creates Adam, SGD, AdamW, etc.

SchedulerBuilder and GradScalerBuilder were removed: their only caller was
the unreachable TrainingPipelineDirector. Scheduler construction lives in
``infrastructure/training/scheduler_resolution.py``; the AMP GradScaler is
owned by ``infrastructure/training/optimizers/amp_policy.py``.

Each builder provides a fluent API for configuration and instantiation.
"""

import logging
from typing import Any

import torch.optim as optim

from spectramr.infrastructure.builders.context import (
    BuilderContext,
    accepts_builder_context,
)
from spectramr.infrastructure.builders.core import FluentBuilder
from spectramr.infrastructure.training.optimizer_registry import (
    accepted_optimizer_kwargs,
)
from spectramr.infrastructure.training.optimizer_resolution import (
    OptimizerSpec,
    build_optimizer_from_spec,
    resolve_optimizer_spec,
)

logger = logging.getLogger(__name__)


class OptimizerBuilder(FluentBuilder[optim.Optimizer]):
    """Builder for creating optimizer instances.

    Supports Adam, SGD, AdamW, RMSprop, and other PyTorch optimizers.

    Example:
        >>> builder = OptimizerBuilder(config, params=model.parameters())
        >>> optimizer = (builder
        ...     .with_type("Adam")
        ...     .with_learning_rate(1e-3)
        ...     .with_weight_decay(1e-4)
        ...     .validate()
        ...     .build())
    """

    @accepts_builder_context
    def __init__(self, ctx: BuilderContext) -> None:
        """Initialize optimizer builder.

        Args:
            ctx: Builder context carrying ``config`` and the optional ``params``
                (model parameters to optimize). ``params`` is ``None`` when the
                builder is constructed config-only.
        """
        super().__init__()
        self._config = ctx.config
        self._params = ctx.params
        self._model = None
        self._lr_multiplier: float = 1.0
        self._optimizer_type: str | None = None
        self._learning_rate: float | None = None
        self._role: str = "generator"
        # ``None`` means "the caller did not say" -- the config decides. These
        # used to be literal defaults (0.0 / (0.9, 0.999) / 1e-8) that shadowed
        # the config every time, so the same TrainingSettings built a different
        # optimizer here than through ``resolve_optimizer_spec``. The schema's
        # own beta1 default is 0.5, not 0.9, so the divergence hit every caller
        # that did not spell the betas out.
        self._weight_decay: float | None = None
        self._betas: tuple | None = None
        self._eps: float | None = None
        self._kwargs: dict[str, Any] = {}
        logger.info("OptimizerBuilder initialized")

    def with_model(self, model: Any) -> "OptimizerBuilder":
        """Set model to extract parameter groups from.

        Args:
            model: Neural network model

        Returns:
            self for chaining
        """
        self._model = model
        return self

    def with_lr_multiplier(self, multiplier: float) -> "OptimizerBuilder":
        """Set learning rate multiplier (mostly for GANs).

        Args:
            multiplier: Float multiplier

        Returns:
            self for chaining
        """
        self._lr_multiplier = multiplier
        return self

    def with_type(self, optimizer_type: str) -> "OptimizerBuilder":
        """Set optimizer type.

        Args:
            optimizer_type: Optimizer class name (Adam, SGD, AdamW, etc)

        Returns:
            self for chaining
        """
        self._optimizer_type = optimizer_type
        return self

    def with_learning_rate(self, lr: float) -> "OptimizerBuilder":
        """Set learning rate.

        Args:
            lr: Learning rate value

        Returns:
            self for chaining
        """
        self._learning_rate = lr
        return self

    def with_role(self, role: str) -> "OptimizerBuilder":
        """Declare which model this optimizer drives.

        ``"discriminator"`` picks up ``optimization.optimizer.discriminator_learning_rate``
        (the two-timescale update rule) the way the config-driven builder does.
        Without this seam the scripting path had no way to say which optimizer
        it was building, so ``fit(paradigm='gan')`` trained D at G's LR and TTUR
        was silently off.

        An explicit :meth:`with_learning_rate` still wins -- the caller said a
        number, and the role is only a default.

        Args:
            role: ``"generator"`` (default) or ``"discriminator"``.

        Returns:
            self for chaining
        """
        if role not in ("generator", "discriminator"):
            raise ValueError(
                f"Unknown optimizer role {role!r}; expected 'generator' or 'discriminator'."
            )
        self._role = role
        return self

    def with_weight_decay(self, weight_decay: float) -> "OptimizerBuilder":
        """Set weight decay (L2 regularization).

        Args:
            weight_decay: Weight decay coefficient

        Returns:
            self for chaining
        """
        self._weight_decay = weight_decay
        return self

    def with_betas(self, beta1: float, beta2: float) -> "OptimizerBuilder":
        """Set Adam betas (momentum coefficients).

        Args:
            beta1: First moment coefficient (0.9 typical)
            beta2: Second moment coefficient (0.999 typical)

        Returns:
            self for chaining
        """
        self._betas = (beta1, beta2)
        return self

    def with_eps(self, eps: float) -> "OptimizerBuilder":
        """Set numerical stability epsilon.

        Args:
            eps: Small value for numerical stability

        Returns:
            self for chaining
        """
        self._eps = eps
        return self

    def with_parameter(self, key: str, value: Any) -> "OptimizerBuilder":
        """Set custom optimizer parameter.

        Args:
            key: Parameter name
            value: Parameter value

        Returns:
            self for chaining
        """
        self._kwargs[key] = value
        return self

    def validate(self) -> "OptimizerBuilder":
        """Validate builder state.

        Returns:
            self for chaining

        Raises:
            ValueError: If required parameters missing
        """
        super().validate()

        if self._optimizer_type is None:
            # Was ``getattr(..., "optimizer", "Adam")`` — a literal default-on-
            # missing. Unreachable from the live path (which always sets the type)
            # but a silent fallback in a validate() method is still a silent
            # fallback: a scripting caller that forgot to set the type got Adam
            # and no indication of it.
            self._optimizer_type = getattr(self._config.optimization.optimizer, "type", None)
            if not self._optimizer_type:
                raise ValueError(
                    "OptimizerBuilder requires an optimizer type: call "
                    ".with_type(...) or set optimization.optimizer.type."
                )

        if self._learning_rate is None:
            # TTUR: an explicit optimization.optimizer.discriminator_learning_rate is D's
            # LR, exactly as OptimizationBuilder resolves it. Falls back to the
            # shared learning_rate when the knob is unset, so a non-TTUR arm is
            # unchanged.
            #
            # Read off `optimization.optimizer`, NOT `optimization`, and with no
            # literal default. Left pointing at the old level these two getattrs
            # would keep "working" after phase 8 while silently building every
            # optimizer at 1e-3 -- a config-independent LR is pitfall #9 with the
            # whole run behind it, and nothing would have raised.
            optimizer = self._config.optimization.optimizer
            disc_lr = optimizer.discriminator_learning_rate
            if self._role == "discriminator" and disc_lr is not None:
                self._learning_rate = disc_lr
            else:
                self._learning_rate = optimizer.learning_rate

        if self._learning_rate <= 0:
            raise ValueError(f"Invalid learning rate: {self._learning_rate}")

        return self

    def build(self) -> optim.Optimizer:
        """Build and return optimizer instance.

        Returns:
            Configured optimizer

        Raises:
            ValueError: If validation fails
            AttributeError: If optimizer type not found
        """
        self.validate()

        # Routes through OptimizerRegistry via ``build_optimizer_from_spec``.
        #
        # This body used to do a reflective ``getattr(torch.optim, name)`` over
        # ``dir(optim)``, which accepted every public attribute of the module —
        # including non-optimizers like ``Optimizer`` and ``lr_scheduler``, which
        # resolved fine here and failed at call time. Worse, it chose which kwargs
        # to forward from a hardcoded family tuple:
        #
        #     if canonical_name in ("Adam", "AdamW"):
        #         opt_kwargs["betas"] = self._betas
        #
        # so ``betas`` was silently dropped for adamax / nadam / radam, all of
        # which accept it, while ``weight_decay`` went the other way and was
        # forwarded to lbfgs / sparseadam, which cannot take it.
        base_lr = self._learning_rate * self._lr_multiplier

        # Start from what the CONFIG resolves to, then overlay only what this
        # builder was explicitly told. Reading the config here rather than
        # re-deriving it keeps one resolver: the previous body assembled the
        # spec from builder-local defaults, so `fit(...)` and `spectramr train`
        # trained different objectives off the same YAML (pitfall #13b, the
        # loss-weight-resolver disease in the optimizer subsystem).
        candidate: dict[str, Any] = dict(resolve_optimizer_spec(self._config.optimization).params)
        if self._weight_decay is not None:
            candidate["weight_decay"] = self._weight_decay
        if self._betas is not None:
            candidate["betas"] = self._betas
        if self._eps is not None:
            candidate["eps"] = self._eps
        candidate.update(self._kwargs)

        name = str(self._optimizer_type).lower()
        accepted = accepted_optimizer_kwargs(name)
        spec = OptimizerSpec(
            name=name,
            lr=base_lr,
            role=self._role,
            # Unaccepted keys are dropped rather than raised here. This is the
            # fluent/scripting entry point: its callers pass values as arguments,
            # not through a schema, so there is no ``model_fields_set`` to tell an
            # explicit request from a carried-along default. The config-driven
            # path (``resolve_optimizer_spec``) keeps the stricter policy, and it
            # is the one experiment YAMLs go through.
            params={k: v for k, v in candidate.items() if k in accepted},
        )

        target = self._model if self._model is not None else self._params
        if target is None:
            raise ValueError("No parameters or model provided to OptimizerBuilder")

        optimizer = build_optimizer_from_spec(spec, target)
        self._product = optimizer
        logger.info(
            f"Optimizer built: {self._optimizer_type} (lr={base_lr}, "
            f"decay={self._weight_decay}, forwarded={sorted(spec.params)})"
        )
        return optimizer


__all__ = [
    "OptimizerBuilder",
]
