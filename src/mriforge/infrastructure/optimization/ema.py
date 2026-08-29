from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn


class ModelEma(nn.Module):
    """
    Exponential Moving Average of model parameters.
    Maintains a shadow copy of the model key parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: torch.device = None,
        warmup: bool = True,
        adaptive: bool = False,
        warmup_steps: int = 0,
        initial_decay: float = 0.0,
        final_decay: float = 0.999,
    ):
        """Build the shadow copy and fix the decay schedule for the whole run.

        Two mutually exclusive schedules are available, and ``adaptive``
        SUPERSEDES both ``warmup`` and the fixed ``decay`` when set (see
        :meth:`_current_decay`):

        * standard (``adaptive=False``) — fixed ``decay``, optionally ramped
          by the timm/diffusers ``(1+n)/(10+n)`` rule when ``warmup=True``.
        * adaptive (``adaptive=True``) — a LINEAR ramp ``initial_decay ->
          final_decay`` spread over ``warmup_steps`` updates, then held at
          ``final_decay``. This is the semantics of the deleted
          ``models/utils/adaptive_ema.py`` (#1294), restored rather than
          reinvented.

        ``warmup_steps`` therefore means "the length of the EMA warmup
        period" on BOTH paths — a hard update-delay gate in the training loop
        on the standard path, and this soft decay ramp on the adaptive path.
        The training loop must not also apply its delay gate when
        ``adaptive`` is set; it reads :attr:`adaptive` off this object to
        decide (``pipelines/training_loop.py``).

        Args:
            model: Live model to shadow (deep-copied at construction).
            decay: Fixed decay for the standard path. Ignored when adaptive.
            device: Optional device for the shadow copy.
            warmup: Enable the timm ramp on the standard path. Ignored when
                adaptive.
            adaptive: Use the linear ``initial_decay -> final_decay`` ramp.
            warmup_steps: Ramp length in updates. Required (> 0) when adaptive.
            initial_decay: Ramp start. Reached at ``num_updates == 0``.
            final_decay: Ramp end, held for every update past ``warmup_steps``.

        Raises:
            ValueError: adaptive with a zero-length ramp, or with inverted
                endpoints — either is a declaration that cannot be honoured,
                and degrading to a fixed decay would make the configured
                values unreadable from the observed behaviour
                (non-negotiable 3, no silent fallbacks).
        """
        if adaptive and warmup_steps <= 0:
            raise ValueError(
                "adaptive EMA requires warmup_steps > 0 (it is the ramp "
                f"length), got warmup_steps={warmup_steps}. A zero-length "
                "ramp would silently collapse to a fixed final_decay."
            )
        if adaptive and final_decay < initial_decay:
            raise ValueError(
                f"adaptive EMA requires final_decay >= initial_decay, got "
                f"final_decay={final_decay} < initial_decay={initial_decay}."
            )
        super().__init__()
        # Create a deep copy of the model
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        # When True, ramp the effective decay with num_updates (warmup) so the
        # shadow tracks the live model early; when False, use the fixed decay
        # (the EMA-lag baseline). Wired from config.ema.warmup.
        self.warmup = warmup
        # Adaptive schedule (#1294). Wired from config.ema.enable_adaptive_ema
        # and friends by ModelBuilder.build_ema.
        self.adaptive = adaptive
        self.warmup_steps = warmup_steps
        self.initial_decay = initial_decay
        self.final_decay = final_decay
        # Number of update() calls so far — drives both decay ramps below.
        self.num_updates = 0

        # Move to device if specified, else keep on same device as model
        if device is not None:
            self.module.to(device)

    def _update(self, model: nn.Module, update_fn: Any):
        """_update.

        Args:
            model (nn.Module): Description.
            update_fn (Any): Description.
        Returns:
            Any: Description.
        """
        with torch.no_grad():
            for ema_v, model_v in zip(
                self.module.state_dict().values(),
                model.state_dict().values(),
                strict=False,
            ):
                if self.device is not None:
                    model_v = model_v.to(self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def _current_decay(self) -> float:
        """Effective decay for the CURRENT ``num_updates``.

        Pure Python arithmetic on host-side ints/floats — no tensor ops and no
        device sync, so it is safe to call once per training step
        (non-negotiable 9).

        Precedence is deliberate and pinned by test: the adaptive ramp
        supersedes BOTH the timm warmup ramp and the fixed ``decay``. Applying
        the timm ramp on top of a declared ``initial_decay`` would make the
        configured value unreadable from the observed behaviour.
        """
        if self.adaptive:
            if self.num_updates < self.warmup_steps:
                progress = self.num_updates / self.warmup_steps
                return self.initial_decay + progress * (self.final_decay - self.initial_decay)
            return self.final_decay
        if self.warmup:
            return min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        return self.decay

    def update(self, model: nn.Module):
        """Update EMA parameters with num_updates-aware (warmup) decay.

        Plain fixed-decay EMA (``ema = decay*ema + (1-decay)*live``) starts the
        shadow as the RANDOM INIT (it is a deepcopy of the model at
        construction) and needs ~``1/(1-decay)`` updates to forget it: at
        ``decay=0.9999`` the half-life is ~6931 steps, so a short
        (<=3000-iter) run validates a shadow that is still ~74% random init.
        That was the Experiment-11 "validation blob" — the model was graded on
        near-random EMA weights while the live weights had genuinely learned.
        We ramp the decay with ``num_updates`` (timm/diffusers convention) so
        the shadow tracks the live model early and converges to the configured
        ``decay`` only after it has learned::

            effective_decay = min(decay, (1 + num_updates) / (10 + num_updates))
        """
        self.num_updates += 1
        decay = self._current_decay()
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.module.state_dict()
            # Blend in fused ``_foreach_lerp_`` batches instead of two kernel
            # launches per tensor. A Scalene profile of
            # experiment_11_attention_none charged this update ~18 s over 300
            # steps: the work is trivial per tensor, so the cost was launch
            # overhead proportional to the parameter COUNT. Buckets key on both
            # dtypes as well as device because ``_foreach_*`` needs a uniform
            # bucket, and a mixed-precision shadow legitimately pairs an fp32
            # ``ema_v`` with an fp16 ``model_v``. The lists are rebuilt each
            # call rather than cached, so a ``load_state_dict(assign=True)`` or
            # module rebuild can never leave us blending a stale tensor.
            buckets: dict[Any, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
            for k, ema_v in esd.items():
                if k in msd:
                    model_v = msd[k]
                    # Reconcile device: a memory-saving CPU-EMA / GPU-model config
                    # (or the reverse) would otherwise raise a device-mismatch in
                    # the in-place ops below. Move the live value onto the shadow's
                    # device before blending.
                    if model_v.device != ema_v.device:
                        model_v = model_v.to(ema_v.device)

                    if not ema_v.dtype.is_floating_point:
                        # For non-float (LongTensor buffers etc), just copy
                        ema_v.copy_(model_v)
                    elif ema_v.dtype is model_v.dtype:
                        shadows, lives = buckets.setdefault((ema_v.device, ema_v.dtype), ([], []))
                        shadows.append(ema_v)
                        lives.append(model_v)
                    else:
                        # Mixed dtypes cannot share a foreach bucket; keep the
                        # per-tensor blend, which upcasts as it always did.
                        ema_v.mul_(decay).add_(model_v, alpha=1.0 - decay)
            # ``lerp_(a, b, w) == (1 - w) * a + w * b``, so ``w = 1 - decay``
            # reproduces ``mul_(decay).add_(live, alpha=1 - decay)``. It is one
            # fused op rather than two, so results may differ in the last ulp --
            # compare EMA weights with a tolerance, never bitwise.
            for shadows, lives in buckets.values():
                torch._foreach_lerp_(shadows, lives, 1.0 - decay)

    # Key under which the warmup counter is persisted alongside the shadow
    # weights. Kept a plain Python int on the instance (no device / no per-step
    # GPU sync) and injected into ``state_dict`` only at checkpoint time.
    _NUM_UPDATES_KEY = "_ema_num_updates"

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Include the warmup counter so a resume continues the decay ramp.

        ``num_updates`` is not a buffer (keeping it off-device avoids a per-step
        ``.item()`` sync in :meth:`update`), so ``nn.Module.state_dict`` would
        otherwise drop it and every SLURM requeue would restart the ramp at 0.
        """
        sd = super().state_dict(*args, **kwargs)
        sd[self._NUM_UPDATES_KEY] = torch.tensor(int(self.num_updates), dtype=torch.long)
        return sd

    def load_state_dict(self, state_dict: Any, strict: bool = True, assign: bool = False) -> Any:
        """Restore the warmup counter, tolerating pre-fix checkpoints.

        Pops the private counter key (absent in checkpoints written before this
        field was persisted, in which case ``num_updates`` stays at its default
        0) before delegating the shadow-weight load to ``nn.Module`` so the
        extra key never trips ``strict=True`` validation.
        """
        sd = dict(state_dict)
        counter = sd.pop(self._NUM_UPDATES_KEY, None)
        if counter is not None:
            self.num_updates = int(counter)
        return super().load_state_dict(sd, strict=strict, assign=assign)

    def forward(self, *args, **kwargs):
        """forward.

        Returns:
            Any: Description.

        forward method for ModelEma.

        Executes PyTorch tensor operations.

        Args:
            None

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.module(*args, **kwargs)

    @property
    def device(self):
        """device.

        Returns:
            Any: Description.
        """
        return next(self.module.parameters()).device
