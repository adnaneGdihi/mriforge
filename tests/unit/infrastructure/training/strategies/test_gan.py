"""Regression: GANTrainingStrategy.train_step must thread the LIVE iteration.

Review finding (2026-07-01): ``train_step`` declares an explicit ``iteration``
parameter, so the training loop's ``strategy.train_step(..., iteration=iteration)``
call binds the real value to that parameter and leaves ``**kwargs`` empty. The
old ``iteration = kwargs.get("iteration", kwargs.get("step", 0))`` therefore
clobbered the correct value back to 0, silently degrading the R1-regularization
cadence (``UnifiedGANLossComputer._should_apply_r1``) from
every-``r1_interval``-iterations to every step of every ``r1_interval``-th epoch.

The fix reads the loop_state seam (``resolve_loop_iteration``) — the same source
already used elsewhere in the file — so the live iteration reaches the
discriminator/generator loss closures.
"""

from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

from mriforge.infrastructure.training.loop_state import LoopState  # noqa: E402
from mriforge.infrastructure.training.strategies import gan as gan_mod  # noqa: E402
from mriforge.infrastructure.training.strategies.gan import (  # noqa: E402
    GANTrainingStrategy,
)


def test_train_step_uses_live_loop_iteration(monkeypatch: pytest.MonkeyPatch) -> None:
    # One discriminator update per step keeps the loop deterministic.
    monkeypatch.setattr(gan_mod, "_resolve_disc_updates", lambda cfg: 1)

    strategy = object.__new__(GANTrainingStrategy)
    strategy._step_counter = 0
    strategy.config = MagicMock()
    strategy.loop_state = LoopState(iteration=42)

    env = MagicMock()
    env.discriminator = MagicMock()
    env.generator = MagicMock()
    env.losses = {}
    strategy.env = env

    strategy._to_device = lambda x: x  # identity; inputs already on CPU

    captured: dict[str, int] = {}

    def _capture_disc(inp, tgt, disc, epoch, iteration, losses):  # noqa: ANN001
        captured["disc_iteration"] = iteration
        return lambda: {}

    def _capture_gen(inp, tgt, disc, epoch, iteration, losses):  # noqa: ANN001
        captured["gen_iteration"] = iteration
        return lambda: {}

    strategy._train_discriminator_step = _capture_disc
    strategy._train_generator_step = _capture_gen

    inp = torch.zeros(2, 1, 8, 8)
    tgt = torch.zeros(2, 1, 8, 8)
    # Pass a bogus iteration= keyword: it binds to the named parameter and must
    # be IGNORED in favour of the loop_state seam (the exact bug being fixed).
    strategy.train_step(None, 0, input_batch=inp, target_batch=tgt, iteration=999)

    assert captured["disc_iteration"] == 42, "R1 gate must see the live iteration"
    assert captured["gen_iteration"] == 42


# --- #707: the per-step metrics accessor must not sync the GPU ---------------


def test_get_last_metrics_does_not_sync_the_gpu(no_gpu_sync):
    from mriforge.infrastructure.training.strategies.gan import GANTrainingStrategy

    no_gpu_sync(GANTrainingStrategy.get_last_metrics)
