"""WS1-core-04: the diffusion strategy must honor the configured validation
reverse-diffusion step count, with the documented precedence
``validation.sampler_steps`` → ``training.diffusion.sampling_steps`` → None
(None = the sampler keeps its own default, so unconfigured arms are unaffected).
"""

from __future__ import annotations

from types import SimpleNamespace

from tests.utils.config_block_stub import block_stub

from mriforge.infrastructure.training.strategies.diffusion import (
    DiffusionTrainingStrategy,
)


def _resolve(
    sampler_steps: int | None,
    training_steps: int | None,
    num_timesteps: int | None = None,
) -> int | None:
    stub = SimpleNamespace(
        num_timesteps=num_timesteps,
        config=SimpleNamespace(
            validation=block_stub(
            "validation",
            sampler_steps=sampler_steps),
            training=SimpleNamespace(
                diffusion=SimpleNamespace(sampling_steps=training_steps)
            ),
        ),
    )
    # Unbound-method call on a stub avoids constructing the heavy strategy.
    return DiffusionTrainingStrategy._resolve_validation_sampling_steps(stub)


def test_validation_sampler_steps_takes_precedence() -> None:
    assert _resolve(40, 100) == 40


def test_falls_back_to_training_sampling_steps() -> None:
    assert _resolve(None, 100) == 100


def test_none_when_neither_configured() -> None:
    assert _resolve(None, None) is None


def test_clamped_to_num_timesteps() -> None:
    # experiment_12 case: sampler_steps=50 but only 16 forward steps → clamp.
    assert _resolve(50, None, num_timesteps=16) == 16
    # within range → unchanged.
    assert _resolve(10, None, num_timesteps=16) == 10
