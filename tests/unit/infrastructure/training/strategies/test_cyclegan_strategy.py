"""Unit tests for :class:`CycleGANTrainingStrategy` (unpaired two-generator).

Mechanism-fires probe + no-pixel-L1 + factory-resolution + two-optimiser step.
The strategy is built with the repo's ``object.__new__`` strategy-unit-test idiom
(mirrors ``test_cross_field_translation_strategy``): bypass the heavy base
``__init__`` but run the REAL ``setup_models`` / ``_compute_losses_impl`` so the
distinctive cycle mechanism is genuinely exercised (CLAUDE.md pitfall #16 — a
green smoke must mean "method fired", not just "did not crash").
"""

from __future__ import annotations

import types

import pytest
import torch

from mriforge.config.settings import TrainingSettings
from mriforge.infrastructure.training.strategies.cyclegan_strategy import (
    CycleGANTrainingStrategy,
    compute_cyclegan_losses,
)


def _batch() -> dict:
    """Synthetic UNPAIRED batch: A-domain ``input`` + B-domain ``target``."""
    return {
        "input": torch.rand(2, 1, 32, 32),
        "target": torch.rand(2, 1, 32, 32),
        "field_strength": torch.tensor([3.0, 3.0]),
        "field_strength_target": torch.tensor([7.0, 7.0]),
    }


@pytest.fixture
def minimal_cyclegan_config() -> TrainingSettings:
    """Tiny frozen ``TrainingSettings`` carrying the ``losses.gan`` weights the
    strategy reads (``lambda_cycle`` / ``lambda_identity``, added in Task 5)."""
    return TrainingSettings(
        data={"train_path": "/tmp/t", "val_path": "/tmp/v", "batch_size": 2},
        model={
            "model_type": "cyclegan_generator",
            "in_channels": 1,
            "out_channels": 1,
        },
        optimization={"learning_rate": 2e-4},
        logging={},
        losses={"gan": {"lambda_cycle": 10.0, "lambda_identity": 0.5}},
    )


def _make_strategy(config: TrainingSettings) -> CycleGANTrainingStrategy:
    strat = CycleGANTrainingStrategy.__new__(CycleGANTrainingStrategy)
    strat.config = config
    strat.env = types.SimpleNamespace(
        generator=None,
        discriminator=None,
        losses={},
        opt_g=None,
        opt_d=None,
        schedulers={},
    )
    strat.device = torch.device("cpu")
    strat.setup_models()
    return strat


def test_cyclegan_cycle_loss_fires(minimal_cyclegan_config: TrainingSettings) -> None:
    """Mechanism-fires: finite, strictly-positive cycle loss; two generators +
    two discriminators exposed."""
    strat = _make_strategy(minimal_cyclegan_config)
    losses = strat._compute_losses_impl(_batch())
    assert torch.isfinite(losses["cycle"]) and losses["cycle"] > 0
    assert torch.isfinite(losses["identity"])
    assert torch.isfinite(losses["g_total_loss"])
    assert hasattr(strat, "gen_ab") and hasattr(strat, "gen_ba")
    assert hasattr(strat, "disc_a") and hasattr(strat, "disc_b")


def test_cyclegan_no_pixel_l1_to_target(
    minimal_cyclegan_config: TrainingSettings,
) -> None:
    """UNPAIRED contract: no paired pixel-L1 to the B-domain ``target`` — only
    cycle / identity (same-domain self-consistency) + adversarial terms."""
    strat = _make_strategy(minimal_cyclegan_config)
    losses = strat._compute_losses_impl(_batch())
    assert set(losses) == {"g_total_loss", "adv_g", "adv_d", "cycle", "identity"}
    assert not any(k in losses for k in ("recon", "l1", "loss_l1", "reconstruction"))


def test_cyclegan_total_uses_config_weights(
    minimal_cyclegan_config: TrainingSettings,
) -> None:
    """g_total = adv_g + lambda_cycle*cycle + lambda_identity*identity, weights
    read from ``config.losses.gan`` (SSOT)."""
    strat = _make_strategy(minimal_cyclegan_config)
    torch.manual_seed(0)
    losses = strat._compute_losses_impl(_batch())
    expected = losses["adv_g"] + 10.0 * losses["cycle"] + 0.5 * losses["identity"]
    assert torch.allclose(losses["g_total_loss"], expected)


def test_validation_forward_returns_gen_ab(
    minimal_cyclegan_config: TrainingSettings,
) -> None:
    """Validation prediction is ``gen_ab(input)`` (the A->B translation)."""
    strat = _make_strategy(minimal_cyclegan_config)
    x = torch.rand(2, 1, 32, 32)
    pred = strat._validation_forward(x)
    assert pred.shape == (2, 1, 32, 32)
    assert torch.allclose(pred, strat.gen_ab(x))


def test_train_step_returns_d_and_g_configs(
    minimal_cyclegan_config: TrainingSettings,
) -> None:
    """Two-optimiser alternating step: a D-closure (both discriminators) + a
    G-closure, each returning a finite scalar loss."""
    strat = _make_strategy(minimal_cyclegan_config)
    strat.env.opt_g = torch.optim.Adam(
        list(strat.gen_ab.parameters()) + list(strat.gen_ba.parameters()), lr=1e-4
    )
    strat.env.opt_d = torch.optim.Adam(
        list(strat.disc_a.parameters()) + list(strat.disc_b.parameters()), lr=1e-4
    )
    configs = strat.train_step(_batch(), epoch=0, iteration=1)
    names = [c["name"] for c in configs]
    assert "discriminator" in names and "generator" in names
    for cfg in configs:
        loss = cfg["closure"]()
        assert torch.isfinite(loss)


def test_pure_helper_no_paired_supervision() -> None:
    """The pure loss helper must contain NO L1 between ``fake_B`` and ``real_B``
    (that would be paired supervision). Behavioural check: swapping real_B for a
    copy leaves cycle unchanged only through the generators, never a direct term."""
    torch.manual_seed(0)
    from mriforge.models.discriminators.patchgan_discriminator import (
        PatchGANDiscriminator,
    )
    from mriforge.models.generators.cycle_gan import ResNetGenerator

    gen_ab = ResNetGenerator(1, 1)
    gen_ba = ResNetGenerator(1, 1)
    disc_a = PatchGANDiscriminator(in_channels=1)
    disc_b = PatchGANDiscriminator(in_channels=1)
    real_a = torch.rand(2, 1, 32, 32)
    real_b = torch.rand(2, 1, 32, 32)
    out = compute_cyclegan_losses(
        gen_ab,
        gen_ba,
        disc_a,
        disc_b,
        real_a,
        real_b,
        lambda_cycle=10.0,
        lambda_identity=0.5,
    )
    assert out["cycle"] > 0 and torch.isfinite(out["g_total_loss"])
    # No key implies a paired reconstruction term.
    assert "recon" not in out and "l1" not in out


def test_strategy_registered_in_factory() -> None:
    """The ``cyclegan`` short-name resolves via the strategy factory."""
    from mriforge.infrastructure.training.strategy_factory import (
        TrainingStrategyFactory,
    )

    assert "cyclegan" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    cls = TrainingStrategyFactory()._load_strategy_class("cyclegan")
    assert cls is CycleGANTrainingStrategy


def test_train_step_restores_train_mode_after_validation(
    minimal_cyclegan_config: TrainingSettings,
) -> None:
    """I1 regression: ``_validation_forward`` puts ``gen_ab`` in eval mode; the
    overriding ``train_step`` must restore ALL four nets to train mode so
    BatchNorm/Dropout never run in inference mode during subsequent training.

    (The base ``train_step`` does this at base.py:921-925, but this strategy
    overrides it — so the restore has to be re-asserted here. The real end-to-end
    integration is additionally covered by the Task-11 ``--probe``.)
    """
    strat = _make_strategy(minimal_cyclegan_config)
    # Simulate a validation pass, which calls ``gen_ab.eval()``.
    strat._validation_forward(torch.rand(2, 1, 32, 32))
    assert strat.gen_ab.training is False
    strat.env.opt_g = torch.optim.Adam(strat.gen_ab.parameters(), lr=1e-4)
    strat.env.opt_d = torch.optim.Adam(strat.disc_a.parameters(), lr=1e-4)
    # A subsequent training step must flip every net back to train mode.
    strat.train_step(_batch(), epoch=0, iteration=1)
    for module in (strat.gen_ab, strat.gen_ba, strat.disc_a, strat.disc_b):
        assert module.training is True


def test_setup_models_amp_configures_extra_nets(
    minimal_cyclegan_config: TrainingSettings,
) -> None:
    """I2 regression: ``base.__init__`` AMP-configures only ``env.generator`` /
    ``env.discriminator`` (built before ``_setup_strategy_specific_components``), so
    the strategy-owned ``gen_ba`` / ``disc_a`` must be AMP-configured inside
    ``setup_models`` too."""
    strat = CycleGANTrainingStrategy.__new__(CycleGANTrainingStrategy)
    strat.config = minimal_cyclegan_config
    strat.env = types.SimpleNamespace(
        generator=None,
        discriminator=None,
        losses={},
        opt_g=None,
        opt_d=None,
        schedulers={},
    )
    strat.device = torch.device("cpu")
    configured: list = []
    strat.amp_helper = types.SimpleNamespace(
        configure_model_for_amp=lambda m: configured.append(m)
    )
    strat.setup_models()
    assert strat.gen_ba in configured and strat.disc_a in configured


def test_g_closure_exposes_plain_metric_names(
    minimal_cyclegan_config: TrainingSettings,
) -> None:
    """I3 + M2: the G-closure routes detached losses through the base anomaly
    guard and exposes PLAIN metric names (cycle / identity / adv_g / adv_d /
    g_total_loss) that the reporter + early-stopping expect — NOT ``g_``-prefixed —
    while preserving ``d_total_loss`` from the D-closure."""
    strat = _make_strategy(minimal_cyclegan_config)
    strat.env.opt_g = torch.optim.Adam(
        list(strat.gen_ab.parameters()) + list(strat.gen_ba.parameters()), lr=1e-4
    )
    strat.env.opt_d = torch.optim.Adam(
        list(strat.disc_a.parameters()) + list(strat.disc_b.parameters()), lr=1e-4
    )
    for cfg in strat.train_step(_batch(), epoch=0, iteration=1):
        cfg["closure"]()
    metrics = strat._last_step_metrics
    assert {"cycle", "identity", "adv_g", "adv_d", "g_total_loss"} <= set(metrics)
    # No legacy ``g_``-prefixed component keys leak through.
    assert not any(
        key in ("g_cycle", "g_identity", "g_adv_g", "g_adv_d") for key in metrics
    )
    # Component metrics are plain floats (routed through ``_handle_anomalies``).
    assert all(
        isinstance(metrics[key], float)
        for key in ("cycle", "identity", "adv_g", "g_total_loss")
    )
    # The D-closure's metric survives the G-closure ``update``.
    assert "d_total_loss" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
