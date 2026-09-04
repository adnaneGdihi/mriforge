"""A warm-up gate defers a loss. It must never delete one.

``resolve_loss_weight`` answers "what is this loss's weight RIGHT NOW", folding in
the warm-up gate: a term in ``LEGACY_WARMUP_LOSSES`` resolves to 0.0 for the
first ``warmup_iterations`` (default **1000**) steps. Correct for scaling a loss.
Wrong for two other questions the codebase was asking it:

1. "Is this loss configured?" -- asked at CONSTRUCTION time, where there is no
   iteration, so the answer is always the iteration-0 answer. ``l1`` is
   warm-up-gated, so it read as "not requested" and
   ``UnifiedDiffusionLossComputer`` never built a reconstruction loss AT ALL --
   permanently, not merely during warm-up.

2. "What weight should this step use?" -- asked per step but WITHOUT passing
   ``iteration``, so the gate saw step 0 forever and the term never switched on.
   That is the same defect fixed for ``unified_gan`` in #1649.

Both fail silently: the run trains, the loss decreases, and a term the config
asked for is simply absent.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _config():
    from spectramr.config.settings import TrainingSettings

    return TrainingSettings.settings_from_dict(
        {
            "model": {"model_type": "unet"},
            "data": {"dataset_type": "synthetic"},
            "optimization": {},
            "logging": {},
            "losses": {
                "output_domain": "image",
                "image_losses": [{"name": "l1", "weight": 1.0}],
            },
        }
    )


def test_l1_is_warmup_gated_so_the_weight_form_cannot_answer_configuration():
    """The premise. If this ever stops holding, the tests below lose their point."""
    from spectramr.models.losses.weights import LEGACY_WARMUP_LOSSES

    config = _config()
    assert "l1" in LEGACY_WARMUP_LOSSES
    assert config.losses.reconstruction.warmup_iterations > 0


def test_is_loss_configured_ignores_the_warmup_gate():
    from spectramr.models.losses.computers.unified_diffusion_reconstruction import (
        UnifiedDiffusionLossComputer,
    )
    from spectramr.models.losses.weights import is_loss_configured

    computer = UnifiedDiffusionLossComputer(_config(), torch.device("cpu"))
    table = computer._weight_table

    # Declared with a non-zero static weight -> configured, whatever the step.
    assert is_loss_configured(table, "l1") is True
    # ...while the WEIGHT form still (correctly) defers it during warm-up.
    assert computer._get_loss_weight("l1", 0, 0) == 0.0
    assert computer._get_loss_weight("l1", 0, 2000) > 0.0

    # Not declared anywhere -> not configured.
    assert is_loss_configured(table, "complex_l1") is False


def test_a_declared_but_disabled_loss_is_not_configured():
    """``enabled: false`` is a real "no", unlike the warm-up gate's "not yet".

    The distinction this helper exists to draw cuts both ways: it must ignore the
    temporal gate WITHOUT also ignoring a deliberate opt-out.
    """
    from spectramr.config.settings import TrainingSettings
    from spectramr.models.losses.computers.unified_diffusion_reconstruction import (
        UnifiedDiffusionLossComputer,
    )
    from spectramr.models.losses.weights import is_loss_configured

    config = TrainingSettings.settings_from_dict(
        {
            "model": {"model_type": "unet"},
            "data": {"dataset_type": "synthetic"},
            "optimization": {},
            "logging": {},
            "losses": {
                "output_domain": "image",
                "image_losses": [{"name": "l1", "weight": 1.0, "enabled": False}],
            },
        }
    )
    computer = UnifiedDiffusionLossComputer(config, torch.device("cpu"))
    assert is_loss_configured(computer._weight_table, "l1") is False


def test_the_reconstruction_loss_is_actually_constructed():
    """THE regression: an `l1` arm got no reconstruction loss object at all.

    Asserting on the weight would have passed against the bug -- the weight was
    correctly 0.0 during warm-up. Only the constructed object distinguishes
    "deferred" from "never built".
    """
    from spectramr.models.losses.computers.unified_diffusion_reconstruction import (
        UnifiedDiffusionLossComputer,
    )

    computer = UnifiedDiffusionLossComputer(_config(), torch.device("cpu"))
    assert computer.reconstruction_loss_fn is not None, (
        "the config asked for l1 and no reconstruction loss was built — the "
        "warm-up gate was consulted as a configuration probe"
    )
    assert computer._recon_fallback_name == "l1"


@pytest.mark.parametrize(
    "module_name",
    [
        "unified_vae",
        "unified_diffusion_reconstruction",
        "unified_disentangled",
        "base",
    ],
)
def test_every_per_step_weight_lookup_threads_iteration(module_name):
    """A per-step lookup that drops ``iteration`` pins the gate at step 0 forever.

    Source-level because the failure is invisible at runtime: the term simply
    never contributes, and nothing reports a term that is silently zero.
    """
    import inspect
    import re

    module = __import__(f"spectramr.models.losses.computers.{module_name}", fromlist=["x"])
    source = inspect.getsource(module)
    offenders = [
        line.strip()
        for line in source.splitlines()
        # calls with an `epoch` argument but no `iteration` — the per-step form
        if re.search(r"_get_loss_weight\([^)]*epoch[^)]*\)", line) and "iteration" not in line
    ]
    assert not offenders, (
        f"{module_name}: per-step weight lookup(s) drop `iteration`, so every "
        f"warm-up-gated loss stays at weight 0.0 forever:\n  " + "\n  ".join(offenders)
    )
