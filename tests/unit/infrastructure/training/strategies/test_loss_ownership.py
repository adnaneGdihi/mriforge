"""Loss ownership declared on the strategy class is pinned against the source (mrixfields review 2026-09-03).

A flag can lie: ``folds_image_losses`` says whether the other declared image
losses reach the objective, and the source is the only witness of that. Every
cohort strategy that overrides ``_compute_losses_impl`` either calls the parent
(``super()._compute_losses_impl``) or the fold (``_apply_builder_image_losses``)
-- then it folds -- or neither -- then it must say False. The same set of
strategies computes its inline L1 weight from the loss-weight table, one owner.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from spectramr.infrastructure.training.strategies.loss_folding import (
    declared_folds_image_losses,
    declared_inline_losses,
)
from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

COHORT_MODES = (
    "doob_bridge",
    "confluence",
    "scattering_besov",
    "steerable_synthesis",
    "brenier_synthesis",
    "cross_field_translation",
    "bloch_synth",
    "ulf_redegrad_tta",
    "ulf_map",
    "ulf_dps",
    "recoverability_vib",
    "field_cold_diffusion",
    "cartoon_texture_safe",
    "monotone_field",
    "generative_refiner",
    "field_conditioned_inr",
    "heteroscedastic_ulf",
    "field_fno",
    "bloch_field",
    "field_wiener",
    "koopman_field",
    "field_flow",
    "field_bridge",
    "fisher_rao_geodesic",
    "field_guided_diffusion",
    "lora_modulation",
    "mccann_field_path",
    "field_cocycle",
)
L1_TABLE_READERS = (
    "scattering_besov",
    "steerable_synthesis",
    "brenier_synthesis",
    "ulf_redegrad_tta",
    "cartoon_texture_safe",
    "monotone_field",
    "field_conditioned_inr",
    "field_fno",
    "bloch_field",
    "field_wiener",
    "koopman_field",
    "fisher_rao_geodesic",
    "lora_modulation",
    "mccann_field_path",
    "recoverability_vib",
)


def _cls(mode: str) -> type:
    path = TrainingStrategyFactory.STRATEGY_CLASS_PATHS[mode]
    module, name = path.rsplit(".", 1)
    return getattr(__import__(module, fromlist=[name]), name)


@pytest.mark.parametrize("mode", COHORT_MODES)
def test_every_cohort_strategy_declares_its_ownership(mode: str) -> None:
    cls = _cls(mode)
    assert declared_inline_losses(cls) is not None, f"{cls.__name__} declares no inline_losses"
    assert declared_folds_image_losses(cls) is not None, f"{cls.__name__} declares no folds flag"


@pytest.mark.parametrize("mode", COHORT_MODES)
def test_the_folds_flag_agrees_with_the_source(mode: str) -> None:
    """Planted-violation shape: flip any strategy's flag and this goes red."""
    cls = _cls(mode)
    src = inspect.getsource(inspect.getmodule(cls))
    if "_compute_losses_impl" not in cls.__dict__ and "fold_builder_image_losses(" not in src:
        # Inherits the parent's loss hook (ulf_map via PnPStrategy): the nearest
        # declaration is the parent's, and the parent's module decides.
        parent = next(k for k in cls.__mro__[1:] if "_compute_losses_impl" in k.__dict__)
        src = inspect.getsource(inspect.getmodule(parent))
    reaches = (
        "super()._compute_losses_impl" in src
        or "_apply_builder_image_losses(" in src
        or "fold_builder_image_losses(" in src
    )
    assert declared_folds_image_losses(cls) is reaches, (
        f"{cls.__name__}.folds_image_losses={declared_folds_image_losses(cls)} but its "
        f"_compute_losses_impl {'calls' if reaches else 'never calls'} the parent or the fold"
    )


@pytest.mark.parametrize("mode", COHORT_MODES)
def test_an_inline_l1_declaration_matches_an_inline_l1_computation(mode: str) -> None:
    """``l1`` is inline iff the strategy's module computes an L1 itself."""
    cls = _cls(mode)
    module_src = inspect.getsource(inspect.getmodule(cls))
    computes_l1 = "l1_loss(" in module_src or "F.l1" in module_src or ".abs().mean()" in module_src
    declared = declared_inline_losses(cls)
    assert declared is not None
    if cls.__name__ == "PnPStrategy" or mode == "ulf_map":
        return  # the parent path computes every declared entry; nothing is inline
    if mode == "field_bridge":
        # Its endpoint anchor ``mean(abs(x_hat - x))`` is an L1 gated by
        # ``lambda_endpoint_l1`` (default 0) and weighted by that knob, not by an
        # image_losses entry, so ``l1`` is deliberately NOT declared inline.
        assert "l1" not in declared
        return
    assert ("l1" in declared) is computes_l1, (cls.__name__, sorted(declared), computes_l1)


@pytest.mark.parametrize("mode", L1_TABLE_READERS)
def test_the_inline_l1_weight_is_read_from_the_table_not_a_training_block(mode: str) -> None:
    """One owner: no reader of ``training.<mode>.lambda_l1`` survives."""
    cls = _cls(mode)
    module_src = inspect.getsource(inspect.getmodule(cls))
    assert 'getattr(cfg, "lambda_l1"' not in module_src and '_g("lambda_l1"' not in module_src
    assert "_declared_inline_l1_weight()" in module_src


def test_the_weight_read_raises_without_an_l1_entry() -> None:
    cls = _cls("brenier_synthesis")
    s = cls.__new__(cls)
    s.config = SimpleNamespace(
        losses=SimpleNamespace(image_losses=[], kspace_losses=[], complex_losses=[])
    )
    with pytest.raises(ValueError, match="losses.image_losses"):
        s._declared_inline_l1_weight()


def test_the_retired_training_block_weights_raise_at_load() -> None:
    """The 15 mode schemas lost ``lambda_l1`` (``lambda_recon`` on the VIB); a rename record
    names the owner instead of pydantic's bare 'extra forbidden'."""
    from spectramr.config.schemas.renames import RENAMES

    retired = {k for k in RENAMES if k.endswith(".lambda_l1") or k.endswith(".lambda_recon")}
    assert len(retired) == 15 and all(RENAMES[k].posture == "raise" for k in retired)
    assert "training.brenier_synthesis.lambda_l1" in retired
    assert "training.recoverability_vib.lambda_recon" in retired
