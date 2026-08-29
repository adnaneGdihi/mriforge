"""Phase 2 regression — Category-B namespace tokens stay off model_type.

The deletion ledger purged 12 cross-axis tokens that had been misfiled as
``model_type``. Phase 2's disposition: each token lives on its CORRECT
config axis (training strategy / preset / embedding whitelist / training
mode), and NONE of them may reappear in ``VALID_MODEL_TYPES``. This test
pins that outcome so a future edit can't silently re-file them as models.
See TODO/deleted_model_types_reimplementation_plan.md Phase 2.
"""

from __future__ import annotations

import pytest

from mriforge.config.validation_constants import VALID_MODEL_TYPES

# Tokens that must NEVER appear under model.model_type again.
_FOREIGN_AXIS_TOKENS = [
    "gan",  # training_mode
    "2d_stack",  # dataset concept (covered by DatasetType.2d / .slice)
    "dicom_series",  # volume format (VolumeFormat.dicom), not a dataset type
    "adversarial_robustness",  # training strategy
    "test_time_adaptation_strategy",  # strategy key (ttt / test_time_adaptation)
    "implicit_neural_representation",  # private helper class
    "latent_unet",  # internal block
    "photonic_fft",  # physics op
    "latent_diffusion_sr",  # config preset
    "rdn",  # factory preset
    "sparse_hybrid_attention",  # block / embedding whitelist string
    "sparse_transformer",  # block / embedding whitelist string
    "vit_adapter",  # block / adapter whitelist string
]


@pytest.mark.registry_contract
@pytest.mark.parametrize("token", _FOREIGN_AXIS_TOKENS)
def test_foreign_token_not_a_model_type(token: str) -> None:
    assert token not in set(VALID_MODEL_TYPES), (
        f"'{token}' is a foreign-axis token (training_mode / dataset / "
        f"strategy / preset / block) and must not be advertised as a model_type"
    )


@pytest.mark.registry_contract
def test_strategy_tokens_resolve_on_strategy_axis() -> None:
    from mriforge.infrastructure.training.strategy_factory import (
        TrainingStrategyFactory,
    )

    keys = set(TrainingStrategyFactory.STRATEGY_CLASS_PATHS)
    assert "adversarial_robustness" in keys
    # test-time adaptation is registered under at least one canonical key
    assert {"ttt", "test_time_adaptation"} & keys
