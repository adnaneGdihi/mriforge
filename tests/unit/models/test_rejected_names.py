"""Phase 4 regression — rejected aspirational names stay rejected.

``spectramr.models.registry.REJECTED_NAMES`` records the 24 deletion-ledger
names that have no implementation and were judged ill-specified. None may be
advertised in ``VALID_MODEL_TYPES`` or resolve in ``MODEL_REGISTRY`` — doing
so would re-create the audit-surface lie the deletion fixed.
See TODO/deleted_model_types_reimplementation_plan.md Phase 4.
"""

from __future__ import annotations

import pytest

from spectramr.config.validation_constants import VALID_MODEL_TYPES
from spectramr.models.registry import REJECTED_NAMES


@pytest.mark.registry_contract
def test_rejected_names_have_rationales() -> None:
    assert REJECTED_NAMES, "REJECTED_NAMES must not be empty"
    for name, reason in REJECTED_NAMES.items():
        assert isinstance(reason, str) and reason.strip(), (
            f"rejected name '{name}' is missing a rationale string"
        )


@pytest.mark.registry_contract
@pytest.mark.parametrize("name", sorted(REJECTED_NAMES))
def test_rejected_name_not_advertised(name: str) -> None:
    assert name not in set(VALID_MODEL_TYPES), (
        f"rejected name '{name}' must not appear in VALID_MODEL_TYPES"
    )


@pytest.mark.registry_contract
@pytest.mark.parametrize("name", sorted(REJECTED_NAMES))
def test_rejected_name_not_registered(name: str) -> None:
    pytest.importorskip("torch")
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY

    populate_model_registry()
    assert name not in MODEL_REGISTRY, (
        f"rejected name '{name}' must not resolve in MODEL_REGISTRY"
    )


@pytest.mark.registry_contract
def test_no_overlap_with_implemented_phase3() -> None:
    """Sanity: the 12 implemented Phase-3 models are NOT rejected."""
    implemented = {
        "glow", "equivariant_flow", "divergence_free_flow", "blurring_diffusion",
        "hierarchical_vae_ladder", "hierarchical_vq_vae", "hyperspherical_vae",
        "moe_vae", "beta_vae_gan", "progressive_gan", "octave_conv", "gated_gnn",
    }
    assert not (implemented & set(REJECTED_NAMES)), (
        "an implemented Phase-3 model is wrongly listed in REJECTED_NAMES"
    )
