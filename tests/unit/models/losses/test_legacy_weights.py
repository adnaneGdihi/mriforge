"""Pin the PRE-SSOT weight semantics that the migration must reproduce.

``_legacy_weights`` is migration scaffolding: it exists so the SSOT switch can be *proved*
a no-op (``tests/experiments/test_loss_weight_no_op_migration.py``). These tests pin the
three legacy behaviours that made the switch dangerous in the first place. If one of them
is wrong, the migration wrote the wrong number into 400 YAMLs.

Delete this file together with ``_legacy_weights.py`` once the tree is clean.
"""

from __future__ import annotations

import pytest

from mriforge.config.schemas.loss import LossConfigSchema
from mriforge.models.losses._legacy_weights import legacy_effective_weight


class _Root:
    def __init__(self, losses: LossConfigSchema) -> None:
        self.losses = losses
        self.objectives = None


def _root(**kw) -> _Root:
    return _Root(LossConfigSchema(**kw))


class TestSchemaDefaultsCountedAsDeclarations:
    """The 10x hazard: `lambda_l1` defaults to 10.0 and `model_dump()` cannot tell a
    default from a declaration, so 245 arms declaring `image_losses: [{l1, weight: 1.0}]`
    were training L1 at 10.0."""

    def test_declared_list_weight_of_one_actually_trained_at_ten(self):
        root = _root(
            output_domain="image", image_losses=[{"name": "l1", "weight": 1.0}]
        )
        assert legacy_effective_weight(root, "l1", family="computer") == 10.0
        assert legacy_effective_weight(root, "l1", family="recon") == 10.0

    def test_hfen_default_of_zero_meant_it_never_computed(self):
        """349 declarations across the corpus resolved to 0.0 and never fired."""
        root = _root(
            output_domain="image", image_losses=[{"name": "hfen", "weight": 0.1}]
        )
        assert legacy_effective_weight(root, "hfen", family="computer") == 0.0


class TestTheFamiliesDisagree:
    """The reason the migration needed a family classifier at all."""

    def test_recon_resolver_never_scanned_the_physics_section(self):
        """The flagship declares losses.physics.lambda_complex_spatial_gradient: 0.2 but
        `resolve_static_loss_weight` only walks reconstruction/spatial/evidential — so
        the 47 kspace_filling arms trained it at the 1.0 default, 5x hot."""
        root = _root(physics={"lambda_complex_spatial_gradient": 0.2})
        assert (
            legacy_effective_weight(root, "complex_spatial_gradient", family="computer")
            == 0.2
        )
        assert (
            legacy_effective_weight(root, "complex_spatial_gradient", family="recon")
            == 1.0
        )

    @pytest.mark.parametrize(
        ("loss", "computer", "recon"),
        [
            ("adversarial", 1.0, 0.01),  # 100x
            ("kl_divergence", 1.0, 0.0001),  # 10,000x
            ("pinn", 1.0, 0.1),
        ],
    )
    def test_the_default_tables_disagree_where_no_schema_field_exists(
        self, loss: str, computer: float, recon: float
    ):
        """The three magic tables only get consulted for losses with no `lambda_<n>`
        schema field -- and there they disagree wildly. An undeclared `kl_divergence`
        resolved to 1.0 or 1e-4 purely by which computer the strategy constructed.
        This is the concrete reason the SSOT raises instead of defaulting."""
        root = _root()
        assert legacy_effective_weight(root, loss, family="computer") == computer
        assert legacy_effective_weight(root, loss, family="recon") == recon


class TestFoldingFamily:
    def test_folding_read_the_list_weight_directly(self):
        root = _root(
            output_domain="image", image_losses=[{"name": "hfen", "weight": 0.25}]
        )
        assert legacy_effective_weight(root, "hfen", family="folding") == 0.25


def test_unknown_family_raises():
    with pytest.raises(ValueError, match="unknown legacy family"):
        legacy_effective_weight(_root(), "l1", family="nonsense")  # type: ignore[arg-type]
