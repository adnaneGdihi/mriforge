"""Tests for the loss-weight SSOT (``mriforge.models.losses.weights``).

These pin the behaviours that the eight legacy resolvers got wrong:
a schema default is not a declaration, aliases collapse, an undeclared loss raises
instead of materialising at 1.0, and a disagreeing dual declaration raises.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mriforge.config.schemas.loss import LossConfigSchema
from mriforge.domain.exceptions import ConfigurationError
from mriforge.models.losses.registry import LossRegistry
from mriforge.models.losses.weights import (
    LEGACY_WARMUP_LOSSES,
    WEIGHT_SEMANTICS_VERSION,
    build_loss_weight_table,
    canonical_loss_name,
    resolve_loss_weight,
)


class TestCanonicalName:
    def test_aliases_collapse_to_canonical(self):
        assert canonical_loss_name("mse") == "l2"
        assert canonical_loss_name("mae") == "l1"
        assert canonical_loss_name("L1") == "l1"

    def test_unregistered_name_passes_through(self):
        # Strategy-inline terms (e.g. lambda_pre_dc_kspace) have no registry entry.
        assert canonical_loss_name("pre_dc_kspace") == "pre_dc_kspace"

    def test_registry_exposes_the_public_seam(self):
        assert LossRegistry.canonical_name("mse") == "l2"


class TestDeclarationSurfaces:
    def test_list_weight_is_a_declaration(self):
        cfg = LossConfigSchema(
            output_domain="image", image_losses=[{"name": "hfen", "weight": 0.1}]
        )
        table = build_loss_weight_table(cfg)
        assert table.weight("hfen") == pytest.approx(0.1)
        assert table["hfen"].source == "losses.image_losses[hfen].weight"

    def test_explicit_lambda_is_a_declaration(self):
        cfg = LossConfigSchema(reconstruction={"lambda_hfen": 0.25})
        assert build_loss_weight_table(cfg).weight("hfen") == pytest.approx(0.25)

    def test_schema_default_does_not_override_an_explicit_declaration(self):
        """The bug that started this: lambda_hfen defaults to 0.0 and lambda_l1 to 10.0.

        The legacy resolvers read those defaults as declarations, which silently zeroed a
        declared `hfen weight: 0.1` and would 10x a declared `l1 weight: 1.0`. An explicit
        declaration must win over the schema default.
        """
        cfg = LossConfigSchema(
            output_domain="image",
            image_losses=[
                {"name": "hfen", "weight": 0.1},
                {"name": "l1", "weight": 1.0},
            ],
        )
        assert cfg.reconstruction.lambda_hfen == 0.0  # the default exists...
        assert "lambda_hfen" not in cfg.reconstruction.model_fields_set  # ...unwritten
        table = build_loss_weight_table(cfg)
        assert table.weight("hfen") == pytest.approx(0.1)  # not 0.0
        assert table.weight("l1", iteration=10**9) == pytest.approx(1.0)  # not 10.0

    def test_undeclared_term_with_a_schema_field_uses_that_visible_default(self):
        """Callers probe a term's weight and gate on `> 0`, so an undeclared term with a
        `lambda_<n>` field must answer with the schema default (0.0 = not requested)
        rather than exploding. That default is declared once, in the schema, and is
        auditable — it is not a magic table."""
        table = build_loss_weight_table(LossConfigSchema())
        assert table.weight("hfen") == 0.0
        assert table.weight("ssim") == 0.0

    def test_alias_across_surfaces_is_one_knob(self):
        """`image_losses: [mse]` and `lambda_l2` are the SAME loss — must not double-declare."""
        cfg = LossConfigSchema(
            output_domain="image",
            image_losses=[{"name": "mse", "weight": 0.5}],
            reconstruction={"lambda_l2": 0.5},
        )
        table = build_loss_weight_table(cfg)
        assert table.weight("mse") == pytest.approx(0.5)
        assert table.weight("l2") == pytest.approx(0.5)  # same entry, either spelling


class TestConflicts:
    def test_disagreeing_dual_declaration_raises(self):
        cfg = LossConfigSchema(
            output_domain="image",
            image_losses=[{"name": "hfen", "weight": 0.1}],
            reconstruction={"lambda_hfen": 0.9},
        )
        with pytest.raises(ConfigurationError, match="DIFFERENT weights"):
            build_loss_weight_table(cfg)

    def test_agreeing_dual_declaration_passes(self):
        """The flagship kspace-cold-diffusion YAML hand-syncs both surfaces."""
        cfg = LossConfigSchema(
            output_domain="image",
            image_losses=[{"name": "hfen", "weight": 0.1}],
            reconstruction={"lambda_hfen": 0.1},
        )
        assert build_loss_weight_table(cfg).weight("hfen") == pytest.approx(0.1)

    def test_disagreeing_alias_declaration_raises(self):
        cfg = LossConfigSchema(
            output_domain="image",
            image_losses=[{"name": "mse", "weight": 0.5}],
            reconstruction={"lambda_l2": 0.1},
        )
        with pytest.raises(ConfigurationError, match="DIFFERENT weights"):
            build_loss_weight_table(cfg)

    def test_conflict_error_lists_every_offender_at_once(self):
        cfg = LossConfigSchema(
            output_domain="image",
            image_losses=[
                {"name": "hfen", "weight": 0.1},
                {"name": "ssim", "weight": 0.2},
            ],
            reconstruction={"lambda_hfen": 0.9, "lambda_ssim": 0.8},
        )
        with pytest.raises(ConfigurationError) as exc:
            build_loss_weight_table(cfg)
        assert "hfen" in str(exc.value) and "ssim" in str(exc.value)


class TestResolution:
    @pytest.mark.parametrize("loss", ["pinn", "totally_made_up_loss"])
    def test_undeclared_loss_with_no_schema_field_raises(self, loss: str):
        """Pitfall #9. A term with no `lambda_<n>` field anywhere had no defensible
        weight: the legacy resolvers fell through to three disagreeing hardcoded tables
        (an undeclared `adversarial` resolved to 1.0 or 0.01 — 100x apart — purely by
        which computer the strategy happened to build). Refuse to invent one.

        The fix for the real terms was to give them a schema home rather than a table
        entry — see `test_probed_terms_now_have_a_visible_schema_home`."""
        table = build_loss_weight_table(LossConfigSchema())
        with pytest.raises(ConfigurationError, match="declared nowhere"):
            resolve_loss_weight(table, loss)

    @pytest.mark.parametrize(
        ("loss", "legacy_default"),
        [
            ("adversarial", 1.0),  # was losses.gan.lambda_adv -- an UNREAD knob
            ("reconstruction", 1.0),  # had no field at all
            ("codebook", 1.0),  # enable_codebook existed; the weight field did not
        ],
    )
    def test_probed_terms_now_have_a_visible_schema_home(
        self, loss: str, legacy_default: float
    ):
        """These are probed by the computers but had no `lambda_<n>` field, so they fell
        into the magic tables. Each now has a schema field carrying its legacy value —
        one visible, auditable home instead of three disagreeing ones (pitfall #15)."""
        table = build_loss_weight_table(LossConfigSchema())
        assert table.weight(loss, iteration=10**9) == pytest.approx(legacy_default)

    def test_lambda_adv_is_finally_read_as_the_adversarial_weight(self):
        """`losses.gan.lambda_adv` was an UNREAD knob: every resolver looked up
        `lambda_adversarial`, never found it, and used the fallback 1.0. One live arm
        declares 0.1 and has been training 10x hot."""
        cfg = LossConfigSchema(gan={"lambda_adv": 0.1})
        table = build_loss_weight_table(cfg)
        assert table.weight("adversarial", iteration=10**9) == pytest.approx(0.1)

    def test_kl_divergence_and_kl_are_one_knob(self):
        """A name-split bug the canonicalisation fixes: the strategy asked for
        'kl_divergence' (no schema field -> magic table -> 1e-4) while the computer asked
        for 'kl' (-> losses.latent.lambda_kl). Two names, one knob, two answers."""
        cfg = LossConfigSchema(latent={"lambda_kl": 0.05})
        table = build_loss_weight_table(cfg)
        assert table.weight("kl") == pytest.approx(0.05)
        assert table.weight("kl_divergence") == pytest.approx(0.05)

    def test_enabled_false_resolves_to_zero_and_does_not_raise(self):
        cfg = LossConfigSchema(
            output_domain="image",
            image_losses=[{"name": "hfen", "weight": 0.1, "enabled": False}],
        )
        table = build_loss_weight_table(cfg)
        assert table.weight("hfen") == 0.0  # declared-off != declared-nowhere

    def test_explicit_zero_weight_is_a_declaration(self):
        cfg = LossConfigSchema(reconstruction={"lambda_hfen": 0.0})
        assert build_loss_weight_table(cfg).weight("hfen") == 0.0

    def test_scheduled_override_supersedes_static_weight(self):
        cfg = LossConfigSchema(reconstruction={"lambda_hfen": 0.1})
        table = build_loss_weight_table(cfg)
        got = resolve_loss_weight(table, "hfen", scheduled={"hfen": 0.7})
        assert got == pytest.approx(0.7)

    def test_scheduled_override_beats_the_warmup_gate(self):
        """A curriculum rule must be able to enable a spatial term before warmup ends."""
        cfg = LossConfigSchema(
            reconstruction={"lambda_l1": 1.0, "warmup_iterations": 1000}
        )
        table = build_loss_weight_table(cfg)
        assert resolve_loss_weight(table, "l1", iteration=0) == 0.0  # gated
        assert resolve_loss_weight(
            table, "l1", scheduled={"l1": 2.0}, iteration=0
        ) == pytest.approx(2.0)


class TestWarmupGate:
    def test_legacy_warmup_set_is_the_default(self):
        cfg = LossConfigSchema(reconstruction={"lambda_l1": 1.0})
        table = build_loss_weight_table(cfg)
        assert table.warmup_losses == frozenset(
            canonical_loss_name(n) for n in LEGACY_WARMUP_LOSSES
        )
        assert table["l1"].warmup_gated is True

    def test_gate_zeroes_before_warmup_and_releases_after(self):
        cfg = LossConfigSchema(
            reconstruction={"lambda_l1": 3.0, "warmup_iterations": 100}
        )
        table = build_loss_weight_table(cfg)
        assert resolve_loss_weight(table, "l1", iteration=99) == 0.0
        assert resolve_loss_weight(table, "l1", iteration=100) == pytest.approx(3.0)

    def test_ungated_loss_is_never_zeroed(self):
        cfg = LossConfigSchema(
            reconstruction={"lambda_hfen": 3.0, "warmup_iterations": 100}
        )
        table = build_loss_weight_table(cfg)
        assert resolve_loss_weight(table, "hfen", iteration=0) == pytest.approx(3.0)

    def test_warmup_losses_is_configurable(self):
        """The set was a hardcoded literal duplicated across two resolvers."""
        cfg = LossConfigSchema(
            reconstruction={
                "lambda_l1": 3.0,
                "lambda_hfen": 2.0,
                "warmup_iterations": 100,
                "warmup_losses": ["hfen"],
            }
        )
        table = build_loss_weight_table(cfg)
        assert resolve_loss_weight(table, "hfen", iteration=0) == 0.0  # now gated
        assert resolve_loss_weight(table, "l1", iteration=0) == pytest.approx(3.0)

    def test_empty_warmup_losses_disables_the_gate(self):
        cfg = LossConfigSchema(
            reconstruction={
                "lambda_l1": 3.0,
                "warmup_iterations": 100,
                "warmup_losses": [],
            }
        )
        table = build_loss_weight_table(cfg)
        assert resolve_loss_weight(table, "l1", iteration=0) == pytest.approx(3.0)

    def test_warmup_losses_is_stamped_into_provenance(self):
        cfg = LossConfigSchema(
            reconstruction={"lambda_hfen": 1.0, "warmup_losses": ["hfen"]}
        )
        assert build_loss_weight_table(cfg).provenance()["warmup_losses"] == ["hfen"]


class TestComponentNameAliases:
    """The computers stack components under names the schema spells differently.

    Every one of these was a knob that no resolver could see, so it fell into the
    hardcoded default tables.
    """

    def test_r1_penalty_component_resolves_to_lambda_r1(self):
        """UnifiedGANLossComputer stacks R1 as `r1_penalty`; the field is `lambda_r1`."""
        cfg = LossConfigSchema(gan={"lambda_r1": 0.4})
        assert build_loss_weight_table(cfg).weight("r1_penalty") == pytest.approx(0.4)

    def test_a_field_spelled_query_is_the_same_knob(self):
        """Some callers thread the schema FIELD name straight through (the disentangled
        computer's {"hist": "lambda_hist"} map), so `lambda_hist` must not read as a
        distinct — and therefore undeclared — loss."""
        cfg = LossConfigSchema(reconstruction={"lambda_hist": 2.0})
        table = build_loss_weight_table(cfg)
        assert table.weight("hist") == pytest.approx(2.0)
        assert table.weight("lambda_hist") == pytest.approx(2.0)


class TestConfigDoubles:
    def test_a_plain_namespace_section_still_declares_its_lambdas(self):
        """A SimpleNamespace/mock has no `model_fields_set`. Treating it as "declares
        nothing" would silently zero every weight a test double set."""
        double = SimpleNamespace(
            reconstruction=SimpleNamespace(lambda_hfen=0.6, warmup_iterations=0)
        )
        assert build_loss_weight_table(double).weight("hfen") == pytest.approx(0.6)

    def test_a_non_int_warmup_iterations_does_not_crash(self):
        double = SimpleNamespace(
            reconstruction=SimpleNamespace(lambda_hfen=0.6, warmup_iterations=object())
        )
        table = build_loss_weight_table(double)
        assert table.warmup_iterations == 1000  # the documented fallback


class TestPinnSectionIsScanned:
    def test_pinn_lambdas_are_read(self):
        """48 arms declare losses.pinn.* — no legacy resolver ever scanned that section."""
        cfg = LossConfigSchema(pinn={"lambda_pde": 0.4})
        assert build_loss_weight_table(cfg).weight("pde") == pytest.approx(0.4)


class TestProvenance:
    def test_table_stamps_every_resolved_knob(self):
        cfg = LossConfigSchema(
            output_domain="image", image_losses=[{"name": "hfen", "weight": 0.1}]
        )
        stamp = build_loss_weight_table(cfg).provenance()
        assert stamp["semantics_version"] == WEIGHT_SEMANTICS_VERSION
        assert stamp["resolved"]["hfen"] == {
            "weight": 0.1,
            "enabled": True,
            "source": "losses.image_losses[hfen].weight",
            "warmup_gated": False,
        }

    def test_none_config_yields_empty_table(self):
        table = build_loss_weight_table(None)
        assert len(table) == 0
        # A term with no schema field has nothing to default from, so it still raises.
        with pytest.raises(ConfigurationError):
            table.weight("totally_made_up_loss")


# ---------------------------------------------------------------------------
# #421 -- two schema fields that canonicalise to ONE loss are a latent crash.
#
# `_declared_lambdas` guards with `model_fields_set` so a schema default is not
# a declaration. That guard is defeated by MATERIALISATION: `Model(**m.model_dump())`
# passes every key explicitly, so all ~138 reconstruction defaults become
# "declarations" and any two fields resolving to one canonical loss at two
# different defaults raise. This is not hypothetical -- `config/presets/cli.py`
# and `strategies/pipeline_strategy.py` both round-trip a complete dump today,
# and `config/overrides.py` carries the war story (SLURM 7796517, 2026-07-25)
# of the arms it killed before that module started dumping `exclude_unset=True`.
#
# Chasing materialisers is unbounded; removing the collisions is not. These
# tests hold the collision SET to a committed baseline by EQUALITY, so fixing
# one forces the baseline down and introducing one turns the suite red.
# ---------------------------------------------------------------------------


def _section_classes(schema=LossConfigSchema, sections=None):
    """Map each lambda-bearing section name to the schema class that owns it.

    A name in ``LAMBDA_SECTIONS`` that is not a field of ``schema`` is skipped,
    exactly as ``_declared_lambdas`` skips it. That silence is a defect in its
    own right, pinned by :class:`TestLambdaSectionsAreAllReal` below.
    """
    from mriforge.models.losses.weights import LAMBDA_SECTIONS

    out = {}
    for sec in LAMBDA_SECTIONS if sections is None else sections:
        field = schema.model_fields.get(sec)
        if field is None:
            continue
        annotation = field.annotation
        cls = next(
            (
                a
                for a in getattr(annotation, "__args__", [annotation])
                if hasattr(a, "model_fields")
            ),
            None,
        )
        if cls is not None:
            out[sec] = cls
    return out


def _lambda_collisions(section_classes):
    """``{canonical_loss: [(dotted_field, default), ...]}`` for names with >1 owner."""
    owners: dict[str, list[tuple[str, object]]] = {}
    for sec, cls in section_classes.items():
        for name, info in cls.model_fields.items():
            if name.startswith("lambda_"):
                owners.setdefault(canonical_loss_name(name), []).append(
                    (f"{sec}.{name}", info.default)
                )
    return {c: o for c, o in owners.items() if len(o) > 1}


def _split(collisions):
    """Partition collisions into (crashing, latent) by whether the defaults differ."""
    crashing, latent = set(), set()
    for canon, owners in collisions.items():
        (crashing if len({v for _, v in owners}) > 1 else latent).add(canon)
    return crashing, latent


#: Canonical losses still owned by two schema fields at DIFFERENT defaults, so a
#: materialised config raises whenever the partner section is declared. Each needs
#: an owner election; none is safe to guess, because two of them may instead be
#: over-aggressive registry aliasing (is `diffusion.lambda_mse` really the same
#: knob as `reconstruction.lambda_l2`? is `pinn.lambda_pde` really
#: `physics.lambda_helmholtz_pde`?). Corpus exposure measured 2026-08-23 over the
#: 647 `experiments/inprogress` arms: gan 24, diffusion 4, pinn 1, latent 0.
BASELINE_CRASHING = {"helmholtz_pde", "l1", "l2", "style"}

#: The SAME defect, dormant: two owners whose defaults coincide today, so
#: `_agree` is satisfied and nothing raises. Editing either default converts one
#: of these into a crash with no other code change, which is why the guard keys
#: on collision IDENTITY and not on the crash count.
BASELINE_LATENT = {"recon", "spectral_kspace", "ssim"}


class TestLambdaFieldCollisionBaseline:
    """The collision set is pinned by equality, in both directions."""

    def test_collision_set_matches_baseline_exactly(self):
        crashing, latent = _split(_lambda_collisions(_section_classes()))
        assert crashing == BASELINE_CRASHING, (
            "The set of canonical losses owned by two schema fields at different "
            "defaults changed. If you FIXED one, remove it from BASELINE_CRASHING. "
            "If you ADDED one, you have introduced a config that crashes on a "
            "materialised round-trip (#421) -- elect one owner and retire the "
            "loser through RENAMES (never a plain field deletion: every loss "
            "sub-schema is extra='ignore', so the key would vanish silently)."
        )
        assert latent == BASELINE_LATENT, (
            "The set of same-canonical field pairs whose defaults happen to agree "
            "changed. These do not raise today but become crashes the moment "
            "either default is edited (#421)."
        )

    def test_the_four_retired_collisions_are_gone(self):
        """#421's elections: these four no longer have two owners at all."""
        collisions = _lambda_collisions(_section_classes())
        for canon in ("perceptual", "bloch_residual", "physics_constraint", "snr_preserving"):
            assert canon not in collisions, (
                f"{canon!r} regained a second owner -- the #421 election was undone."
            )

    def test_materialised_round_trip_no_longer_raises(self):
        """Both legs of the original reproduction, with and without a physics block.

        Leg A always passed; leg B is the one that raised. The physics variant
        matters because `losses.physics` is declared by 57 of the 647 corpus arms,
        while `losses.reconstruction` is materialised on every single one -- which
        is why the content/perceptual pair crashed unconditionally and the three
        physics pairs only crashed for those 57.
        """
        for kwargs in ({}, {"physics": {"lambda_bloch_residual": 0.2}}):
            cfg = LossConfigSchema(**kwargs)
            build_loss_weight_table(cfg)  # leg A
            build_loss_weight_table(LossConfigSchema(**cfg.model_dump()))  # leg B


class TestCollisionDetectorPlants:
    """Planted violations, one per shape the rule can take (non-negotiable 15).

    The production baseline above is green, so on its own it proves nothing about
    whether the detector can SEE a collision. Each plant feeds `_lambda_collisions`
    a synthetic schema and asserts it goes red. Plants are synthetic rather than a
    live collision precisely so a future election cannot silently disarm them.
    """

    @staticmethod
    def _model(**fields):
        from pydantic import Field, create_model

        return create_model(
            "Planted",
            **{k: (float, Field(default=v)) for k, v in fields.items()},
        )

    def test_plant_cross_section_differing_defaults_is_caught(self):
        """The `reconstruction`/`physics` shape: two sections, one canonical name."""
        sections = {
            "reconstruction": self._model(lambda_planted=0.0),
            "physics": self._model(lambda_planted=1.0),
        }
        crashing, latent = _split(_lambda_collisions(sections))
        assert crashing == {"planted"} and latent == set()

    def test_plant_same_section_is_caught(self):
        """The `lambda_content`/`lambda_perceptual` shape -- BOTH owners in ONE class.

        This is the shape that made #421 fire on every arm, and the shape a
        cross-section-only detector would miss entirely.
        """
        sections = {"reconstruction": self._model(lambda_l1=1.0, lambda_mae=2.0)}
        crashing, _ = _split(_lambda_collisions(sections))
        assert crashing == {"l1"}, "alias `mae` -> `l1` inside one section was missed"

    def test_plant_alias_mediated_differing_field_names_is_caught(self):
        """Two DIFFERENT field names colliding only via the registry alias table.

        `lambda_l2` and `lambda_mse` share no substring; a detector comparing raw
        field names rather than canonical names reads them as two distinct losses.
        """
        sections = {
            "reconstruction": self._model(lambda_l2=0.0),
            "diffusion": self._model(lambda_mse=1.0),
        }
        crashing, _ = _split(_lambda_collisions(sections))
        assert crashing == {"l2"}

    def test_plant_latent_collision_promotes_to_crashing_on_a_default_edit(self):
        """A dormant collision must move buckets the instant a default diverges."""
        agree = {
            "reconstruction": self._model(lambda_planted=0.0),
            "latent": self._model(lambda_planted=0.0),
        }
        crashing, latent = _split(_lambda_collisions(agree))
        assert crashing == set() and latent == {"planted"}

        diverged = {
            "reconstruction": self._model(lambda_planted=0.0),
            "latent": self._model(lambda_planted=0.5),
        }
        crashing, latent = _split(_lambda_collisions(diverged))
        assert crashing == {"planted"} and latent == set()

    def test_plant_non_lambda_fields_are_not_collisions(self):
        """Negative control: the detector keys on the `lambda_` prefix only."""
        sections = {
            "reconstruction": self._model(enable_planted=0.0),
            "physics": self._model(enable_planted=1.0),
        }
        assert _lambda_collisions(sections) == {}


class TestLambdaSectionsAreAllReal:
    """Every `LAMBDA_SECTIONS` entry must name a real field of `LossConfigSchema`.

    `_declared_lambdas` does `getattr(loss_config, section_name, None)` and skips a
    miss silently, so a stale entry is a no-op on every call with no diagnostic.
    """

    def test_adversarial_is_the_only_known_dead_entry(self):
        from mriforge.models.losses.weights import LAMBDA_SECTIONS

        dead = [s for s in LAMBDA_SECTIONS if s not in LossConfigSchema.model_fields]
        assert dead == ["adversarial"], (
            "The set of dead LAMBDA_SECTIONS entries changed. `adversarial` is a "
            "known, separately-tracked no-op kept pending a history check (it is "
            "either stale or a section that was meant to exist -- prefer wiring "
            "over deletion). A NEW dead entry means a section's lambdas are being "
            "silently skipped by `_declared_lambdas`."
        )
