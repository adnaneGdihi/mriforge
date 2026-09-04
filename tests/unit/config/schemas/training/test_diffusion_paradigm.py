"""`training.diffusion` is a paradigm SELECTOR, and these pin that it selects.

The block used to be one `extra="allow"` class: a typo became a live untyped
attribute that downstream `getattr` read, and `type` was a free string nothing
branched on. It is now a discriminated union, which makes the union itself the
factory -- an unknown tag raises at parse with every valid tag named, satisfying
non-negotiable #3 without a single `if/elif`.

These tests exist because each property below has a plausible way of silently
degrading: a normaliser that coerces instead of raising, a variant that loses its
inherited validators, an enum that drifts from the union it documents.
"""

from __future__ import annotations

import pytest

from spectramr.config.schemas.enums import (
    DIFFUSION_PARADIGM_ALIASES,
    DiffusionParadigm,
)
from spectramr.config.schemas.training.base import (
    ChiSquareParams,
    ColdParams,
    DDIMParams,
    DDPMParams,
    DiffusionParadigmParams,
    LatentDiffusionParams,
    RectifiedFlowParams,
    ScoreBasedParams,
    TrainingStrategyConfigSchema,
    UnspecifiedParams,
)


def _block(**kwargs: object) -> object:
    return TrainingStrategyConfigSchema(diffusion=kwargs).diffusion


class TestTheUnionSelects:
    @pytest.mark.parametrize(
        "tag,expected",
        [
            ("cold", ColdParams),
            ("ddpm", DDPMParams),
            ("score_based", ScoreBasedParams),
            ("rectified_flow", RectifiedFlowParams),
            ("latent_diffusion", LatentDiffusionParams),
            ("ddim", DDIMParams),
            ("chi_square", ChiSquareParams),
        ],
    )
    def test_each_tag_selects_its_variant(self, tag: str, expected: type) -> None:
        assert isinstance(_block(type=tag), expected)

    def test_an_unknown_tag_raises_and_names_the_valid_set(self) -> None:
        """Non-negotiable #3: an unregistered value raises, never degrades."""
        with pytest.raises(Exception) as exc:
            _block(type="definitely_not_a_paradigm")
        message = str(exc.value)
        assert "definitely_not_a_paradigm" in message
        assert "cold" in message, "the error must name what IS accepted"

    def test_the_normaliser_does_not_swallow_a_bad_tag(self) -> None:
        """The normaliser injects a default when the tag is ABSENT.

        If it also defaulted an unrecognised tag it would reintroduce the exact
        silent fallback this union removes -- a typo would train happily as
        `UnspecifiedParams`.
        """
        with pytest.raises(Exception):
            _block(type="colf")  # one keystroke from `cold`


class TestTheUntaggedCase:
    def test_an_absent_tag_selects_the_transitional_variant(self) -> None:
        """A discriminated union needs the tag PRESENT.

        Without the normaliser injecting it, the 75 untagged arms in the corpus
        raise "Unable to extract tag using discriminator 'type'" -- i.e. the
        union would make them unloadable rather than behaviour-neutral.
        """
        assert isinstance(_block(timesteps=500), UnspecifiedParams)

    def test_the_untagged_variant_stays_permissive(self) -> None:
        """Landing the union must not change what an untagged arm accepts."""
        block = _block(some_knob_no_schema_declares=1)
        assert block.model_extra == {"some_knob_no_schema_declares": 1}


class TestPerVariantStrictness:
    @pytest.mark.parametrize("tag", ["ddpm", "score_based", "rectified_flow", "ddim", "chi_square"])
    def test_measured_clean_variants_forbid_extras(self, tag: str) -> None:
        """Strict because nothing measured needs them permissive.

        The first three had ZERO undeclared keys across their corpus arms.
        `ddim` and `chi_square` have zero ARMS -- they exist as capability, not
        because anything uses them yet -- so there is no legacy YAML to
        accommodate and the first arm to adopt one gets typed validation
        immediately rather than inheriting a permissive block it never needed.
        """
        with pytest.raises(Exception, match="Extra inputs are not permitted"):
            _block(type=tag, a_key_no_schema_declares=1)

    @pytest.mark.parametrize("tag", ["cold", "latent_diffusion"])
    def test_variants_pinned_by_readerless_keys_stay_permissive(self, tag: str) -> None:
        """Transitional, and the reason is named rather than assumed.

        `clip_sample`, `clip_sample_range` and `dynamic_thresholding_func` are
        declared by live arms and read by NOTHING in `src/`. A typed field would
        advertise an unread knob (#8); a `raise` rename would make those arms
        unloadable; dropping them silently is the defect being removed. Until an
        owner decides, the variant admits it instead of pretending.

        When those keys get a home, this test should be inverted -- not deleted.
        """
        assert _block(type=tag, clip_sample=True).model_extra == {"clip_sample": True}


class TestInheritedMechanismsSurviveTheSplit:
    """A variant that loses its base's validators is silently less safe."""

    @pytest.mark.parametrize(
        "tag",
        [
            "cold",
            "ddpm",
            "score_based",
            "rectified_flow",
            "latent_diffusion",
            "ddim",
            "chi_square",
        ],
    )
    def test_the_nested_rename_fold_fires_on_every_variant(self, tag: str) -> None:
        """`fold_renamed_keys("training.diffusion")` is mounted on the BASE.

        The union dissolves the single class the mount used to sit on, so this
        asserts per variant rather than once -- otherwise `num_timesteps` would
        keep folding on `cold` and silently stop on the strict variants, where
        `extra="forbid"` would turn it into a hard error instead.
        """
        assert _block(type=tag, num_timesteps=250).timesteps == 250

    def test_every_variant_is_frozen(self) -> None:
        for variant in (
            UnspecifiedParams,
            ColdParams,
            DDPMParams,
            ScoreBasedParams,
            RectifiedFlowParams,
            LatentDiffusionParams,
            DDIMParams,
            ChiSquareParams,
        ):
            assert variant.model_config["frozen"] is True, variant.__name__


class TestTheEnumIsNotDocumentation:
    def test_the_enum_and_the_union_agree(self) -> None:
        """An enum nothing checks is a comment that can rot.

        `DiffusionParadigm` is the declared closed set; the union's Literal tags
        are what actually discriminates. This is what keeps them one fact.
        """
        import typing

        variants = typing.get_args(typing.get_args(DiffusionParadigmParams)[0])
        tags = set()
        for variant in variants:
            tags |= set(typing.get_args(variant.model_fields["type"].annotation))

        assert tags == {m.value for m in DiffusionParadigm} | {None}

    def test_every_alias_targets_a_real_member(self) -> None:
        members = {m.value for m in DiffusionParadigm}
        for alias, target in DIFFUSION_PARADIGM_ALIASES.items():
            assert target in members, f"{alias} -> {target} is not a paradigm"

    def test_aliases_normalise_to_their_target(self) -> None:
        for alias, target in DIFFUSION_PARADIGM_ALIASES.items():
            assert _block(type=alias).type == target
