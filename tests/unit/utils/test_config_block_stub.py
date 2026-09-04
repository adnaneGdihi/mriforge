"""``tests.utils.config_block_stub`` must track the real schemas, not restate them.

Mirrors ``test_data_config_stub.py`` for the ``validation`` / ``logging`` /
``losses`` / ``undersampling`` blocks. The properties under test are the ones
that make the stub worth having at all: it IS the real schema, and its
flat->canonical routing is DERIVED from ``RENAMES`` rather than hand-written.
A hand-written table is a second resolver, and a second resolver agreeing with
production is exactly how a stale stub keeps a dead reader green.
"""

from __future__ import annotations

import pytest

from tests.utils.config_block_stub import BLOCK_SCHEMAS, block_stub, flat_to_canonical


class TestTheStubIsTheRealSchema:
    @pytest.mark.parametrize("block", sorted(BLOCK_SCHEMAS))
    def test_every_registered_block_constructs_bare(self, block: str) -> None:
        obj = block_stub(block)
        assert type(obj).__name__ == BLOCK_SCHEMAS[block][1]

    def test_defaults_come_from_the_schema_not_from_here(self) -> None:
        """If this module restated a default it could drift; it must not."""
        from spectramr.config.schemas.validation import ValidationConfigSchema

        assert (
            block_stub("validation").schedule.interval_steps
            == ValidationConfigSchema().schedule.interval_steps
        )

    def test_an_unregistered_block_raises_and_names_the_known_ones(self) -> None:
        with pytest.raises(KeyError, match="no schema registered"):
            block_stub("not_a_block")


class TestRoutingIsDerivedFromRenames:
    def test_the_legacy_spelling_lands_where_the_reader_walks(self) -> None:
        # The exact pair that broke ~8 tests: production reads
        # `validation.scoring.compute`, stubs were setting `metrics`.
        assert block_stub("validation", metrics=["psnr"]).scoring.compute == ["psnr"]
        assert block_stub("logging", save_validation_images=True).images.save_validation
        assert block_stub("losses", output_domain="kspace").policy.output_domain == (
            "kspace"
        )

    def test_routing_agrees_with_the_rename_ssot(self) -> None:
        """Anti-drift: every route must BE a RENAMES record, not a lookalike."""
        from spectramr.config.schemas.renames import RENAMES

        for block in BLOCK_SCHEMAS:
            routes = flat_to_canonical(block)
            for leaf, path in routes.items():
                record = RENAMES.get(f"{block}.{leaf}")
                assert record is not None, f"{block}.{leaf} is not a RENAMES record"
                assert record.canonical == f"{block}.{'.'.join(path)}"

    def test_routing_is_non_empty(self) -> None:
        """Anti-vacuity: an empty map would make every routing test pass."""
        counts = {b: len(flat_to_canonical(b)) for b in BLOCK_SCHEMAS}
        assert counts["validation"] > 0 and counts["logging"] > 0, counts

    def test_every_route_target_exists_on_the_schema(self) -> None:
        """A route naming a path the schema lost must fail here, not in a test."""
        for block in BLOCK_SCHEMAS:
            obj = block_stub(block)
            for leaf, path in flat_to_canonical(block).items():
                walked = obj
                for step in path[:-1]:
                    walked = getattr(walked, step, None)
                    assert walked is not None, f"{block}.{step} missing (from {leaf})"
                assert hasattr(
                    walked, path[-1]
                ), f"{block}.{'.'.join(path)} missing (from {leaf})"

    def test_a_route_deeper_than_one_level_still_lands(self) -> None:
        """The three `gradient.clip.*` records are the only >1-level routes.

        A depth cap dropped them into stray direct attributes while the reader
        walked `optimization.gradient.clip.value` and got the schema default --
        green, and pinning a value the test never chose.
        """
        stub = block_stub(
            "optimization", gradient_clip_value=7.7, enable_gradient_clipping=True
        )

        assert stub.gradient.clip.value == 7.7
        assert stub.gradient.clip.enabled is True
        assert not hasattr(stub, "gradient_clip_value"), (
            "the legacy spelling survived as a stray attribute, so the value "
            "landed somewhere no reader walks"
        )

    def test_routing_covers_every_same_block_rename(self) -> None:
        """No same-block fold record may be silently unroutable."""
        from spectramr.config.schemas.renames import RENAMES

        for block in BLOCK_SCHEMAS:
            prefix = f"{block}."
            routes = flat_to_canonical(block)
            for legacy, record in RENAMES.items():
                if not legacy.startswith(prefix) or "." in legacy[len(prefix) :]:
                    continue
                if not record.canonical.startswith(prefix):
                    continue  # cross-block moves have no home in a block stub
                assert legacy[len(prefix) :] in routes, (
                    f"{legacy} -> {record.canonical} is unroutable, so a test "
                    f"passing it would silently get the schema default"
                )


class TestDirectKwargs:
    def test_an_unrouted_kwarg_is_set_directly(self) -> None:
        assert block_stub("validation", made_up_knob=7).made_up_knob == 7

    def test_a_canonical_kwarg_is_not_double_routed(self) -> None:
        """Passing the canonical sub-block wholesale must win, not be ignored."""
        from spectramr.config.schemas.validation import ValidationConfigSchema

        scoring = ValidationConfigSchema().scoring.model_copy(
            update={"compute": ["ssim"]}
        )
        assert block_stub("validation", scoring=scoring).scoring.compute == ["ssim"]
