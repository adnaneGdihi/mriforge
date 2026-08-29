"""``GeneratorBuilder`` must tell author-declared kwargs from forwarded ones.

``build()`` deliberately forwards every top-level ``model.*`` field the
constructor *might* accept, relying on the factory's signature filter to drop
what does not fit. That best-effort forward and an arm's own
``model.model_kwargs`` are indistinguishable by the time they reach the factory
-- so the builder captures the author-declared subset **before** the forward and
passes it down as ``_declared_keys``.

Without that split, the ``**kwargs`` recording added for issue #878 fires once
per forwarded schema field per arm, burying the ``model_kwargs`` entries that
are the actual signal.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mriforge.infrastructure.builders.leaf.model_builders import GeneratorBuilder


def _config(**model_fields):
    """A stand-in config whose ``model`` block exposes ``model_fields``.

    Deliberately a pydantic-shaped stub rather than a bare SimpleNamespace: the
    builder iterates ``type(model_cfg).model_fields``, so a stub without that
    attribute would make the forward loop silently no-op and the test would pass
    for the wrong reason.
    """
    from pydantic import BaseModel

    class _Model(BaseModel):
        model_config = {"extra": "allow"}
        spatial_dims: int | None = None
        base_channels: int | None = None

    return SimpleNamespace(
        model=_Model(**model_fields),
        optimization=SimpleNamespace(
            gradient=SimpleNamespace(enable_checkpointing=False)
        ),
    )


@pytest.fixture
def captured_call():
    """Capture the kwargs `create_generator` is called with."""
    seen = {}

    def _create_generator(**kwargs):
        seen.update(kwargs)
        return MagicMock()

    factory = MagicMock()
    factory.create_generator.side_effect = _create_generator
    with patch(
        "mriforge.models.factories.model_factory.ModelFactory", return_value=factory
    ):
        yield seen


def test_declared_keys_holds_only_author_written_model_kwargs(captured_call):
    config = _config(spatial_dims=3)
    builder = (
        GeneratorBuilder(config, device="cpu")
        .with_architecture("unet")
        .with_input_channels(2)
        .with_output_channels(2)
        .with_parameter("depth", 4)
    )
    builder.validate().build()

    assert captured_call["_declared_keys"] == frozenset({"depth"}), (
        "only `depth` was author-written; `spatial_dims` was forwarded from the "
        "top-level model block on a best-effort basis"
    )


def test_forwarded_field_still_reaches_the_factory(captured_call):
    """The split must not change WHAT is forwarded, only what is attributed."""
    config = _config(spatial_dims=3)
    builder = (
        GeneratorBuilder(config, device="cpu")
        .with_architecture("unet")
        .with_input_channels(2)
        .with_output_channels(2)
        .with_parameter("depth", 4)
    )
    builder.validate().build()

    assert captured_call["spatial_dims"] == 3
    assert captured_call["depth"] == 4


def test_in_and_out_channels_are_not_counted_as_declared(captured_call):
    """They are stripped from kwargs to avoid a duplicate-keyword TypeError, so
    they must not appear in the declared set either."""
    config = _config()
    builder = (
        GeneratorBuilder(config, device="cpu")
        .with_architecture("unet")
        .with_input_channels(2)
        .with_output_channels(2)
        .with_parameter("in_channels", 99)
    )
    builder.validate().build()

    assert "in_channels" not in captured_call["_declared_keys"]
