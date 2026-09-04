"""Tests for :mod:`spectramr.data.transforms.qmap_ulf_operator_transform`.

The load-bearing test here is :func:`test_the_forward_operator_actually_runs`:
non-negotiable 16 is satisfied when the production path *resolves* the operator
and the firing has been **observed**, not when a class exists that could call
it. Issue #1708 reports precisely the unobserved state.
"""

from __future__ import annotations

import pytest
import torch
import torchio as tio

pytestmark = pytest.mark.unit

H = W = 16
D = 2


def _subject(t1_ms: float = 1200.0, **overrides) -> tio.Subject:
    """A quantitative subject shaped like QuantitativeDataset's output.

    Keys and layout mirror ``quantitative_dataset.py:275-296``: one
    ``tio.ScalarImage`` per declared map, named, in ``[C, X, Y, Z]``.
    """
    keys = {
        "pd": torch.rand(1, H, W, D),
        "t1": torch.full((1, H, W, D), t1_ms),
        "t2": torch.full((1, H, W, D), 80.0),
    }
    keys.update(overrides)
    return tio.Subject(**{k: tio.ScalarImage(tensor=v) for k, v in keys.items() if v is not None})


def _build(**kwargs):
    from spectramr.data.transforms.registry import build_transform

    kwargs.setdefault("b0_source", 3.0)
    return build_transform("simulate_ulf_from_qmaps", **kwargs)


def test_registered_under_a_name_that_states_its_difference() -> None:
    """Both ULF transforms are registered; the names must not be confusable.

    ``simulate_ulf_from_hf`` degrades an image, this one renders from maps
    (non-negotiable 17: the difference is stated at registration).
    """
    from spectramr.data.transforms.registry import get_transform

    reg = get_transform("simulate_ulf_from_qmaps")
    assert reg.cls.__name__ == "SimulateULFFromQMaps"
    assert reg.requires == ("t1", "t2", "pd")
    # The sibling is still reachable and still distinct.
    assert get_transform("simulate_ulf_from_hf").cls.__name__ == "SimulateULFFromHF"


def test_the_forward_operator_actually_runs(monkeypatch) -> None:
    """Spy proof that ``DifferentiableULFForwardOperator.forward`` executed.

    This is the #1708 evidence. A test that only asserts the output shape would
    pass against a transform that quietly rendered something else, which is the
    failure mode the issue describes -- so the assertion is on the *call*, and
    on the argument units at the seam.
    """
    from spectramr.infrastructure.physics import ulf_forward_operator as ufo

    calls: list[dict] = []
    real_forward = ufo.DifferentiableULFForwardOperator.forward

    def spy(self, rho, t1_3t, t2_3t, *args, **kwargs):
        calls.append(
            {
                "b0_target": self.b0_target,
                "b0_source": self.b0_source,
                "t1_median": t1_3t.median().item(),
                "shape": tuple(rho.shape),
                "gnl": kwargs.get("gnl_coefficients"),
                "trajectory": kwargs.get("trajectory"),
                "enable_gnl": self.enable_gnl,
                "enable_maxwell": self.enable_maxwell,
            }
        )
        return real_forward(self, rho, t1_3t, t2_3t, *args, **kwargs)

    monkeypatch.setattr(ufo.DifferentiableULFForwardOperator, "forward", spy)

    out = _build(b0_target=0.064, render_target=True)(_subject())

    # Two renders: the ULF one and the paired source-field one.
    assert len(calls) == 2, f"expected ULF + paired HF render, got {len(calls)}"
    assert {c["b0_target"] for c in calls} == {0.064, 3.0}
    assert out["input"].data.shape == (1, H, W, D)
    assert out["target"].data.shape == (1, H, W, D)

    for c in calls:
        # Units at the seam: T1 must arrive in ms (order 10^3), not seconds.
        assert 20.0 <= c["t1_median"] <= 20_000.0, c["t1_median"]
        # The depth axis became the batch, so the operator sees [D, 1, H, W].
        assert c["shape"] == (D, 1, H, W)
        # Tier 1: co-registered volumes must not be re-distorted.
        assert c["gnl"] is None and c["enable_gnl"] is False
        assert c["enable_maxwell"] is False
        # Cartesian branch, not NUFFT.
        assert c["trajectory"] is None


def test_render_target_false_writes_no_target() -> None:
    out = _build(render_target=False)(_subject())
    assert "input" in out
    assert "target" not in out


@pytest.mark.parametrize("missing", ["pd", "t1", "t2"])
def test_missing_map_raises_naming_it(missing: str) -> None:
    """A subject without the maps must raise, not render something plausible.

    ``TransformRegistration.requires`` is declarative only -- no production
    code reads it -- so this explicit check is the enforcement.
    """
    subject = _subject(**{missing: None})
    with pytest.raises(ValueError, match=missing):
        _build()(subject)


@pytest.mark.parametrize(
    ("t1_value", "why"),
    [(1.2, "seconds instead of milliseconds"), (0.5, "normalized to unit range")],
)
def test_non_millisecond_t1_raises(t1_value: float, why: str) -> None:
    """The 1000x hazard: a seconds- or unit-scaled T1 renders plausibly and wrong.

    ``quantitative_config.units='normalized'`` rescales every map to [0, 1],
    and nothing downstream would notice -- the exponentials just saturate.
    """
    with pytest.raises(ValueError, match="milliseconds"):
        _build()(_subject(t1_ms=t1_value))


def test_zero_or_negative_field_raises() -> None:
    with pytest.raises(ValueError, match="b0_source"):
        _build(b0_source=0.0)
    with pytest.raises(ValueError, match="b0_target"):
        _build(b0_target=-1.0)


def test_b0_source_has_no_default() -> None:
    """The maps' reference field must be declared, never assumed to be 3 T.

    The operator names its arguments ``t1_3t``/``t2_3t``; nothing says a
    quantitative corpus was measured at 3 T, and a wrong reference
    mis-transports every voxel silently.
    """
    import inspect

    from spectramr.data.transforms.qmap_ulf_operator_transform import SimulateULFFromQMaps

    param = inspect.signature(SimulateULFFromQMaps.__init__).parameters["b0_source"]
    assert param.default is inspect.Parameter.empty


def test_operator_is_cached_per_geometry() -> None:
    """One operator per (field, H, W) -- not one per subject.

    Rebuilding it per item would allocate the noise and SH buffers in every
    dataloader worker call.
    """
    transform = _build(render_target=False)
    transform(_subject())
    transform(_subject())
    assert len(transform._operators) == 1


# ---------------------------------------------------------------------------
# The production path: YAML -> TransformSpecSchema -> builder -> tio.Compose.
#
# The spy test above proves the transform calls the operator. It does NOT prove
# the *production* path reaches the transform -- it constructs the transform
# directly, which no production code does. #1708 is about the second gap, so
# these drive the real builder function instead.
# ---------------------------------------------------------------------------


def _transform_config(**kwargs):
    """A ``TorchIOTransformConfig`` declaring this transform, as YAML would."""
    from spectramr.data.builders.torchio_transform_builder import TorchIOTransformConfig

    kwargs.setdefault("extra_transforms", [("simulate_ulf_from_qmaps", {"b0_source": 3.0})])
    return TorchIOTransformConfig(**kwargs)


def _names(compose) -> list[str]:
    return [type(t).__name__ for t in compose.transforms]


def test_the_production_builder_appends_it_to_both_chains() -> None:
    """``_append_registry_transforms`` is the real read site; drive it.

    Applying a declared transform to train only would grade the model on data
    the transform never touched, so BOTH chains are asserted (the docstring at
    ``torchio_transform_builder.py:1509-1512`` states this as the contract).
    """
    from spectramr.data.builders.torchio_transform_builder import TorchIOTransformBuilder

    config = _transform_config()
    train = TorchIOTransformBuilder.build_train_transforms(config)
    val = TorchIOTransformBuilder.build_val_transforms(config)

    assert "SimulateULFFromQMaps" in _names(train)
    assert "SimulateULFFromQMaps" in _names(val)


def test_the_built_transform_carries_the_declared_kwargs() -> None:
    """The kwargs survive the builder, rather than the default being rebuilt.

    ``b0_target`` has a default (0.064), so asserting it alone cannot tell a
    threaded kwarg from a defaulted one. Declare a non-default value.
    """
    from spectramr.data.builders.torchio_transform_builder import TorchIOTransformBuilder

    config = _transform_config(
        extra_transforms=[("simulate_ulf_from_qmaps", {"b0_source": 7.0, "b0_target": 0.1})]
    )
    built = [
        t
        for t in TorchIOTransformBuilder.build_train_transforms(config).transforms
        if type(t).__name__ == "SimulateULFFromQMaps"
    ]
    assert len(built) == 1
    assert built[0].b0_source == 7.0
    assert built[0].b0_target == 0.1


def test_the_yaml_spec_resolves_through_the_same_two_calls_the_builder_makes() -> None:
    """``TransformSpecSchema`` is the YAML object; the read site calls exactly
    ``resolved_kwargs()`` then ``get_transform(name)``
    (``torchio_transform_builder.py:708-709, 725``). Pin both.
    """
    from spectramr.config.schemas.data import TransformSpecSchema
    from spectramr.data.transforms.registry import get_transform

    spec = TransformSpecSchema(
        name="simulate_ulf_from_qmaps", kwargs={"b0_source": 3.0, "b0_target": 0.064}
    )
    assert spec.resolved_kwargs() == {"b0_source": 3.0, "b0_target": 0.064}
    assert get_transform(spec.name).cls.__name__ == "SimulateULFFromQMaps"


def test_an_unknown_kwarg_raises_rather_than_being_ignored() -> None:
    """Non-negotiable 8: an advertised knob is read, and an unread one raises.

    The registry constructs with ``cls(**kwargs)``, so an unknown YAML key is a
    ``TypeError`` at build time rather than a silently dropped setting.
    """
    with pytest.raises(TypeError):
        _build(b0_targett=0.064)  # transposed letter, the realistic typo


def test_an_unregistered_name_raises_listing_the_registered_ones() -> None:
    """Non-negotiable 3: an unknown name must raise, never degrade to a default."""
    from spectramr.data.transforms.registry import get_transform

    with pytest.raises(KeyError):
        get_transform("simulate_ulf_from_qmapz")


def test_declared_image_normalization_runs_before_the_render_and_is_overwritten() -> None:
    """The ordering finding, pinned so it cannot regress silently.

    ``ImageNormalizationTransform`` is appended at ``:1226`` (train) / ``:1456``
    (val); registry transforms at ``:1276`` / ``:1485``. Its whitelist is
    ``IMAGE_KEYS = {"input", "target", "image", "mri", "gt"}``
    (``normalization.py:1181``), which does NOT include ``t1``/``t2``/``pd`` --
    so the quantitative maps reach this transform un-normalized (good, the unit
    guard stays meaningful), but any normalization of ``input``/``target`` is
    then **discarded**, because the render overwrites both keys.

    An arm therefore declares ``normalization_type: none``; declaring anything
    else advertises a knob that does nothing for the keys it names.
    """
    from spectramr.data.builders.torchio_transform_builder import TorchIOTransformBuilder

    config = _transform_config(normalization_type="standard")
    for compose in (
        TorchIOTransformBuilder.build_train_transforms(config),
        TorchIOTransformBuilder.build_val_transforms(config),
    ):
        names = _names(compose)
        assert "ImageNormalizationTransform" in names, names
        assert names.index("ImageNormalizationTransform") < names.index("SimulateULFFromQMaps"), (
            "normalization must precede the render for this finding to hold"
        )

    # And the render really does replace both keys it writes.
    out = _build(render_target=True)(_subject())
    assert set(out.get_images_names()) >= {"input", "target"}
