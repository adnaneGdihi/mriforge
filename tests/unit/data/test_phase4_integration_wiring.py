"""Integration tests pinning the Phase 4 transform / wrapper wiring.

These tests confirm that when a YAML enables a Phase 4 block, the
runtime actually composes the corresponding transform / wrapper into
the pipeline. Catches regressions where a schema is added but the
builder forgets to consume it (the original gap that prompted this audit).
"""
from __future__ import annotations

import pytest
import torch
import torchio as tio

from spectramr.config.schemas.data import DataConfigSchema
from spectramr.data.builders.torchio_transform_builder import (
    TorchIOTransformBuilder,
    TorchIOTransformConfig,
)


def _proxy(data_cfg, accel=None):
    """Wrap a bare `data:` block the way every production call site does.

    The attribute is `undersampling`, not `acceleration`: the top-level block
    was renamed (RENAMES, fold posture) and `from_training_config` reads
    `config.undersampling` (torchio_transform_builder.py:718). A YAML still
    DECLARING `acceleration:` loads, which is why this proxy kept the old
    spelling without anyone noticing -- a fold rescues the document, never a
    Python attribute read.

    It also defaults to a real `AccelerationConfigSchema()` rather than None,
    since the reader walks into the block; None is what produced the
    "'NoneType' object is not iterable" pair.
    """
    from spectramr.config.schemas.acceleration import AccelerationConfigSchema

    class _P:
        def __init__(self, d, a):
            self._d = d
            self.undersampling = a if a is not None else AccelerationConfigSchema()

        def __getattr__(self, n):
            return getattr(self._d, n)

    return _P(data_cfg, accel)


# ── Phase 4b: LoadAcquisitionMetadata in transform chain ─────────────────────


def test_acquisition_metadata_disabled_skips_loader() -> None:
    """Default YAML ⇒ no LoadAcquisitionMetadata in the chain."""
    cfg = DataConfigSchema()
    tcfg = TorchIOTransformConfig.from_training_config(_proxy(cfg))
    train_tfm = TorchIOTransformBuilder.build_train_transforms(tcfg)
    names = [t.__class__.__name__ for t in train_tfm.transforms]
    assert "LoadAcquisitionMetadata" not in names


def test_acquisition_metadata_enabled_appends_loader() -> None:
    """YAML with acquisition_metadata.enabled=true ⇒ transform is added."""
    cfg = DataConfigSchema(
        acquisition_metadata={
            "enabled": True,
            "source": "yaml_inline",
            "fields": ["TE", "TR"],
            "defaults": {"TE": 4.5, "TR": 8.0},
        }
    )
    tcfg = TorchIOTransformConfig.from_training_config(_proxy(cfg))
    train_tfm = TorchIOTransformBuilder.build_train_transforms(tcfg)
    names = [t.__class__.__name__ for t in train_tfm.transforms]
    assert "LoadAcquisitionMetadata" in names


def test_acquisition_metadata_in_val_transforms_too() -> None:
    """Metadata transform must fire on val too — inference parity."""
    cfg = DataConfigSchema(
        acquisition_metadata={
            "enabled": True,
            "source": "yaml_inline",
            "fields": ["TE"],
            "defaults": {"TE": 4.5},
        }
    )
    tcfg = TorchIOTransformConfig.from_training_config(_proxy(cfg))
    val_tfm = TorchIOTransformBuilder.build_val_transforms(tcfg)
    names = [t.__class__.__name__ for t in val_tfm.transforms]
    assert "LoadAcquisitionMetadata" in names


# ── Phase 4b-extra: multi-echo bundling ──────────────────────────────────────


def test_multi_echo_disabled_skips_bundle() -> None:
    cfg = DataConfigSchema()
    tcfg = TorchIOTransformConfig.from_training_config(_proxy(cfg))
    train_tfm = TorchIOTransformBuilder.build_train_transforms(tcfg)
    names = [t.__class__.__name__ for t in train_tfm.transforms]
    assert "BundleMultiEcho" not in names


def test_quantitative_enabled_appends_bundle_multi_echo() -> None:
    """quantitative.enabled=true ⇒ multi_echo bundler fires (validates TE list).

    The transform builder treats quantitative as the activation flag —
    when the user asks for parameter-map regression, multi-echo
    discipline is the only sensible default.
    """
    cfg = DataConfigSchema(
        quantitative={
            "enabled": True,
            "target_maps": ["t1", "t2"],
            "input_source": "contrasts",
        }
    )
    tcfg = TorchIOTransformConfig.from_training_config(_proxy(cfg))
    train_tfm = TorchIOTransformBuilder.build_train_transforms(tcfg)
    names = [t.__class__.__name__ for t in train_tfm.transforms]
    assert "BundleMultiEcho" in names


# ── Phase 4b-extra: DWI metadata ─────────────────────────────────────────────


def test_dwi_metadata_disabled_skips_loader() -> None:
    cfg = DataConfigSchema()
    tcfg = TorchIOTransformConfig.from_training_config(_proxy(cfg))
    train_tfm = TorchIOTransformBuilder.build_train_transforms(tcfg)
    names = [t.__class__.__name__ for t in train_tfm.transforms]
    assert "LoadDWIMetadata" not in names


def test_dwi_metadata_enabled_appends_loader() -> None:
    """When acquisition_metadata.fields includes b_value, LoadDWIMetadata fires."""
    cfg = DataConfigSchema(
        acquisition_metadata={
            "enabled": True,
            "source": "bids_json",
            "fields": ["TE", "b_value", "bvec"],
            "defaults": {},
        }
    )
    tcfg = TorchIOTransformConfig.from_training_config(_proxy(cfg))
    train_tfm = TorchIOTransformBuilder.build_train_transforms(tcfg)
    names = [t.__class__.__name__ for t in train_tfm.transforms]
    assert "LoadDWIMetadata" in names


# ── Phase 4f Amendment I: coordinate grid ────────────────────────────────────


def test_coord_emission_disabled_skips_transform() -> None:
    cfg = DataConfigSchema()
    tcfg = TorchIOTransformConfig.from_training_config(_proxy(cfg))
    train_tfm = TorchIOTransformBuilder.build_train_transforms(tcfg)
    names = [t.__class__.__name__ for t in train_tfm.transforms]
    assert "CoordinateGridTransform" not in names


def test_coord_emission_enabled_appends_transform() -> None:
    cfg = DataConfigSchema(
        emit_coordinates={"enabled": True, "normalize": True}
    )
    tcfg = TorchIOTransformConfig.from_training_config(_proxy(cfg))
    train_tfm = TorchIOTransformBuilder.build_train_transforms(tcfg)
    names = [t.__class__.__name__ for t in train_tfm.transforms]
    assert "CoordinateGridTransform" in names


def test_coord_emission_runs_after_spatial_ops() -> None:
    """CoordinateGridTransform must be the LAST entry (after final
    EnsureSpatialConsistency) so coord_resolution matches final shape."""
    cfg = DataConfigSchema(
        emit_coordinates={"enabled": True, "normalize": True}
    )
    tcfg = TorchIOTransformConfig.from_training_config(_proxy(cfg))
    train_tfm = TorchIOTransformBuilder.build_train_transforms(tcfg)
    names = [t.__class__.__name__ for t in train_tfm.transforms]
    # The chain terminates in _SyncSubjectAttributes (#1213), a bookkeeping no-op
    # that re-binds ``Subject.__dict__`` and changes no shape, so
    # ``coord_resolution`` still matches the final shape at names[-2].
    assert names[-1] == "_SyncSubjectAttributes"
    assert names[-2] == "CoordinateGridTransform"


# ── Phase 4f Amendment H + J: director-level wrappers ────────────────────────


def test_director_wraps_in_lazy_encode_when_enabled(
    tmp_path, monkeypatch
) -> None:
    """latent_diffusion.enabled=true ⇒ datasets are wrapped in LazyEncodeWrapper."""
    from spectramr.infrastructure.builders.directors.data_pipeline_director import (
        DataPipelineDirector,
    )
    from spectramr.data.builders.lazy_encode import LazyEncodeWrapper

    # Write a dummy checkpoint (a module, since LazyEncodeWrapper expects
    # torch.save(module, ...) shape by default).
    ckpt_path = tmp_path / "dummy_ckpt.pt"
    torch.save(torch.nn.Identity(), str(ckpt_path))
    data_cfg = DataConfigSchema(
        latent_diffusion={
            "enabled": True,
            "stage1_checkpoint": str(ckpt_path),
        }
    )

    # Toy inner datasets.
    class _ToyDs:
        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            return tio.Subject(input=tio.ScalarImage(tensor=torch.zeros(1, 2, 2, 2)))

    train_ds, val_ds = DataPipelineDirector._apply_optional_wrappers(
        _ToyDs(3),
        _ToyDs(2),
        data_cfg,
        device=torch.device("cpu"),
    )
    assert isinstance(train_ds, LazyEncodeWrapper)
    assert isinstance(val_ds, LazyEncodeWrapper)


def test_director_wraps_in_meta_learning_when_enabled() -> None:
    """meta_learning.enabled=true ⇒ train dataset wrapped in MetaLearningDataset."""
    from spectramr.infrastructure.builders.directors.data_pipeline_director import (
        DataPipelineDirector,
    )
    from spectramr.data.datasets.meta_learning_dataset import MetaLearningDataset

    data_cfg = DataConfigSchema(
        meta_learning={
            "enabled": True,
            "support_size": 2,
            "query_size": 3,
            "tasks_per_epoch": 4,
            "task_params": {"acceleration_factors": [2, 4]},
        }
    )

    from torch.utils.data import TensorDataset

    base_train = TensorDataset(torch.randn(8, 1, 4, 4))
    base_val = TensorDataset(torch.randn(4, 1, 4, 4))

    train_ds, val_ds = DataPipelineDirector._apply_optional_wrappers(
        base_train, base_val, data_cfg
    )
    assert isinstance(train_ds, MetaLearningDataset)
    # Val stays unwrapped — meta-tasks are training-only.
    assert val_ds is base_val


def test_director_default_is_noop() -> None:
    """No Phase 4f blocks enabled ⇒ datasets pass through unchanged."""
    from spectramr.infrastructure.builders.directors.data_pipeline_director import (
        DataPipelineDirector,
    )

    data_cfg = DataConfigSchema()  # all blocks disabled
    sentinel_train = object()
    sentinel_val = object()
    out_train, out_val = DataPipelineDirector._apply_optional_wrappers(
        sentinel_train, sentinel_val, data_cfg
    )
    assert out_train is sentinel_train
    assert out_val is sentinel_val


# ── Phase 4c: temporal sampler dispatch fails loud ───────────────────────────


def test_temporal_sampler_raises_when_dispatched_via_queue() -> None:
    """sampler.type=temporal_uniform via tio.Queue ⇒ ValueError (fail-loud)."""
    from spectramr.data.builders.torchio_queue_builder import (
        TorchIOQueueBuilder,
        TorchIOQueueConfig,
    )

    cfg = TorchIOQueueConfig(sampler_type="temporal_uniform")
    # Pass a None dataset — we only need to reach the sampler-dispatch
    # branch before failure.
    with pytest.raises(ValueError, match="temporal"):
        TorchIOQueueBuilder.build_train_queue(train_dataset=None, config=cfg)


def test_unknown_sampler_type_raises() -> None:
    """Unknown sampler.type ⇒ ValueError listing valid options."""
    from spectramr.data.builders.torchio_queue_builder import (
        TorchIOQueueBuilder,
        TorchIOQueueConfig,
    )

    cfg = TorchIOQueueConfig(sampler_type="bogus")
    with pytest.raises(ValueError, match="Unknown sampler"):
        TorchIOQueueBuilder.build_train_queue(train_dataset=None, config=cfg)
