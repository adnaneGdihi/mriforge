"""Phase 4d — OutputWriter (multi-channel + frame-ordered inference outputs)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from mriforge.config.schemas.data import (
    DataConfigSchema,
    ModeOutputSchema,
    MultiChannelOutputSchema,
    OutputChannelSchema,
    TemporalOutputSchema,
)
from mriforge.data.writers.output_writer import OutputWriter, write_output

# ── Schema validation ───────────────────────────────────────────────────────


def test_mode_output_defaults_are_disabled() -> None:
    cfg = ModeOutputSchema()
    assert cfg.format == "nifti"
    assert cfg.multi_channel.enabled is False
    assert cfg.temporal.enabled is False


def test_multi_channel_enabled_requires_channels() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="channels"):
        MultiChannelOutputSchema(enabled=True, channels=[])


def test_writer_format_enum_is_closed() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ModeOutputSchema(format="zip")  # type: ignore[arg-type]


def test_output_channel_dtype_enum_is_closed() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        OutputChannelSchema(name="t1", dtype="float64")  # type: ignore[arg-type]


def test_data_config_threads_infer_output_block() -> None:
    """ModeConfigSchema.output forward ref must resolve via model_rebuild."""
    cfg = DataConfigSchema(
        modes={
            "infer": {
                "output": {
                    "format": "npy",
                    "multi_channel": {
                        "enabled": True,
                        "channels": [{"name": "t1"}, {"name": "t2"}],
                    },
                }
            }
        }
    )
    assert cfg.modes.infer.output is not None
    assert cfg.modes.infer.output.format == "npy"
    assert cfg.modes.infer.output.multi_channel.channels[0].name == "t1"


# ── Single-tensor (legacy) write ────────────────────────────────────────────


def test_writer_npy_single_tensor(tmp_path: Path) -> None:
    cfg = ModeOutputSchema(format="npy")
    writer = OutputWriter(cfg, tmp_path)
    pred = torch.randn(1, 4, 4, 4)
    paths = writer.write(tensor=pred, subject_id="sub-001")
    assert len(paths) == 1
    assert paths[0].exists()
    assert paths[0].suffix == ".npy"


def test_writer_default_filename_template_uses_subject_id(tmp_path: Path) -> None:
    cfg = ModeOutputSchema(format="npy")
    writer = OutputWriter(cfg, tmp_path)
    paths = writer.write(tensor=torch.ones(1, 2, 2, 2), subject_id="sub-042")
    assert paths[0].name == "sub-042_pred.npy"


def test_writer_rejects_low_dim_tensor(tmp_path: Path) -> None:
    cfg = ModeOutputSchema(format="npy")
    writer = OutputWriter(cfg, tmp_path)
    with pytest.raises(ValueError, match="at least 3D"):
        writer.write(tensor=torch.zeros(2, 2), subject_id="sub-x")


# ── Multi-channel write (Phase 4b qMRI use case) ────────────────────────────


def test_writer_multi_channel_splits_into_per_map_files(tmp_path: Path) -> None:
    cfg = ModeOutputSchema(
        format="npy",
        multi_channel=MultiChannelOutputSchema(
            enabled=True,
            channels=[
                OutputChannelSchema(name="t1", scale=1000.0),  # ms scale
                OutputChannelSchema(name="t2", scale=100.0),
                OutputChannelSchema(name="pd"),  # already [0,1]
            ],
        ),
    )
    writer = OutputWriter(cfg, tmp_path)
    pred = torch.ones(3, 2, 2, 2)
    paths = writer.write(tensor=pred, subject_id="sub-001")
    names = sorted(p.name for p in paths)
    assert names == ["sub-001_pd.npy", "sub-001_t1.npy", "sub-001_t2.npy"]


def test_writer_multi_channel_applies_per_channel_scale(tmp_path: Path) -> None:
    """T1 scale=1000 ⇒ on-disk value should be input * 1000."""
    cfg = ModeOutputSchema(
        format="npy",
        multi_channel=MultiChannelOutputSchema(
            enabled=True,
            channels=[OutputChannelSchema(name="t1", scale=1000.0, offset=10.0)],
        ),
    )
    writer = OutputWriter(cfg, tmp_path)
    pred = torch.full((1, 2, 2, 2), 0.5)  # → 0.5 * 1000 + 10 = 510
    paths = writer.write(tensor=pred, subject_id="sub-001")
    loaded = np.load(paths[0])
    assert np.allclose(loaded, 510.0)


def test_writer_multi_channel_count_mismatch_raises(tmp_path: Path) -> None:
    """Channels declared = 2, tensor channels = 3 ⇒ ValueError."""
    cfg = ModeOutputSchema(
        format="npy",
        multi_channel=MultiChannelOutputSchema(
            enabled=True,
            channels=[OutputChannelSchema(name="t1"), OutputChannelSchema(name="t2")],
        ),
    )
    writer = OutputWriter(cfg, tmp_path)
    with pytest.raises(ValueError, match="channels"):
        writer.write(tensor=torch.zeros(3, 2, 2, 2), subject_id="x")


def test_writer_int16_dtype_clips_after_scale(tmp_path: Path) -> None:
    """int16 dtype ⇒ values clip to [-32768, 32767]."""
    cfg = ModeOutputSchema(
        format="npy",
        multi_channel=MultiChannelOutputSchema(
            enabled=True,
            channels=[OutputChannelSchema(name="t1", scale=100000.0, dtype="int16")],
        ),
    )
    writer = OutputWriter(cfg, tmp_path)
    pred = torch.ones(1, 2, 2, 2)  # 1 * 100000 → clipped to 32767
    paths = writer.write(tensor=pred, subject_id="sub-001")
    loaded = np.load(paths[0])
    assert loaded.dtype == np.int16
    assert loaded.max() == 32767


# ── Frame-ordered cine write (Phase 4c) ─────────────────────────────────────


def test_writer_temporal_preserves_identity_frame_order(tmp_path: Path) -> None:
    """frame_order = [0,1,2,3] ⇒ no permutation."""
    cfg = ModeOutputSchema(
        format="npy",
        temporal=TemporalOutputSchema(enabled=True, frame_axis_out=0),
    )
    writer = OutputWriter(cfg, tmp_path)
    # 4-frame volume, distinct per frame.
    pred = torch.stack([torch.full((2, 2, 1), float(f)) for f in range(4)], dim=0)
    paths = writer.write(
        tensor=pred,
        subject_id="cine-001",
        frame_order=[0, 1, 2, 3],
    )
    loaded = np.load(paths[0])
    # On-disk frame f should equal f.
    for f in range(4):
        assert np.all(loaded[f] == float(f))


def test_writer_temporal_undoes_shuffled_frame_order(tmp_path: Path) -> None:
    """frame_order = [2, 0, 3, 1] ⇒ writer restores natural order before write."""
    cfg = ModeOutputSchema(
        format="npy",
        temporal=TemporalOutputSchema(enabled=True, frame_axis_out=0),
    )
    writer = OutputWriter(cfg, tmp_path)
    # Frames carry values 0..3 in their input order, but the subject was
    # loaded with frame_order [2, 0, 3, 1] — i.e., pred[0] is original frame 2.
    pred = torch.stack([torch.full((2, 2, 1), float(f)) for f in range(4)], dim=0)
    paths = writer.write(
        tensor=pred,
        subject_id="cine-002",
        frame_order=[2, 0, 3, 1],
    )
    loaded = np.load(paths[0])
    # After restore, on-disk[f] = pred[inverse[f]]:
    #   inverse: original frame 2 lived at position 0 → on-disk[2] = pred[0] = 0.0
    #            original frame 0 lived at position 1 → on-disk[0] = pred[1] = 1.0
    #            original frame 3 lived at position 2 → on-disk[3] = pred[2] = 2.0
    #            original frame 1 lived at position 3 → on-disk[1] = pred[3] = 3.0
    expected = np.array([1.0, 3.0, 0.0, 2.0])
    assert np.allclose(loaded[:, 0, 0, 0], expected)


@pytest.mark.parametrize("frame_axis_out", [0, 1, 2, 3])
def test_writer_applies_frame_axis_out_without_a_frame_order(
    tmp_path: Path, frame_axis_out: int
) -> None:
    """Axis placement needs only the config -- it must not need ``frame_order``.

    This is the production call shape: ``pipelines/infer.py::_save_output`` is
    file-driven and passes no ``frame_order`` at all. While the permute sat
    inside the ``frame_order is not None`` guard, a cine arm declaring
    ``frame_axis_out: 3`` silently got its frames written on axis 0.

    The three pre-existing temporal tests missed it twice over: each passes a
    ``frame_order``, and each pins ``frame_axis_out=0``, the one value that
    makes the permute an identity.
    """
    cfg = ModeOutputSchema(
        format="npy",
        temporal=TemporalOutputSchema(enabled=True, frame_axis_out=frame_axis_out),
    )
    writer = OutputWriter(cfg, tmp_path)
    pred = torch.zeros(6, 2, 3, 4)  # (frames, X, Y, Z)

    paths = writer.write(tensor=pred, subject_id=f"cine-ax{frame_axis_out}")

    loaded = np.load(paths[0])
    assert (
        loaded.shape[frame_axis_out] == 6
    ), f"frames should land on axis {frame_axis_out}; got shape {loaded.shape}"
    expected = [2, 3, 4]
    expected.insert(frame_axis_out, 6)
    assert loaded.shape == tuple(expected)


def test_writer_preserve_frame_order_false_skips_the_restore(tmp_path: Path) -> None:
    """The knob is read. Its documented ``false`` branch had no code behind it.

    ``docs/data_pipeline_reference.rst`` described this branch while the gate
    read only ``temporal.enabled`` (pitfall #15).
    """
    pred = torch.stack([torch.full((2, 2, 1), float(f)) for f in range(4)], dim=0)
    shuffled = [2, 0, 3, 1]

    def _write(preserve: bool) -> np.ndarray:
        cfg = ModeOutputSchema(
            format="npy",
            temporal=TemporalOutputSchema(
                enabled=True, frame_axis_out=0, preserve_frame_order=preserve
            ),
        )
        writer = OutputWriter(cfg, tmp_path)
        paths = writer.write(
            tensor=pred,
            subject_id=f"cine-preserve-{preserve}",
            frame_order=shuffled,
        )
        return np.load(paths[0])[:, 0, 0, 0]

    assert np.allclose(_write(True), [1.0, 3.0, 0.0, 2.0])  # restored
    assert np.allclose(_write(False), [0.0, 1.0, 2.0, 3.0])  # untouched


def test_writer_temporal_disabled_skips_reordering(tmp_path: Path) -> None:
    """temporal.enabled=False ⇒ frame_order ignored."""
    cfg = ModeOutputSchema(format="npy")  # temporal disabled by default
    writer = OutputWriter(cfg, tmp_path)
    pred = torch.stack([torch.full((2, 2, 1), float(f)) for f in range(4)], dim=0)
    paths = writer.write(
        tensor=pred,
        subject_id="cine-003",
        frame_order=[2, 0, 3, 1],  # ignored
    )
    loaded = np.load(paths[0])
    # On-disk frame 0 still has value 0 (no permute applied).
    assert loaded[0, 0, 0, 0] == 0.0


# ── Format dispatch ─────────────────────────────────────────────────────────


def test_writer_h5_format(tmp_path: Path) -> None:
    try:
        import h5py  # noqa: F401
    except ImportError:
        pytest.skip("h5py not installed")
    cfg = ModeOutputSchema(format="h5")
    writer = OutputWriter(cfg, tmp_path)
    paths = writer.write(tensor=torch.ones(1, 2, 2, 2), subject_id="x")
    assert paths[0].suffix == ".h5"


def test_writer_nifti_format(tmp_path: Path) -> None:
    try:
        import nibabel  # noqa: F401
    except ImportError:
        pytest.skip("nibabel not installed")
    cfg = ModeOutputSchema(format="nifti")
    writer = OutputWriter(cfg, tmp_path)
    paths = writer.write(tensor=torch.ones(1, 2, 2, 2), subject_id="x")
    assert paths[0].name.endswith(".nii.gz")


# ── Functional wrapper ──────────────────────────────────────────────────────


def test_write_output_functional_wrapper(tmp_path: Path) -> None:
    cfg = ModeOutputSchema(format="npy")
    paths = write_output(
        tensor=torch.zeros(1, 2, 2, 2),
        config=cfg,
        output_dir=tmp_path,
        subject_id="sub-z",
    )
    assert len(paths) == 1
    assert paths[0].exists()


def test_writer_filename_template_supports_idx_placeholder(tmp_path: Path) -> None:
    cfg = ModeOutputSchema(
        format="npy",
        filename_template="{subject_id}_idx{idx}_{name}",
    )
    writer = OutputWriter(cfg, tmp_path)
    paths = writer.write(tensor=torch.zeros(1, 2, 2, 2), subject_id="sub-z", idx=42)
    assert paths[0].name == "sub-z_idx42_pred.npy"
