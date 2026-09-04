"""Regression tests for the federated cross-contrast validation slice.

The 2026-05-13 mosaic audit traced the experiment_11 "doubled target" PNGs
to the M4Raw cross-contrast dataloader stacking
``[source_kspace, target_kspace]`` along the channel axis. The validation
save then RSS-combined ALL 16 channels (8 source + 8 target) into one
magnitude image, mixing T1 and T2/FLAIR anatomy in a single render —
the visual "doubled brain" the user reported.

Fix scaffold:
  1. M4RawRepetitionDataset publishes ``federated_target_channel_start``
     on the subject (the channel index where the target half begins).
  2. The diffusion strategy validation logger accepts that boundary as
     ``federated_target_channel_start`` kwarg, slices both prediction
     and target tensors to ``[:, start:, ...]``, and only then runs
     ``kspace_to_image`` — so the RSS only mixes coils within one
     contrast.

The two grep-pins below read the source text; the rest exercise
:meth:`DiffusionTrainingStrategy._resolve_federated_target_start` directly.
"""

from __future__ import annotations

import pathlib

import pytest
import torch

from spectramr.infrastructure.training.strategies.diffusion import (
    DiffusionTrainingStrategy,
)

# The grep-pins read files by path, so anchor on the repo root rather than the
# process cwd — tests/unit/infrastructure/training/strategies/ -> parents[5].
REPO_ROOT = pathlib.Path(__file__).resolve().parents[5]


# ─── Grep-pins: the dataset must publish the marker, the strategy must
# forward it. A refactor that drops either side is the regression we
# want to catch in CI before the bug reaches a smoke run.


def test_m4raw_dataset_publishes_federated_split_metadata():
    """Cross-contrast M4Raw must mark the (subject) with the channel boundary.

    Without this metadata, the strategy can't recover the split and the
    visualization would RSS T1 + T2 anatomy into a single mixed render.
    """
    src = (REPO_ROOT / "src/spectramr/data/datasets/m4raw_dataset.py").read_text(
        encoding="utf-8"
    )
    assert 'subject["federated_target_channel_start"]' in src, (
        "M4RawRepetitionDataset must publish ``federated_target_channel_start`` "
        "on the cross-contrast subject so the validation save can slice the "
        "target-contrast half only."
    )


def test_diffusion_strategy_forwards_federated_marker():
    """The diffusion strategy must extract and forward the marker.

    A refactor that calls ``_log_validation_images_to_tensorboard`` without
    forwarding the marker reintroduces the doubled-target bug for
    federated cross-contrast configs.
    """
    src = (
        REPO_ROOT / "src/spectramr/infrastructure/training/strategies/diffusion.py"
    ).read_text(encoding="utf-8")
    assert "federated_target_channel_start" in src, (
        "diffusion.py must read federated_target_channel_start from the batch "
        "and forward it into _log_validation_images_to_tensorboard so the "
        "validation save can slice the target half."
    )
    assert "_resolve_federated_target_start" in src, (
        "diffusion.py must validate/normalize the marker via "
        "_resolve_federated_target_start before slicing — otherwise a "
        "malformed marker silently corrupts the visualization."
    )


# ─── Direct unit tests of the marker-validation helper.
#
# These used to load diffusion.py via ``spec_from_file_location`` under the
# synthetic name ``_diffusion_strategy_for_test``. That gives the module no
# package, so every relative import inside it raised "attempted relative import
# with no known parent package" — and the blanket ``except Exception:
# pytest.skip`` turned a broken loader into seven green-looking skips, so the
# helper below went untested for as long as the loader was broken. The module
# imports fine by its real name (two sibling test files already do it), which
# also makes the "transitive imports are heavy" rationale moot.


@pytest.fixture(scope="module")
def Strategy():
    return DiffusionTrainingStrategy


def test_resolve_federated_target_start_returns_none_when_marker_absent(Strategy):
    """``None`` marker → render full stack (historical behaviour preserved)."""
    preds = torch.zeros(2, 16, 8, 8)
    tgts = torch.zeros(2, 16, 8, 8)
    assert Strategy._resolve_federated_target_start(None, preds, tgts) is None


def test_resolve_federated_target_start_accepts_int(Strategy):
    """A scalar int marker (test-fixture style) is honoured."""
    preds = torch.zeros(1, 16, 8, 8)
    tgts = torch.zeros(1, 16, 8, 8)
    assert Strategy._resolve_federated_target_start(8, preds, tgts) == 8


def test_resolve_federated_target_start_accepts_0d_tensor(Strategy):
    """The dataset publishes the marker as a 0-d long tensor."""
    preds = torch.zeros(1, 16, 8, 8)
    tgts = torch.zeros(1, 16, 8, 8)
    marker = torch.tensor(8, dtype=torch.long)
    assert Strategy._resolve_federated_target_start(marker, preds, tgts) == 8


def test_resolve_federated_target_start_accepts_stacked_uniform_batch(Strategy):
    """All entries in a stacked batch agree → return the common value."""
    preds = torch.zeros(4, 16, 8, 8)
    tgts = torch.zeros(4, 16, 8, 8)
    marker = torch.tensor([8, 8, 8, 8], dtype=torch.long)
    assert Strategy._resolve_federated_target_start(marker, preds, tgts) == 8


def test_resolve_federated_target_start_returns_none_on_mixed_batch(Strategy):
    """Mixed federated / single-contrast batch can't be sliced uniformly.

    Returning ``None`` falls back to the historical full-stack render —
    safer than silently slicing only some samples.
    """
    preds = torch.zeros(2, 16, 8, 8)
    tgts = torch.zeros(2, 16, 8, 8)
    marker = torch.tensor([8, 0], dtype=torch.long)
    assert Strategy._resolve_federated_target_start(marker, preds, tgts) is None


def test_resolve_federated_target_start_rejects_out_of_bounds(Strategy):
    """A stale marker pointing past tensor shape is treated as absent."""
    preds = torch.zeros(1, 16, 8, 8)
    tgts = torch.zeros(1, 16, 8, 8)
    assert Strategy._resolve_federated_target_start(99, preds, tgts) is None
    assert Strategy._resolve_federated_target_start(0, preds, tgts) is None
    assert Strategy._resolve_federated_target_start(-1, preds, tgts) is None


def test_resolve_federated_target_start_rejects_shape_mismatch(Strategy):
    """If predictions and targets disagree on channel count, the marker is
    only honoured when the boundary fits BOTH — otherwise None."""
    preds = torch.zeros(1, 16, 8, 8)
    tgts = torch.zeros(1, 4, 8, 8)  # smaller, e.g. magnitude target
    assert Strategy._resolve_federated_target_start(8, preds, tgts) is None
