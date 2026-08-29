"""Matched controls: the guard that turns a size effect into a lesion effect."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mriforge.data.annotations.fastmri_plus import LesionBox, YOrigin  # noqa: E402
from mriforge.data.annotations.fastmri_plus_classes import LesionGroup  # noqa: E402
from mriforge.data.annotations.matched_control import (  # noqa: E402
    sample_matched_control,
    sample_matched_controls,
)


def _brain(h: int = 128, w: int = 128, margin: int = 16) -> torch.Tensor:
    """A centred rectangular 'brain', symmetric about the column centroid."""
    m = torch.zeros(h, w, dtype=torch.bool)
    m[margin : h - margin, margin : w - margin] = True
    return m


def _box(x: int = 24, y: int = 40, w: int = 16, h: int = 16, **kw) -> LesionBox:
    defaults = {
        "file_id": "file_brain_001",
        "slice_index": 7,
        "x": x,
        "y_raw": y,
        "width": w,
        "height": h,
        "label": "Edema",
        "group": LesionGroup.EDEMA,
        "y_origin": YOrigin.TOP_LEFT,
    }
    return LesionBox(**{**defaults, **kw})


class TestSizeMatching:
    def test_the_control_is_exactly_the_lesion_size(self) -> None:
        """The whole point. A control of a different size would reintroduce the size
        confound it exists to remove (pitfall #17)."""
        box = _box(w=16, h=24)
        brain = _brain()
        control, why = sample_matched_control(
            box,
            brain=brain,
            lesion_union=box.to_bool_mask(128, 128),
            global_seed=0,
        )
        assert why == "ok"
        assert control is not None
        assert (control.width, control.height) == (16, 24)
        assert int(control.to_bool_mask(128, 128).sum()) == int(box.to_bool_mask(128, 128).sum())


class TestContralateralMirror:
    def test_the_mirror_lands_on_the_other_side_of_the_centroid(self) -> None:
        box = _box(x=24, y=40, w=16, h=16)  # left of a brain centred at col ~63.5
        brain = _brain()
        control, _ = sample_matched_control(
            box, brain=brain, lesion_union=box.to_bool_mask(128, 128), global_seed=0
        )
        assert control is not None
        assert control.strategy == "contralateral_mirror"
        # Mirrored about the brain's column centroid: box spans [24, 40) -> [87, 103).
        assert control.x == 87
        assert control.y == box.y_raw  # rows are untouched by a left-right mirror

    def test_the_control_stays_inside_the_brain(self) -> None:
        box = _box()
        brain = _brain()
        control, _ = sample_matched_control(
            box, brain=brain, lesion_union=box.to_bool_mask(128, 128), global_seed=0
        )
        assert control is not None
        assert control.brain_coverage >= 0.95


class TestTheControlAvoidsEveryLesion:
    def test_a_mirror_landing_on_another_lesion_is_rejected(self) -> None:
        """The control must avoid ALL lesions on the slice, not just its own -- a
        'control' sitting on a second lesion is not a control."""
        left = _box(x=24, y=40)
        right = _box(x=87, y=40)  # exactly where left's mirror would land
        brain = _brain()
        union = left.to_bool_mask(128, 128) | right.to_bool_mask(128, 128)

        control, why = sample_matched_control(left, brain=brain, lesion_union=union, global_seed=0)
        assert why == "ok"
        assert control is not None
        # The mirror was blocked, so it fell back to rejection sampling...
        assert control.strategy == "rejection_sample"
        # ...and the result overlaps neither lesion.
        assert not bool((control.to_bool_mask(128, 128) & union).any())

    def test_the_clearance_cordon_keeps_the_control_off_lesion_edges(self) -> None:
        """A control abutting a lesion picks up its edge gradient -- and the
        artifact's effect on that edge is precisely what the lesion ROI measures."""
        box = _box(x=24, y=40, w=16, h=16)
        brain = _brain()
        union = box.to_bool_mask(128, 128)
        control, _ = sample_matched_control(
            box,
            brain=brain,
            lesion_union=union,
            global_seed=3,
            lesion_clearance_px=2,
        )
        assert control is not None
        grown = (
            torch.nn.functional.max_pool2d(
                union[None, None].float(), kernel_size=5, stride=1, padding=2
            )[0, 0]
            > 0
        )
        assert not bool((control.to_bool_mask(128, 128) & grown).any())


class TestDeterminism:
    def test_the_same_lesion_always_gets_the_same_control(self) -> None:
        """The seed is a pure function of WHAT is sampled -- never of iteration
        order or worker count. An enumerate()-derived seed would hand you a
        different control depending on how many dataloader workers were running."""
        box = _box(x=24, y=40)
        brain = _brain()
        union = box.to_bool_mask(128, 128) | _box(x=87, y=40).to_bool_mask(128, 128)

        a, _ = sample_matched_control(box, brain=brain, lesion_union=union, global_seed=42)
        b, _ = sample_matched_control(box, brain=brain, lesion_union=union, global_seed=42)
        assert a is not None and b is not None
        assert (a.x, a.y, a.seed) == (b.x, b.y, b.seed)

    def test_a_different_global_seed_gives_a_different_draw(self) -> None:
        box = _box(x=24, y=40)
        brain = _brain()
        union = box.to_bool_mask(128, 128) | _box(x=87, y=40).to_bool_mask(128, 128)
        a, _ = sample_matched_control(box, brain=brain, lesion_union=union, global_seed=1)
        b, _ = sample_matched_control(box, brain=brain, lesion_union=union, global_seed=2)
        assert a is not None and b is not None
        assert a.seed != b.seed

    def test_the_seed_does_not_depend_on_the_other_boxes(self) -> None:
        """Two slices differing only in an unrelated second lesion must still give
        the first lesion the same seed."""
        box = _box(x=24, y=40)
        brain = _brain()
        alone = sample_matched_control(
            box, brain=brain, lesion_union=box.to_bool_mask(128, 128), global_seed=7
        )[0]
        crowded = sample_matched_control(
            box,
            brain=brain,
            lesion_union=box.to_bool_mask(128, 128) | _box(x=87, y=40).to_bool_mask(128, 128),
            global_seed=7,
        )[0]
        assert alone is not None and crowded is not None
        assert alone.seed == crowded.seed


class TestAnUnpairedLesionIsDropped:
    def test_a_lesion_with_no_valid_control_is_excluded_from_the_study(self) -> None:
        """Keeping it would silently re-introduce the size confound: the lesion pool
        would hold regions with no size-matched counterpart, and the 'paired'
        comparison would quietly become unpaired for part of the cohort."""
        # A brain so small that a 16x16 control cannot be placed anywhere clear of
        # the lesion.
        brain = torch.zeros(128, 128, dtype=torch.bool)
        brain[40:58, 40:58] = True
        box = _box(x=40, y=40, w=16, h=16)

        report = sample_matched_controls([box], brain=brain, global_seed=0)
        assert report.n_paired == 0
        assert report.n_dropped == 1
        lesion_id, why = report.dropped_lesions[0]
        assert "no_valid_control" in why
        assert lesion_id.startswith("path:")

    def test_the_drop_is_counted_in_the_report(self) -> None:
        brain = torch.zeros(64, 64, dtype=torch.bool)
        brain[10:26, 10:26] = True
        report = sample_matched_controls([_box(x=10, y=10, w=16, h=16)], brain=brain, global_seed=0)
        d = report.to_dict()
        assert d["dropped_unpaired"] == 1
        assert d["paired"] == 0

    def test_an_empty_brain_mask_yields_no_control(self) -> None:
        control, why = sample_matched_control(
            _box(),
            brain=torch.zeros(128, 128, dtype=torch.bool),
            lesion_union=torch.zeros(128, 128, dtype=torch.bool),
            global_seed=0,
        )
        assert control is None
        assert why == "empty_brain_mask"


class TestBatchPairing:
    def test_every_lesion_is_either_paired_or_reported(self) -> None:
        boxes = [_box(x=24, y=40), _box(x=30, y=70)]
        report = sample_matched_controls(boxes, brain=_brain(), global_seed=11)
        assert report.n_paired + report.n_dropped == len(boxes)

    def test_controls_carry_their_pairing_provenance(self) -> None:
        report = sample_matched_controls([_box()], brain=_brain(), global_seed=11)
        assert report.n_paired == 1
        prov = report.controls[0].provenance
        assert prov["paired_with"].startswith("path:")
        assert prov["strategy"] in {"contralateral_mirror", "rejection_sample"}
