"""Tests for baseline discovery + eval-task expansion (Task 1).

Covers:
- discover_baselines: parses the cluster baselines/ tree into BaselineSpec objects.
- build_eval_tasks: expands specs into EvalTask objects for all cross-field pairs.
- joint_domain: maps (contrast_dir, field_dir) to an integer in [0, 15).
- Missing checkpoint files: that method dir is skipped with a warning.
- Zero-checkpoint root: raises ValueError.
- task3_pairs modes: "all", "to7t", "task1_task2".
"""

from __future__ import annotations

import logging

import pytest

from spectramr.infrastructure.evaluation.mrixfields_baselines.discovery import (
    FIELDS,
    MODALITIES,
    build_eval_tasks,
    discover_baselines,
    joint_domain,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def baselines_root(tmp_path):
    """Build a minimal cluster-layout baselines/ tree with empty .pth files."""

    def _touch(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    root = tmp_path / "baselines"

    # task1: 0.1T to 7T, T1W -- two methods
    _touch(
        root
        / "task1_0.1T_to_7T_T1W"
        / "cut"
        / "pro_pretrained"
        / "weights"
        / "checkpoint_epoch100.pth"
    )
    _touch(
        root
        / "task1_0.1T_to_7T_T1W"
        / "cyclegan"
        / "pro_pretrained"
        / "weights"
        / "checkpoint_epoch100.pth"
    )

    # task2: 0.1T to 3T, T2FLAIR -- cyclegan only (cut has no checkpoint -> skipped)
    _touch(
        root
        / "task2_0.1T_to_3T_T2FLAIR"
        / "cyclegan"
        / "pro_pretrained"
        / "weights"
        / "checkpoint_epoch100.pth"
    )
    # 'cut' subdir exists but has no .pth -- leave it empty (no weights dir)
    (root / "task2_0.1T_to_3T_T2FLAIR" / "cut").mkdir(parents=True, exist_ok=True)

    # task3: any-to-any stargan_v2
    _touch(
        root
        / "task3_any_to_any_multimodal"
        / "stargan_v2"
        / "pro_pretrained"
        / "weights"
        / "checkpoint_epoch50.pth"
    )

    # Non-matching entries that should be ignored
    (root / "README.md").touch()
    (root / "LICENSE").touch()

    return root


# ---------------------------------------------------------------------------
# joint_domain
# ---------------------------------------------------------------------------


class TestJointDomain:
    def test_t1w_01t(self):
        # T1W=0, 0.1T=0 -> 0*5+0 = 0
        assert joint_domain("T1W", "0.1T") == 0

    def test_t1w_7t(self):
        # T1W=0, 7T=4 -> 0*5+4 = 4
        assert joint_domain("T1W", "7T") == 4

    def test_t2w_3t(self):
        # T2W=1, 3T=2 -> 1*5+2 = 7
        assert joint_domain("T2W", "3T") == 7

    def test_t2flair_5t(self):
        # T2FLAIR=2, 5T=3 -> 2*5+3 = 13
        assert joint_domain("T2FLAIR", "5T") == 13

    def test_range_all(self):
        for i, mod in enumerate(MODALITIES):
            for j, fld in enumerate(FIELDS):
                assert joint_domain(mod, fld) == i * len(FIELDS) + j


# ---------------------------------------------------------------------------
# discover_baselines
# ---------------------------------------------------------------------------


class TestDiscoverBaselines:
    def test_task1_cut_spec(self, baselines_root):
        specs = discover_baselines(baselines_root)
        cut_specs = [s for s in specs if s.task == 1 and s.method == "cut"]
        assert len(cut_specs) == 1
        s = cut_specs[0]
        assert s.source_field == pytest.approx(0.1)
        assert s.target_field == pytest.approx(7.0)
        assert s.contrast == "T1w"
        assert s.checkpoint.exists()

    def test_task1_cyclegan_spec(self, baselines_root):
        specs = discover_baselines(baselines_root)
        cyc = [s for s in specs if s.task == 1 and s.method == "cyclegan"]
        assert len(cyc) == 1
        assert cyc[0].contrast == "T1w"

    def test_task2_cyclegan_only(self, baselines_root):
        """task2 cut has no checkpoint -> only cyclegan spec returned."""
        specs = discover_baselines(baselines_root)
        t2 = [s for s in specs if s.task == 2]
        assert len(t2) == 1
        assert t2[0].method == "cyclegan"
        assert t2[0].source_field == pytest.approx(0.1)
        assert t2[0].target_field == pytest.approx(3.0)
        assert t2[0].contrast == "T2FLAIR"

    def test_task3_spec(self, baselines_root):
        specs = discover_baselines(baselines_root)
        t3 = [s for s in specs if s.task == 3]
        assert len(t3) == 1
        s = t3[0]
        assert s.method == "stargan_v2"
        assert s.source_field is None
        assert s.target_field is None
        assert s.contrast is None
        assert s.checkpoint.name == "checkpoint_epoch50.pth"

    def test_non_matching_dirs_ignored(self, baselines_root):
        specs = discover_baselines(baselines_root)
        # README.md and LICENSE are not task dirs
        assert all(not s.name.startswith("README") for s in specs)
        assert all(not s.name.startswith("LICENSE") for s in specs)

    def test_missing_checkpoint_logs_warning(self, baselines_root, caplog):
        with caplog.at_level(
            logging.WARNING,
            logger="spectramr.infrastructure.evaluation.mrixfields_baselines.discovery",
        ):
            discover_baselines(baselines_root)
        # The missing cut checkpoint for task2 should produce a warning
        assert any("cut" in r.message and "skipping" in r.message for r in caplog.records)

    def test_zero_checkpoints_raises(self, tmp_path):
        root = tmp_path / "empty_baselines"
        root.mkdir()
        (root / "README.md").touch()
        with pytest.raises(ValueError, match="no baseline checkpoints"):
            discover_baselines(root)

    def test_total_spec_count(self, baselines_root):
        specs = discover_baselines(baselines_root)
        # task1: cut + cyclegan = 2, task2: cyclegan = 1, task3: stargan_v2 = 1
        assert len(specs) == 4

    def test_multiple_checkpoints_logs_warning(self, tmp_path, caplog):
        """M2: >1 .pth under weights/ -> WARNING naming the chosen (first) file."""
        weights = (
            tmp_path
            / "baselines"
            / "task1_0.1T_to_7T_T1W"
            / "cut"
            / "pro_pretrained"
            / "weights"
        )
        weights.mkdir(parents=True, exist_ok=True)
        (weights / "checkpoint_epoch050.pth").touch()
        (weights / "checkpoint_epoch100.pth").touch()

        with caplog.at_level(
            logging.WARNING,
            logger="spectramr.infrastructure.evaluation.mrixfields_baselines.discovery",
        ):
            specs = discover_baselines(tmp_path / "baselines")

        # exactly one spec (the chosen checkpoint), and it is the first sorted name
        cut = [s for s in specs if s.task == 1 and s.method == "cut"]
        assert len(cut) == 1
        assert cut[0].checkpoint.name == "checkpoint_epoch050.pth"
        # warning names both the count and the chosen file
        assert any(
            "2 checkpoints" in r.message and "checkpoint_epoch050.pth" in r.message
            for r in caplog.records
        )


# ---------------------------------------------------------------------------
# build_eval_tasks
# ---------------------------------------------------------------------------


CONTRASTS = ("T1w", "T2w", "T2FLAIR")
FIELDS_TUPLE = (0.1, 1.5, 3.0, 5.0, 7.0)


class TestBuildEvalTasks:
    def test_task1_cut_gives_one_task(self, baselines_root):
        specs = [
            s
            for s in discover_baselines(baselines_root)
            if s.task == 1 and s.method == "cut"
        ]
        tasks = build_eval_tasks(specs, contrasts=CONTRASTS, fields=FIELDS_TUPLE, task3_pairs="all")
        assert len(tasks) == 1
        t = tasks[0]
        assert t.source_field == pytest.approx(0.1)
        assert t.target_field == pytest.approx(7.0)
        assert t.contrast == "T1w"
        assert t.target_domain is None

    def test_task3_all_pairs_count(self, baselines_root):
        specs = [s for s in discover_baselines(baselines_root) if s.task == 3]
        tasks = build_eval_tasks(specs, contrasts=CONTRASTS, fields=FIELDS_TUPLE, task3_pairs="all")
        # 20 ordered (src!=tgt) pairs x 3 contrasts = 60
        assert len(tasks) == 60

    def test_task3_target_domain_set(self, baselines_root):
        specs = [s for s in discover_baselines(baselines_root) if s.task == 3]
        tasks = build_eval_tasks(specs, contrasts=CONTRASTS, fields=FIELDS_TUPLE, task3_pairs="all")
        for t in tasks:
            assert t.target_domain is not None
            assert 0 <= t.target_domain < len(MODALITIES) * len(FIELDS)

    def test_task3_to7t_mode(self, baselines_root):
        specs = [s for s in discover_baselines(baselines_root) if s.task == 3]
        tasks = build_eval_tasks(specs, contrasts=CONTRASTS, fields=FIELDS_TUPLE, task3_pairs="to7t")
        # 4 sources (not 7T) x 3 contrasts = 12
        assert len(tasks) == 12
        assert all(t.target_field == pytest.approx(7.0) for t in tasks)

    def test_task3_task1_task2_mode(self, baselines_root):
        specs = [s for s in discover_baselines(baselines_root) if s.task == 3]
        tasks = build_eval_tasks(
            specs, contrasts=CONTRASTS, fields=FIELDS_TUPLE, task3_pairs="task1_task2"
        )
        # {*->7T} U {0.1->*}: 4 + 4 pairs share (0.1->7T) => 7 unique per contrast
        # x 3 contrasts = 21
        assert len(tasks) == 21
        target_fields = {t.target_field for t in tasks}
        source_fields = {t.source_field for t in tasks}
        assert 7.0 in target_fields
        assert 0.1 in source_fields

    def test_unknown_task3_pairs_raises(self, baselines_root):
        specs = [s for s in discover_baselines(baselines_root) if s.task == 3]
        with pytest.raises(ValueError, match="unknown task3_pairs"):
            build_eval_tasks(
                specs, contrasts=CONTRASTS, fields=FIELDS_TUPLE, task3_pairs="bogus"
            )

    def test_task3_all_target_domain_sample(self, baselines_root):
        """Verify a specific target_domain value matches the joint_domain formula."""
        specs = [s for s in discover_baselines(baselines_root) if s.task == 3]
        tasks = build_eval_tasks(specs, contrasts=("T1w",), fields=(0.1, 7.0), task3_pairs="all")
        # only 2 ordered pairs: (0.1->7.0) and (7.0->0.1)
        assert len(tasks) == 2
        t_01_7 = next(t for t in tasks if t.source_field == pytest.approx(0.1))
        # T1w -> "T1W", 7T -> idx 4, T1W -> idx 0 -> 0*5+4 = 4
        assert t_01_7.target_domain == joint_domain("T1W", "7T")

    def test_all_tasks_combined(self, baselines_root):
        specs = discover_baselines(baselines_root)
        tasks = build_eval_tasks(specs, contrasts=CONTRASTS, fields=FIELDS_TUPLE, task3_pairs="all")
        # task1 cut: 1, task1 cyclegan: 1, task2 cyclegan: 1, task3 stargan: 60
        assert len(tasks) == 63
