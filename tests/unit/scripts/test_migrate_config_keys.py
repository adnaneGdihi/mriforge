"""Tests for the rename fixer's three move shapes.

The migrator is the only sanctioned way to change a retired key across the
corpus, so its failure modes matter more than its happy path: a wrong move
rewrites hundreds of arms, and a silent one rewrites them invisibly. Every case
here asserts on the **file text**, not just the parsed value, because the whole
design point is that untouched lines stay byte-identical.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import ClassVar

import pytest

from spectramr.config.schemas.renames import RenameRecord

_MOD_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "ci" / "migrate_config_keys.py"
)


def _load_migrator():
    spec = importlib.util.spec_from_file_location("_mck", _MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mck():
    return _load_migrator()


def _run(mck, tmp_path: Path, text: str, records: list[RenameRecord]):
    """Point the module's table at ``records`` and migrate one file."""
    mck.RENAMES = {r.legacy: r for r in records}
    f = tmp_path / "arm.yaml"
    f.write_text(text)
    report = mck.migrate_file(f, apply=True)
    return f.read_text(), report


def _rec(legacy: str, canonical: str, **kw) -> RenameRecord:
    return RenameRecord(
        legacy=legacy, canonical=canonical, since="2026-01-01", reason="test.", **kw
    )


class TestSameBlockRename:
    def test_renames_the_key_and_touches_nothing_else(self, mck, tmp_path):
        text = "optimization:\n  learning_rate: 0.001  # keep me\n  old: 4\n"
        out, report = _run(
            mck, tmp_path, text, [_rec("optimization.old", "optimization.new")]
        )
        assert out == "optimization:\n  learning_rate: 0.001  # keep me\n  new: 4\n"
        assert "renamed" in report[0]


class TestNestedMove:
    def test_creates_the_missing_sub_block(self, mck, tmp_path):
        text = "data:\n  batch_size: 8\n  num_workers: 4\n"
        out, report = _run(
            mck, tmp_path, text, [_rec("data.batch_size", "data.loader.batch_size")]
        )
        lines = out.splitlines()
        # Whole-line assertions: `"    batch_size"` is a substring of
        # `"      batch_size"`, so a substring check passes on a block indented
        # one level too deep — which is exactly the bug this caught.
        assert "  loader:" in lines
        assert "    batch_size: 8" in lines
        assert "  num_workers: 4" in lines  # untouched sibling keeps its indent
        assert "moved" in report[0]

    def test_reuses_an_existing_sub_block(self, mck, tmp_path):
        text = "data:\n  batch_size: 8\n  loader:\n    num_workers: 4\n"
        out, _ = _run(
            mck, tmp_path, text, [_rec("data.batch_size", "data.loader.batch_size")]
        )
        assert out.count("loader:") == 1
        assert "    batch_size: 8" in out.splitlines()

    def test_a_trailing_comment_block_stays_with_the_key_below_it(self, mck, tmp_path):
        """A comment between two keys documents the one BELOW it.

        ``experiments/inprogress/workflow_baselines/b0`` is the live case: six
        lines explaining why ``task: denoising`` is declared sit directly under
        ``name:``. Sweeping them into ``name:``'s span carries ``task:``'s
        rationale into whatever block ``name:`` moves to.
        """
        text = (
            "run:\n"
            "  seed: 42\n"
            "workflow:\n"
            "  name: mri_structural\n"
            "  # DENOISING is in supported_tasks, so the check passes.\n"
            "  # It DISAGREES with the strategy's own tag.\n"
            "  task: denoising\n"
        )
        out, report = _run(mck, tmp_path, text, [_rec("workflow.name", "run.regime")])
        lines = out.splitlines()
        comment = "  # DENOISING is in supported_tasks, so the check passes."
        # The comments stay under `workflow:`, above `task:` — not dragged into
        # `run:` with the key they merely happened to follow.
        assert lines.index(comment) < lines.index("  task: denoising")
        assert lines.index("  regime: mri_structural") < lines.index(comment)
        assert lines.index("  regime: mri_structural") < lines.index("workflow:")
        assert (
            "value line(s)" not in report[0]
        ), f"comments counted as value lines: {report[0]}"

    def test_moves_a_multi_line_value_whole(self, mck, tmp_path):
        """A list or nested mapping spans several lines; moving only the key line
        would strand the body under the old parent."""
        text = (
            "data:\n"
            "  transforms:\n"
            "    - name: flip\n"
            "    - name: blur\n"
            "  batch_size: 2\n"
        )
        out, report = _run(
            mck,
            tmp_path,
            text,
            [_rec("data.transforms", "data.processing.transforms")],
        )
        lines = out.splitlines()
        assert "  processing:" in lines
        assert "    transforms:" in lines
        assert "      - name: flip" in lines
        assert "      - name: blur" in lines
        assert "  batch_size: 2" in lines
        assert "value line(s)" in report[0]


class TestCrossBlockMove:
    def test_moves_between_top_level_blocks(self, mck, tmp_path):
        text = "acceleration:\n  mixed_precision: true\noptimization:\n  lr: 0.1\n"
        out, report = _run(
            mck,
            tmp_path,
            text,
            [_rec("acceleration.mixed_precision", "optimization.precision.enabled")],
        )
        lines = out.splitlines()
        # Phase 8: the destination is two levels deep, so the fixer has to
        # BUILD `precision:` under an existing `optimization:` block.
        assert "  precision:" in lines
        assert "    enabled: true" in lines
        assert "mixed_precision" not in out
        assert "  lr: 0.1" in lines
        assert "moved" in report[0]

    def test_refuses_when_the_destination_block_is_absent(self, mck, tmp_path):
        """Creating a top-level block would invent structure the author never
        wrote, so this is a refusal rather than a best guess."""
        text = "acceleration:\n  mixed_precision: true\n"
        out, report = _run(
            mck,
            tmp_path,
            text,
            [_rec("acceleration.mixed_precision", "optimization.precision.enabled")],
        )
        assert out == text  # untouched
        assert "SKIP" in report[0] and "optimization:" in report[0]


class TestTwoRecordsOneDestination:
    """Two legacy spellings may collapse into one canonical key.

    ``seed`` and ``training.seed`` both become ``run.seed``. The second record
    runs against a document the first has already rewritten, so any comparison
    against the file *as it arrived* is stale: it saw ``run.seed`` absent, called
    an agreeing pair a disagreement, skipped, and left the legacy key behind to
    fail the post-write verification with "survived the rewrite".
    """

    RECORDS: ClassVar[list] = [
        _rec("seed", "run.seed", create_destination_block=True),
        _rec("training.seed", "run.seed", create_destination_block=True),
    ]

    def test_agreeing_pair_collapses_to_one_key(self, mck, tmp_path):
        text = "seed: 42\ntraining:\n  seed: 42\n  epochs: 5\n"
        out, report = _run(mck, tmp_path, text, self.RECORDS)
        lines = out.splitlines()
        assert "run:" in lines
        assert "  seed: 42" in lines
        assert lines.count("  seed: 42") == 1, f"seed written twice:\n{out}"
        assert "seed: 42" not in lines  # the root spelling is gone
        assert "  epochs: 5" in lines
        assert not any("ERROR" in r or "SKIP" in r for r in report), report

    def test_a_genuine_disagreement_is_still_refused(self, mck, tmp_path):
        """The stale-read fix must not turn every pair into a silent merge."""
        text = "seed: 42\ntraining:\n  seed: 99\n"
        out, report = _run(mck, tmp_path, text, self.RECORDS)
        assert any("SKIP" in r and "disagree" in r for r in report), report
        assert "  seed: 99" in out.splitlines()  # left for a human


class TestCreateDestinationBlock:
    def test_opt_in_creates_the_missing_top_level_block(self, mck, tmp_path):
        text = "config_version: '6.1'\nseed: 7\ndata:\n  batch_size: 8\n"
        out, report = _run(
            mck,
            tmp_path,
            text,
            [_rec("seed", "run.seed", create_destination_block=True)],
        )
        lines = out.splitlines()
        assert "run:" in lines
        assert "  seed: 7" in lines
        assert "seed: 7" not in lines  # the root spelling is gone
        assert "moved" in report[0]

    def test_without_the_opt_in_it_is_still_refused(self, mck, tmp_path):
        """The default must stay a refusal: for a move between two EXISTING
        schema blocks, a missing destination usually means the arm relies on
        that block's defaults, and conjuring it invents structure."""
        text = "config_version: '6.1'\nseed: 7\n"
        out, report = _run(mck, tmp_path, text, [_rec("seed", "run.seed")])
        assert out == text
        assert "SKIP" in report[0] and "run:" in report[0]

    def test_a_created_block_lands_at_the_end_not_line_zero(self, mck, tmp_path):
        """Line 0 sits above the document's header comment and above
        `config_version:`, so a block there reads as though the header belongs
        to it."""
        text = "# Arm rationale, top of file.\nconfig_version: '6.1'\nseed: 7\n"
        out, _ = _run(
            mck,
            tmp_path,
            text,
            [_rec("seed", "run.seed", create_destination_block=True)],
        )
        lines = out.splitlines()
        assert lines[0] == "# Arm rationale, top of file."
        assert lines.index("run:") > lines.index("config_version: '6.1'")


class TestLeadingComments:
    def test_a_comment_above_the_key_moves_with_it(self, mck, tmp_path):
        """Mirror of the trailing-comment rule: a comment ABOVE a key documents
        that key, so leaving it behind strands an explanation pointing at
        whatever now follows."""
        text = (
            "config_version: '6.1'\n"
            "# Fixed across the cohort so ablations are comparable.\n"
            "seed: 7\n"
            "data:\n"
            "  batch_size: 8\n"
        )
        out, _ = _run(
            mck,
            tmp_path,
            text,
            [_rec("seed", "run.seed", create_destination_block=True)],
        )
        lines = out.splitlines()
        comment = "  # Fixed across the cohort so ablations are comparable."
        assert comment in lines, f"comment stranded at the old location:\n{out}"
        assert lines.index(comment) == lines.index("  seed: 7") - 1
        assert lines.index(comment) > lines.index("run:")

    def test_a_dedented_banner_stays_with_its_section(self, mck, tmp_path):
        """Only same-indent comments are carried. A section banner sitting at
        the parent's indent documents the section, not the first key under it."""
        text = "training:\n# ---- training ----\n  seed: 7\n  epochs: 5\n"
        out, _ = _run(
            mck,
            tmp_path,
            text,
            [_rec("training.seed", "run.seed", create_destination_block=True)],
        )
        lines = out.splitlines()
        assert "# ---- training ----" in lines
        assert lines.index("# ---- training ----") < lines.index("  epochs: 5")


class TestRefusals:
    def test_both_spellings_present_is_a_skip(self, mck, tmp_path):
        text = "optimization:\n  old: 4\n  new: 9\n"
        out, report = _run(
            mck, tmp_path, text, [_rec("optimization.old", "optimization.new")]
        )
        assert out == text
        assert "SKIP" in report[0]

    def test_non_literal_bool_under_negate_is_a_skip(self, mck, tmp_path):
        text = "losses:\n  disable_x: 3\n"
        out, report = _run(
            mck,
            tmp_path,
            text,
            [_rec("losses.disable_x", "losses.enable_x", value_transform="negate")],
        )
        assert out == text
        assert "SKIP" in report[0] and "negate" in report[0]


class TestNegate:
    def test_inverts_the_literal_while_moving(self, mck, tmp_path):
        text = "losses:\n  disable_defaults: true\n"
        out, _ = _run(
            mck,
            tmp_path,
            text,
            [
                _rec(
                    "losses.disable_defaults",
                    "losses.policy.enable_defaults",
                    value_transform="negate",
                )
            ],
        )
        assert "    enable_defaults: false" in out.splitlines()

    def test_redundant_duplicate_is_dropped_when_values_agree(self, mck, tmp_path):
        """Both spellings present and equal is not an ambiguity — the legacy line
        is merely redundant, and renaming it would emit the canonical key twice."""
        text = "optimization:\n  old: 4\n  new: 4\n"
        out, report = _run(
            mck, tmp_path, text, [_rec("optimization.old", "optimization.new")]
        )
        assert out == "optimization:\n  new: 4\n"
        assert "dropped redundant" in report[0]


class TestBlockSequenceValues:
    """A list value written at the KEY's own indent must move with its key.

    ``yaml.safe_dump`` and ``ruamel`` both emit a block sequence flush with its
    key rather than indented under it::

        patch_size:
        - 256
        - 256

    That is valid YAML, and an indentation-only span reads ``- 256`` as a
    sibling: the span ends at the key line, the list is stranded, and the
    rewrite does not parse. The fixer's own verification then refuses the file,
    so the failure is safe but total -- **every** list-valued key
    (``patch_size``, ``transforms``, ``debug_log_steps``, ``validation.metrics``)
    was quietly unmigratable, and the corpus migration would have skipped all of
    them while reporting success on everything else.

    It surfaced only once two records targeted one canonical key: the second
    record parses the intermediate text to compare both spellings, and that
    parse -- unlike the final one -- is not guarded, so the whole run crashed
    instead of skipping one file.
    """

    def test_a_flush_block_sequence_moves_with_its_key(self, mck, tmp_path):
        text = "data:\n  patch_size:\n  - 256\n  - 256\n  - 1\n  batch_size: 2\n"
        out, report = _run(
            mck, tmp_path, text, [_rec("data.patch_size", "data.sampling.patch_size")]
        )
        assert "MIGRATED" in report[0], report
        # The destination block is appended, so `batch_size` (untouched) leads.
        assert out == (
            "data:\n"
            "  batch_size: 2\n"
            "  sampling:\n"
            "    patch_size:\n"
            "    - 256\n"
            "    - 256\n"
            "    - 1\n"
        ), out

    def test_the_following_key_is_untouched(self, mck, tmp_path):
        """The span must end AT the next real sibling, not swallow it."""
        text = "data:\n  transforms:\n  - a\n  - b\n  dataset_type: nifti\n"
        out, _ = _run(
            mck, tmp_path, text, [_rec("data.transforms", "data.processing.transforms")]
        )
        assert "  dataset_type: nifti\n" in out
        assert out.count("dataset_type") == 1

    def test_an_indented_sequence_still_moves(self, mck, tmp_path):
        """The other legal spelling must keep working."""
        text = "data:\n  patch_size:\n    - 256\n    - 256\n  batch_size: 2\n"
        out, report = _run(
            mck, tmp_path, text, [_rec("data.patch_size", "data.sampling.patch_size")]
        )
        assert "MIGRATED" in report[0], report
        assert "- 256" in out
        assert "  batch_size: 2\n" in out

    def test_a_sequence_of_mappings_moves_whole(self, mck, tmp_path):
        text = "data:\n  datasets:\n  - name: a\n    root: /x\n  - name: b\n  batch_size: 2\n"
        out, _ = _run(
            mck, tmp_path, text, [_rec("data.datasets", "data.source.datasets")]
        )
        assert "name: a" in out and "root: /x" in out and "name: b" in out
        assert "  batch_size: 2\n" in out


class TestRecordsScoping:
    """`records=` must scope the post-write verification, not just the rewrite.

    The verification loop iterated the whole `RENAMES` table regardless of what
    the caller asked for. Since 828 of 829 corpus files carry a legacy key from
    *some* record, `migrate_file(apply=True, records={one})` reported
    "<unrelated key> survived the rewrite" and returned BEFORE `path.write_text`
    -- silently discarding work it had just done correctly.

    That made per-record draining impossible, which is the unit the promotion
    rule operates on ("when a record's count reaches zero, flip its posture to
    `raise`"). The failure mode is a SILENT no-op, so the assertion that matters
    is "the file changed", not "no error was reported".
    """

    TEXT = (
        "config_version: '6.1'\n"
        "data:\n"
        "  patch_size: 64\n"
        "optimization:\n"
        "  optimizer_type: adam\n"
    )

    def test_a_scoped_apply_actually_writes(self, mck, tmp_path) -> None:
        out, report = _run(
            mck,
            tmp_path,
            self.TEXT,
            [_rec("data.patch_size", "data.sampling.patch_size")],
        )
        assert "MIGRATED" in report[0], report
        assert out != self.TEXT, (
            "the file was not written -- the verification loop rejected it over a "
            "key this call was never asked to migrate"
        )
        assert "sampling:" in out

    def test_an_untargeted_legacy_key_survives(self, mck, tmp_path) -> None:
        """Scoping must LEAVE the other keys, not migrate or delete them."""
        out, _ = _run(
            mck,
            tmp_path,
            self.TEXT,
            [_rec("data.patch_size", "data.sampling.patch_size")],
        )
        assert "  optimizer_type: adam\n" in out

    def test_an_unscoped_run_still_verifies_everything(self, mck, tmp_path) -> None:
        """Control: passing no subset must keep the full-table verification, or
        the fix would have traded a silent discard for a silent miss."""
        import inspect

        src = inspect.getsource(mck.migrate_file)
        assert "RENAMES if records is None else records" in src, (
            "the verification loop no longer falls back to the full table when "
            "no subset is given"
        )


class TestSupersededBy:
    """An adjudicated pair must migrate, not SKIP forever.

    Two spellings with different values is normally a refusal — only a human can
    say which the arm meant. But `validation.val_batch_size` beats
    `validation.validation_batch_size` by both fields' own descriptions, and
    `ValidationConfigSchema._resolve_batch_size_duplicate` already enforces that
    at parse time. Without `superseded_by` the migrator SKIPped all 74 corpus
    arms declaring the pair, so the record's count never reached zero and its
    fold shim could never be promoted to `raise` — the drain stalled on one
    record, permanently.

    Dropping the loser is safe precisely because the validator already drops it:
    the RESOLVED document is unchanged. Verified on 40 real arms.
    """

    TEXT = (
        "config_version: '6.1'\n"
        "validation:\n"
        "  val_batch_size: 2\n"
        "  validation_batch_size: 1\n"
    )

    def test_the_loser_is_dropped_rather_than_skipped(self, mck, tmp_path) -> None:
        out, report = _run(
            mck,
            tmp_path,
            self.TEXT,
            [
                _rec("validation.val_batch_size", "validation.loader.batch_size"),
                _rec(
                    "validation.validation_batch_size",
                    "validation.loader.batch_size",
                    superseded_by="validation.val_batch_size",
                ),
            ],
        )
        assert not any("SKIP" in line for line in report), report
        assert any("dropped superseded" in line for line in report), report
        assert "validation_batch_size" not in out
        assert "    batch_size: 2" in out.splitlines(), out

    def test_without_superseded_by_it_still_refuses(self, mck, tmp_path) -> None:
        """The control. Making disagreements droppable in general would silently
        decide which objective an arm trains."""
        _, report = _run(
            mck,
            tmp_path,
            self.TEXT,
            [
                _rec("validation.val_batch_size", "validation.loader.batch_size"),
                _rec(
                    "validation.validation_batch_size",
                    "validation.loader.batch_size",
                ),
            ],
        )
        assert any("SKIP" in line for line in report), report

    def test_the_winner_is_declared_before_the_loser(self) -> None:
        """Ordering is load-bearing and silent if wrong.

        The winner's record must run first, so the destination already holds its
        value by the time the loser is dropped. Declared the other way round the
        loser writes ITS value and the winner — which has no `superseded_by` —
        then SKIPs, inverting the documented precedence with no error. That is
        exactly what happened on the first attempt.
        """
        from spectramr.config.schemas.renames import RENAMES

        order = list(RENAMES)
        for legacy, rec in RENAMES.items():
            if rec.superseded_by:
                assert rec.superseded_by in order, rec.superseded_by
                assert order.index(rec.superseded_by) < order.index(legacy), (
                    f"{legacy} is declared before its winner {rec.superseded_by}; "
                    f"the loser would migrate first and the winner would SKIP"
                )

    def test_the_schema_resolver_reads_the_same_field(self) -> None:
        """One rule, not a validator and a fixer that agree until they don't."""
        import inspect

        from spectramr.config.schemas import validation as v

        src = inspect.getsource(v.ValidationConfigSchema._resolve_batch_size_duplicate)
        assert "superseded_by" in src and "RENAMES" in src, (
            "the schema resolver hardcodes the pair again; the migrator and the "
            "validator can now drift about which spelling wins"
        )
