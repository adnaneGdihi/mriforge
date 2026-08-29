"""Unit tests for the shared train/val index split helper (data SSOT).

These are pure-Python and run without torch: ``split_index`` is the single
source of truth every dataset instantiator / manifest loader now delegates to,
so its contract (no leak, no empty split, honest single-file handling) is
pinned here once instead of re-tested per call site.
"""

import pytest

from mriforge.data.split_utils import split_index


class TestSplitIndexHappyPath:
    def test_non_overlapping_disjoint_union(self):
        items = list(range(10))
        train, val = split_index(items, 0.2)
        assert set(train).isdisjoint(val)
        assert sorted(train + val) == items
        assert len(val) == 2 and len(train) == 8

    def test_val_is_the_tail(self):
        items = list(range(10))
        train, val = split_index(items, 0.3)
        # Convention: validation is the last n_val items (matches every caller).
        assert val == [7, 8, 9]
        assert train == [0, 1, 2, 3, 4, 5, 6]

    def test_deterministic(self):
        items = list(range(17))
        assert split_index(items, 0.1) == split_index(items, 0.1)

    def test_rounds_to_nearest(self):
        # round(15 * 0.1) == 2 (int() would have truncated to 1).
        _, val = split_index(list(range(15)), 0.1)
        assert len(val) == 2

    def test_both_splits_non_empty_for_multi_file(self):
        for n in range(2, 12):
            for frac in (0.01, 0.1, 0.5, 0.9, 0.99):
                train, val = split_index(list(range(n)), frac)
                assert len(train) >= 1, (n, frac)
                assert len(val) >= 1, (n, frac)

    def test_does_not_mutate_input(self):
        items = [3, 1, 2]
        split_index(items, 0.5)
        assert items == [3, 1, 2]


class TestSplitIndexTrainOnly:
    def test_zero_split_is_train_only(self):
        items = list(range(5))
        train, val = split_index(items, 0.0)
        assert train == items
        assert val == []

    def test_negative_split_is_train_only(self):
        train, val = split_index([1, 2, 3], -0.1)
        assert train == [1, 2, 3]
        assert val == []

    def test_single_file_train_only_is_allowed(self):
        # The escape hatch: one file + split==0 is a legitimate train-only run.
        train, val = split_index(["only.h5"], 0.0)
        assert train == ["only.h5"]
        assert val == []


class TestSplitIndexSingleFileRaises:
    def test_single_file_with_split_raises(self):
        with pytest.raises(ValueError, match="single file"):
            split_index(["only.h5"], 0.1)

    def test_error_names_the_escape_hatch(self):
        with pytest.raises(ValueError, match="validation_split: 0"):
            split_index([object()], 0.5)


class TestSplitIndexEmpty:
    def test_empty_returns_two_empties(self):
        assert split_index([], 0.1) == ([], [])

    def test_empty_train_only(self):
        assert split_index([], 0.0) == ([], [])


class TestSplitIndexIsTheOnlySplitter:
    """``split_index`` is the data-layer SSOT; nothing may re-derive a partition.

    Two modules did, and they disagreed with it — and with each other — in
    different ways (2026-08-05):

    * ``manifest_loader``'s carve-from-train fallback truncated with ``int()``
      where the SSOT rounds, and on a single train record produced
      ``max(1, 0) == 1``, handing the only file to validation and leaving
      TRAINING EMPTY.
    * ``preprocessed_dataset._apply_split`` sliced validation off the START
      while every other loader takes it off the end, truncated the same way, and
      yielded a silently EMPTY validation set whenever ``n * fraction < 1``.

    Those are three of the exact drift behaviours ``split_utils``'s module
    docstring names as its reason to exist ("silent leak, empty train,
    warn-and-leak"), which is what makes a structural guard worth more than
    another value assertion: the values only tell you about the cases you
    thought to enumerate.
    """

    @staticmethod
    def _split_size_expressions(tree, source: str) -> list[str]:
        """Multiplications by a validation-fraction-ish name — i.e. someone
        computing how many items to hold out."""
        import ast

        names = ("validation_split", "validation_fraction", "val_frac", "val_split")
        hits = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)):
                continue
            for side in (node.left, node.right):
                spelled = (
                    side.id
                    if isinstance(side, ast.Name)
                    else side.attr
                    if isinstance(side, ast.Attribute)
                    else None
                )
                if spelled in names:
                    hits.append(f"line {node.lineno}: {ast.unparse(node)}")
        return hits

    def test_no_module_computes_its_own_split_size(self) -> None:
        import ast
        from pathlib import Path

        root = Path(__file__).resolve().parents[3] / "src" / "mriforge"
        offenders: list[str] = []
        for path in sorted(root.rglob("*.py")):
            if path.name == "split_utils.py":
                continue  # the SSOT itself is where the arithmetic belongs
            source = path.read_text(encoding="utf-8")
            if not any(
                n in source
                for n in ("validation_split", "validation_fraction", "val_frac")
            ):
                continue
            for hit in self._split_size_expressions(ast.parse(source), source):
                offenders.append(f"{path.relative_to(root.parents[1])}: {hit}")

        assert offenders == [], (
            "A module is computing its own train/val partition size instead of "
            "calling mriforge.data.split_utils.split_index:\n  "
            + "\n  ".join(offenders)
        )

    def test_the_guard_would_catch_a_reintroduction(self) -> None:
        """A structural check that cannot fail is not a check."""
        import ast

        src = "n_val = int(len(items) * validation_split)\n"
        hits = self._split_size_expressions(ast.parse(src), src)
        assert len(hits) == 1 and "validation_split" in hits[0]

    def test_attribute_spelling_is_caught_too(self) -> None:
        """``config.split.validation_fraction`` is an Attribute, not a Name —
        the shape the corpus actually uses at most call sites."""
        import ast

        src = "n = round(len(x) * config.split.validation_fraction)\n"
        assert len(self._split_size_expressions(ast.parse(src), src)) == 1
