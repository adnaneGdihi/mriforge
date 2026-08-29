"""Unit tests for ``scripts/ci/check_dataloader_construction_ssot.py`` (D22, #1362).

The gate's whole value is that it resolves the SYMBOL rather than matching text,
so the tests are weighted toward the two ways a text matcher fails: the shapes it
must ignore (class definitions, docstring mentions, same-named locals) and the
spellings it must not miss (bare, fully-qualified, aliased, and -- since #1362 --
every ``DataLoader`` SUBCLASS).

Non-negotiable #15: a gate is only a gate for the violation shape you have
watched it fail on. So every rule here is planted, one plant per shape, and the
end-to-end plants run through :func:`find_unsanctioned` on a real tree rather
than through the AST helper alone -- a helper that reports a site while the
tree-walker filters it out is still a hole.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "ci" / "check_dataloader_construction_ssot.py"
_REFRESHER = _REPO_ROOT / "scripts" / "ci" / "refresh_dataloader_binding_names.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate():
    """The checker module.

    Loaded via ``spec_from_file_location`` because ``scripts/`` is not an
    importable package. Any failure here RAISES rather than skipping: a skip
    would make a deleted or renamed script read as a green test file, which is
    how this repo previously accumulated tests that covered nothing.
    """
    return _load_module(_SCRIPT, "_dl_ssot_gate")


def _plant(tmp_path: Path, source: str, name: str = "rogue.py") -> Path:
    """Write ``source`` into a throwaway ``src/mriforge`` tree and return its root."""
    pkg = tmp_path / "src" / "mriforge" / "infrastructure"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / name).write_text(source, encoding="utf-8")
    return tmp_path


class TestSpellingsItMustNotMiss:
    """Reachable spellings of the torch loader; knowing two reports 'clean'."""

    def test_bare_name_import(self, gate):
        src = "from torch.utils.data import DataLoader\ndef f():\n    return DataLoader(ds)\n"
        assert gate.find_construction_sites(src) == [(3, "f")]

    def test_fully_qualified_attribute_chain(self, gate):
        src = "import torch\ndef f():\n    return torch.utils.data.DataLoader(ds)\n"
        assert gate.find_construction_sites(src) == [(3, "f")]

    def test_aliased_import(self, gate):
        """``from ... import DataLoader as _DataLoader`` — the real spelling in
        ``data/builders/data_loader_builder.py``."""
        src = (
            "from torch.utils.data import DataLoader as _DataLoader\n"
            "def f():\n"
            "    return _DataLoader(ds)\n"
        )
        assert gate.find_construction_sites(src) == [(3, "f")]

    def test_dataloader_submodule_import_path(self, gate):
        src = (
            "from torch.utils.data.dataloader import DataLoader\n"
            "def f():\n"
            "    return DataLoader(ds)\n"
        )
        assert gate.find_construction_sites(src) == [(3, "f")]


class TestSubclassSpellings:
    """#1362: a subclass IS a DataLoader, and the vocabulary must be discovered.

    Hand-reasoning the subclass names gives ``{DataLoader, SubjectsLoader,
    ThreadDataLoader}``. Discovery returns five, because monai re-exports the
    torch loader under names that appear in no class ``__name__``. The
    ``TorchDataLoader`` case below is the plant that stays GREEN against a
    hand-written list — which is exactly what "not a gate" looks like.
    """

    def test_torchio_subjectsloader_as_attribute(self, gate):
        """The shape that hid four real construction sites under ``data/``."""
        src = "import torchio as tio\ndef f():\n    return tio.SubjectsLoader(ds)\n"
        assert gate.find_construction_sites(src) == [(3, "f")]

    def test_torchio_subjectsloader_as_bare_name(self, gate):
        src = "from torchio import SubjectsLoader\ndef f():\n    return SubjectsLoader(ds)\n"
        assert gate.find_construction_sites(src) == [(3, "f")]

    def test_subclass_import_can_be_aliased_too(self, gate):
        src = "from torchio import SubjectsLoader as _SL\ndef f():\n    return _SL(ds)\n"
        assert gate.find_construction_sites(src) == [(3, "f")]

    def test_monai_threaddataloader(self, gate):
        src = "from monai.data import ThreadDataLoader\nl = ThreadDataLoader(ds)\n"
        assert gate.find_construction_sites(src) == [(2, "<module>")]

    @pytest.mark.parametrize("name", ["TorchDataLoader", "_TorchDataLoader"])
    def test_reexport_binding_name_with_no_matching_class_name(self, gate, name):
        """The plant a ``__name__``-derived vocabulary walks straight past.

        ``monai.transforms.inverse_batch_transform`` re-exports the torch loader
        as ``TorchDataLoader``/``_TorchDataLoader``. Neither string is any
        class's ``__name__``, so only a vocabulary built from ACCESSIBLE
        ATTRIBUTE NAMES sees it.
        """
        src = (
            f"from monai.transforms.inverse_batch_transform import {name}\n"
            f"def f():\n"
            f"    return {name}(ds)\n"
        )
        assert gate.find_construction_sites(src) == [(3, "f")]

    def test_function_local_import_and_construction(self, gate):
        """The ``^``-anchored blind spot: an import that is not at column 0.

        ``check_layering.sh`` was blind to exactly this shape for its whole life
        (#1183). ``ast.walk`` does not care about nesting depth, so this must
        resolve — and the site must be attributed to the enclosing function.
        """
        src = "def f():\n    from torchio import SubjectsLoader\n    return SubjectsLoader(ds)\n"
        assert gate.find_construction_sites(src) == [(3, "f")]


class TestShapesItMustIgnore:
    """Every one of these is a live false positive for ``grep 'DataLoader('``."""

    def test_class_definition_is_not_a_construction(self, gate):
        """``class IDataLoader(ABC)`` in domain/interfaces — a base-class list."""
        src = "from abc import ABC\nclass IDataLoader(ABC):\n    pass\n"
        assert gate.find_construction_sites(src) == []

    def test_docstring_mention_is_not_a_construction(self, gate):
        """``consolidated_dataset_factory`` names the symbol in a mermaid diagram."""
        src = '"""\nDataLoaderBuilder ..> DataLoader\nDataLoader(dataset)\n"""\nx = 1\n'
        assert gate.find_construction_sites(src) == []

    def test_same_named_local_class_without_the_torch_import(self, gate):
        """A module that never imports the symbol cannot violate the SSOT."""
        src = "class DataLoader:\n    pass\ndef f():\n    return DataLoader()\n"
        assert gate.find_construction_sites(src) == []

    def test_annotation_only_use_is_not_a_construction(self, gate):
        src = (
            "from torch.utils.data import DataLoader\n"
            "def f(x: DataLoader) -> DataLoader:\n"
            "    return x\n"
        )
        assert gate.find_construction_sites(src) == []

    def test_a_loader_that_is_not_a_dataloader_subclass(self, gate):
        """``YAMLConfigLoader`` is a real repo symbol; six live call sites.

        The widened gate must not degenerate into "anything named ``*Loader``" —
        that would be wrong in the other direction, and noisily so.
        """
        src = (
            "from mriforge.config.schemas.loader import YAMLConfigLoader\n"
            "def f():\n"
            "    return YAMLConfigLoader(path)\n"
        )
        assert gate.find_construction_sites(src) == []


class TestEnclosingFunctionResolution:
    """The allow-list keys on the enclosing function, so nesting must be exact."""

    def test_innermost_function_wins(self, gate):
        """A nested helper inside an allow-listed method must report the HELPER.

        Otherwise the allow-list widens from one method to everything defined
        beneath it — the failure mode the span-width sort exists to prevent.

        Since ``D05#8`` the name is class-qualified, so the helper reports the
        whole path and still ends in its own name: the widening stays closed and
        the entry now says *which* ``build`` it means.
        """
        src = (
            "from torch.utils.data import DataLoader\n"
            "def build():\n"
            "    def _helper():\n"
            "        return DataLoader(ds)\n"
            "    return _helper()\n"
        )
        assert gate.find_construction_sites(src) == [(4, "build._helper")]

    def test_module_level_construction_is_attributed_to_module(self, gate):
        src = "from torch.utils.data import DataLoader\nloader = DataLoader(ds)\n"
        assert gate.find_construction_sites(src) == [(2, "<module>")]


class TestVocabularyContract:
    """An unknown vocabulary is a state to REPORT, never one to infer (#3, #18)."""

    def test_missing_file_raises_rather_than_defaulting(self, gate, tmp_path):
        """Falling back to ``{"DataLoader"}`` would restore the pre-#1362 blind
        spot AND print "OK" while doing it."""
        with pytest.raises(FileNotFoundError, match="Regenerate it with"):
            gate.read_vocabulary(tmp_path / "absent.txt")

    def test_comments_only_file_raises(self, gate, tmp_path):
        """A file of pure header comments parses to the empty set — which would
        match nothing and report every site as clean."""
        path = tmp_path / "vocab.txt"
        path.write_text("# generated\n#   (none)\n\n", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            gate.read_vocabulary(path)

    def test_committed_vocabulary_is_what_discovery_finds(self, gate):
        """Freshness: a new loader class in a new dependency must not go unseen.

        This is the half plants cannot cover — it catches the spelling nobody
        thought to plant. It needs torch/torchio/monai importable, so it lives
        here rather than in the dependency-free ``guards`` job; that makes it
        non-blocking until ``tests/unit/`` becomes a blocking lane (#207).
        """
        refresher = _load_module(_REFRESHER, "_dl_vocab_refresh")
        discovered, unimportable = refresher.discover(_REPO_ROOT)
        committed = gate.read_vocabulary()
        assert committed == discovered, (
            f"Vocabulary is stale. Regenerate with `python {_REFRESHER.name}`. "
            f"missing={sorted(discovered - committed)} "
            f"extra={sorted(committed - discovered)}"
        )
        assert isinstance(unimportable, list), "unimportable must be reported, not swallowed"


class TestStaleAllowListEntries:
    """An exemption that matches nothing stays satisfied forever."""

    def test_entry_matching_no_site_is_reported(self, gate):
        stale = gate.find_stale_entries(matched=set())
        assert len(stale) == sum(len(e) for e in gate._ALLOWED.values())
        assert all("matched no construction site" in s for s in stale)

    def test_fully_matched_allow_list_is_silent(self, gate):
        every = {(rel, func) for rel, entries in gate._ALLOWED.items() for func, _ in entries}
        assert gate.find_stale_entries(matched=every) == []

    def test_the_real_tree_matches_every_entry(self, gate):
        """The regression pin for #1362 itself.

        Four of the five entries are ``tio.SubjectsLoader`` sites the gate could
        not see before the vocabulary was widened. If matching ever narrows back
        to the name ``DataLoader``, those entries stop matching and this fails —
        so the fix cannot silently regress.
        """
        _, matched = gate.find_unsanctioned(_REPO_ROOT)
        assert gate.find_stale_entries(matched) == []
        assert (
            "src/mriforge/data/builders/torchio_queue_builder.py",
            "TorchIOQueueBuilder.build_train_queue",
        ) in matched


class TestGateOnTheRealTree:
    """The gate must be GREEN on this checkout and RED on a violation."""

    def test_repo_is_clean(self, gate):
        assert gate.check(_REPO_ROOT) == [], (
            "Unsanctioned DataLoader construction site(s) appeared. Route "
            "through the leaf DataLoaderBuilder, or extend _ALLOWED with a reason."
        )

    @pytest.mark.parametrize(
        ("shape", "source"),
        [
            (
                "bare torch name",
                "from torch.utils.data import DataLoader\ndef make():\n    return DataLoader(ds)\n",
            ),
            (
                "qualified chain",
                "import torch\ndef make():\n    return torch.utils.data.DataLoader(ds)\n",
            ),
            (
                "aliased torch",
                "from torch.utils.data import DataLoader as _DL\ndef make():\n    return _DL(ds)\n",
            ),
            (
                "torchio attribute",
                "import torchio as tio\ndef make():\n    return tio.SubjectsLoader(ds)\n",
            ),
            (
                "torchio bare name",
                "from torchio import SubjectsLoader\ndef make():\n    return SubjectsLoader(ds)\n",
            ),
            (
                "monai threaded",
                "from monai.data import ThreadDataLoader\ndef make():\n    return ThreadDataLoader(ds)\n",
            ),
            (
                "monai re-export",
                "from monai.transforms.inverse_batch_transform import TorchDataLoader\ndef make():\n    return TorchDataLoader(ds)\n",
            ),
            (
                "function-local import",
                "def make():\n    from torchio import SubjectsLoader\n    return SubjectsLoader(ds)\n",
            ),
        ],
    )
    def test_each_shape_turns_the_gate_red(self, gate, tmp_path, shape, source):
        """One plant per SHAPE, end-to-end — a checker that cannot fail is not a gate.

        ``find_unsanctioned`` rather than ``check`` because the planted tree
        contains only the rogue file, which would make every real allow-list
        entry read as stale and drown the signal being tested.
        """
        violations, _ = gate.find_unsanctioned(_plant(tmp_path, source))
        assert len(violations) == 1, f"{shape}: expected exactly one violation"
        assert "rogue.py" in violations[0] and "'make'" in violations[0]

    def test_allow_listed_sites_are_keyed_to_a_function_and_a_reason(self, gate):
        """An allow-list entry without a reason rots into a blanket exemption."""
        assert gate._ALLOWED, "the allow-list must not be empty"
        for path, entries in gate._ALLOWED.items():
            assert entries, f"{path}: allow-listed with no entries"
            assert (_REPO_ROOT / path).exists(), f"{path}: allow-listed file is gone"
            for func, reason in entries:
                assert func, f"{path}: allow-listed with no enclosing function"
                assert len(reason) > 40, f"{path}:{func}: allow-listed without a real reason"


class TestTheAllowListIsClassQualified:
    """Plan row ``D05#8``: a bare method name sanctions that name on EVERY class.

    ``_ALLOWED`` keyed the enclosing function by ``node.name``, so the entry
    ``"build"`` for ``infrastructure/builders/leaf/data_builders.py`` covered both
    ``DatasetBuilder.build`` (:165) and ``DataLoaderBuilder.build`` (:404). Its own
    reason string named only the second, so the allow-list *documented* one method
    and *exempted* two -- and a loader constructed in ``DatasetBuilder.build`` would
    have been accepted in silence.

    This is the other half of the gate #1362 fixed: **row 8 keys the allow-list,
    #1362 keys the match.** Fixing one and not the other leaves a hole either way.

    Every plant below runs end-to-end through ``find_unsanctioned``, and the
    mutation at the bottom restores the pre-fix resolver and watches the plant go
    UNCAUGHT -- a plant no mutation kills is not a demonstration.
    """

    _REL = "src/mriforge/infrastructure/rogue.py"

    #: Two classes, same method name, one construction each -- the real shape of
    #: ``data_builders.py`` reduced to its essentials.
    _TWO_CLASSES = (
        "from torch.utils.data import DataLoader\n"
        "class DatasetBuilder:\n"
        "    def build(self):\n"
        "        return DataLoader(ds)\n"
        "class DataLoaderBuilder:\n"
        "    def build(self):\n"
        "        return DataLoader(ds)\n"
    )

    _REASON = "planted allow-list entry, long enough to satisfy the reason contract"

    def test_the_premise_still_holds_in_the_real_file(self) -> None:
        """Guard the row's premise, not just the fix.

        If ``data_builders.py`` ever collapses to one ``build``, the plants below
        still pass while testing a shape the repo no longer has -- and the reader
        would have no way to tell.
        """
        import ast

        tree = ast.parse(
            (
                _REPO_ROOT
                / "src"
                / "mriforge"
                / "infrastructure"
                / "builders"
                / "leaf"
                / "data_builders.py"
            ).read_text(encoding="utf-8")
        )
        owners = sorted(
            cls.name
            for cls in ast.walk(tree)
            if isinstance(cls, ast.ClassDef)
            and any(
                isinstance(m, ast.FunctionDef | ast.AsyncFunctionDef) and m.name == "build"
                for m in cls.body
            )
        )
        assert owners == ["DataLoaderBuilder", "DatasetBuilder"], owners

    def test_the_two_sites_report_different_names(self, gate) -> None:
        assert sorted(gate.find_construction_sites(self._TWO_CLASSES)) == [
            (4, "DatasetBuilder.build"),
            (7, "DataLoaderBuilder.build"),
        ]

    def test_only_the_named_class_is_sanctioned(self, gate, tmp_path, monkeypatch) -> None:
        """The plant: sanction ``DataLoaderBuilder.build``, and the sibling is caught."""
        root = _plant(tmp_path, self._TWO_CLASSES)
        monkeypatch.setitem(gate._ALLOWED, self._REL, (("DataLoaderBuilder.build", self._REASON),))
        violations, matched = gate.find_unsanctioned(root)
        assert len(violations) == 1, violations
        assert "DatasetBuilder.build" in violations[0]
        assert (self._REL, "DataLoaderBuilder.build") in matched

    def test_a_stale_bare_key_fails_loud_instead_of_broad(
        self, gate, tmp_path, monkeypatch
    ) -> None:
        """A leftover bare ``build`` entry now matches NOTHING, not everything.

        The direction matters more than the count: the mis-spelled entry
        over-reports (two violations plus a stale-entry report) rather than
        silently widening, so the failure mode of getting this wrong is a red
        gate, never a quiet exemption.
        """
        root = _plant(tmp_path, self._TWO_CLASSES)
        monkeypatch.setitem(gate._ALLOWED, self._REL, (("build", self._REASON),))
        violations, matched = gate.find_unsanctioned(root)
        assert len(violations) == 2, violations
        assert (self._REL, "build") not in matched
        assert any("matched no construction site" in s for s in gate.find_stale_entries(matched))

    @pytest.mark.parametrize(
        ("shape", "source", "expected"),
        [
            (
                "module-level function keeps a bare name",
                "from torch.utils.data import DataLoader\ndef make():\n    return DataLoader(ds)\n",
                "make",
            ),
            (
                "class nested in a function qualifies through both",
                "from torch.utils.data import DataLoader\n"
                "def outer():\n"
                "    class Inner:\n"
                "        def make(self):\n"
                "            return DataLoader(ds)\n",
                "outer.Inner.make",
            ),
            (
                "def guarded by an if keeps its class prefix",
                "from torch.utils.data import DataLoader\n"
                "import sys\n"
                "class B:\n"
                "    if sys.version_info:\n"
                "        def make(self):\n"
                "            return DataLoader(ds)\n",
                "B.make",
            ),
            (
                "async method",
                "from torch.utils.data import DataLoader\n"
                "class B:\n"
                "    async def make(self):\n"
                "        return DataLoader(ds)\n",
                "B.make",
            ),
        ],
    )
    def test_the_qualified_name_for_each_nesting_shape(self, gate, shape, source, expected) -> None:
        """One case per shape the enclosing path can take (non-negotiable 15)."""
        sites = gate.find_construction_sites(source)
        assert [name for _, name in sites] == [expected], shape

    def test_reverting_to_bare_names_makes_the_plant_go_uncaught(
        self, gate, tmp_path, monkeypatch
    ) -> None:
        """The mutation. Restore the pre-``D05#8`` resolver and the hole reopens."""
        import ast

        def bare(tree):
            spans = [
                (n.lineno, n.end_lineno or n.lineno, n.name)
                for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
            ]
            spans.sort(key=lambda s: s[1] - s[0], reverse=True)
            owner: dict[int, str] = {}
            for start, end, name in spans:
                for line in range(start, end + 1):
                    owner[line] = name
            return owner

        monkeypatch.setattr(gate, "_enclosing_functions", bare)
        root = _plant(tmp_path, self._TWO_CLASSES)
        monkeypatch.setitem(gate._ALLOWED, self._REL, (("build", self._REASON),))
        violations, _ = gate.find_unsanctioned(root)
        assert violations == [], (
            "the pre-D05#8 resolver is expected to accept BOTH classes; it caught "
            "something, so this plant does not demonstrate the fix"
        )
