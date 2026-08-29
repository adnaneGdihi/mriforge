"""
TASK III.2 – Training-loop performance discipline fitness function.

Enforces CLAUDE.md "Training-loop performance rules":

    No ``.item()``, ``.cpu()``, ``.tolist()``, ``.numpy()``, or
    ``torch.cuda.empty_cache()`` inside the bodies of methods named
    ``training_step``, ``train_step``, ``optimizer_step``, or
    ``compute_loss`` in files under
    ``mriforge/infrastructure/training/strategies/``.

These calls synchronise the GPU and kill throughput.

AST detection: the test walks into *method bodies* specifically — docstrings
that mention ``.item()`` are not flagged (they don't generate call nodes).

Gate test (fast, always runs):
    Fail on any violation not in _known_violations.json["training_loop"].

Cleanup tracker (slow, opt-in with -m slow):
    Fails until all known violations are fixed.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.parent
STRATEGIES_ROOT = REPO_ROOT / "src" / "mriforge" / "infrastructure" / "training" / "strategies"
VIOLATIONS_FILE = Path(__file__).parent / "_known_violations.json"

# The entry point of the per-step path. Everything the base strategy calls from
# here, transitively, IS the training loop.
STEP_ENTRY_POINT = "train_step"


def _derive_target_methods() -> frozenset[str]:
    """Methods on the per-step path, read off ``base.py``'s own call graph.

    This set used to be enumerated: ``{training_step, train_step,
    optimizer_step, compute_loss}``. Three of those four names were defined
    ZERO times anywhere under ``strategies/`` -- the audit was of a vocabulary,
    not of the code -- while the real per-step hook ``_compute_losses_impl``
    (144 definers, versus ``train_step``'s 16) was never looked at. A gate that
    covers ~10% of its own subject is the shape non-negotiable #15 exists to
    stop, and no amount of running it green would have revealed that.

    So derive instead: start at ``train_step`` on the base strategy and walk
    ``self.X(...)`` transitively through the methods ``base.py`` itself defines.
    Anything reachable runs once per step by construction, and a new hook added
    to the template is audited the day it is added rather than the day someone
    remembers to extend a literal.

    ``validation_step`` is deliberately NOT a root: it is not defined in
    ``base.py``, and the rule this enforces is about the *training* loop. Adding
    it is a separate decision with a much larger allow-list, not a free widening.
    """
    tree = ast.parse((STRATEGIES_ROOT / "base.py").read_text(encoding="utf-8"))
    defined: dict[str, set[str]] = {}
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        for m in cls.body:
            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined[m.name] = {
                    n.func.attr
                    for n in ast.walk(m)
                    if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and isinstance(n.func.value, ast.Name)
                    and n.func.value.id == "self"
                }
    if STEP_ENTRY_POINT not in defined:
        # Absent is a state to REPORT, never one to infer (non-negotiable #18).
        # Silently returning {} here would make every violation vanish and the
        # gate print green -- the exact failure this rewrite exists to remove.
        raise RuntimeError(
            f"{STEP_ENTRY_POINT!r} is not defined in {STRATEGIES_ROOT / 'base.py'}. "
            "The per-step entry point was renamed; update STEP_ENTRY_POINT rather "
            "than letting the audited set silently empty out."
        )
    reached = {STEP_ENTRY_POINT}
    frontier = [STEP_ENTRY_POINT]
    while frontier:
        for callee in defined.get(frontier.pop(), ()):
            if callee in defined and callee not in reached:
                reached.add(callee)
                frontier.append(callee)
    return frozenset(reached)


# Methods whose bodies are audited
TARGET_METHODS: frozenset[str] = _derive_target_methods()

# Banned attribute names (accessed as method calls, e.g. tensor.item())
BANNED_ATTRS: frozenset[str] = frozenset({"item", "cpu", "tolist", "numpy"})

# Calls whose result forces a device->host transfer the moment it is used as a
# Python truth value -- ``D12#5``.  ``.any()``/``.all()`` return a 0-dim tensor
# and the predicates return a bool tensor; either one in an ``if`` test makes
# the branch wait on the GPU.  ``_BannedCallFinder`` cannot see these: no
# ``.item()`` is written anywhere.
IMPLICIT_SYNC_ATTRS: frozenset[str] = frozenset({"any", "all", "isnan", "isinf", "isfinite"})

# The same predicates reached as bare names (``from torch import isnan``).
# ``any``/``all`` are deliberately NOT here: as a bare name they are the Python
# builtins over a generator, which never touch a tensor.  Three sites in the
# audited methods rely on that distinction -- ``all(k in params for k in ...)``
# in ``cycle_bloch_digital_twin_strategy`` (twice) and an ``any(isinstance(...))``
# in ``domain_adaptation`` -- and flagging them would be a loud wrong answer.
IMPLICIT_SYNC_NAMES: frozenset[str] = frozenset({"isnan", "isinf", "isfinite"})

# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _forced_sync_in_test(test: ast.expr) -> str | None:
    """Name the call that forces ``test`` to a Python bool, or ``None``.

    Returns at most **one** label per ``if``, deliberately.
    ``torch.isnan(x).any() or torch.isinf(x).any()`` is one branch and one sync
    point; its four matching calls counted separately would inflate the debt
    report from 4 sites to 7.  ``ast.walk`` is breadth-first, so the first hit
    is the outermost forcing call -- ``any()`` for that expression, ``isfinite``
    for ``not torch.isfinite(total)`` -- which is the one actually booled.
    """
    for node in ast.walk(test):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr in IMPLICIT_SYNC_ATTRS:
            return f"if {node.func.attr}()"
        if isinstance(node.func, ast.Name) and node.func.id in IMPLICIT_SYNC_NAMES:
            return f"if {node.func.id}()"
    return None


class _ImplicitSyncFinder(ast.NodeVisitor):
    """Collect (label, lineno) for ``if`` tests that force a tensor to bool.

    Scoped to ``ast.If`` as the row specifies.  ``while`` and ``assert`` sync
    the same way but are a separate decision with a separate blast radius, and
    an ``x if c else y`` ternary is ``ast.IfExp`` -- none are widened into here
    without their own measurement.  Each ``elif`` is its own ``ast.If`` and so
    is reported separately, which is correct: each is a distinct branch.
    """

    def __init__(self) -> None:
        self.found: list[tuple[str, int]] = []

    def visit_If(self, node: ast.If) -> None:
        label = _forced_sync_in_test(node.test)
        if label is not None:
            self.found.append((label, node.lineno))
        self.generic_visit(node)


class _BannedCallFinder(ast.NodeVisitor):
    """Collect (attr_name, lineno) for banned calls inside a node."""

    def __init__(self) -> None:
        self.found: list[tuple[str, int]] = []

    def visit_Call(self, node: ast.Call) -> None:
        # tensor.item(), tensor.cpu(), etc.
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in BANNED_ATTRS:
                self.found.append((node.func.attr + "()", node.lineno))
            # torch.cuda.empty_cache()
            if node.func.attr == "empty_cache":
                self.found.append(("torch.cuda.empty_cache()", node.lineno))
        self.generic_visit(node)


def _scan_violations(root: Path | None = None) -> list[dict[str, str | int]]:
    """Return all training-loop discipline violations found via AST.

    ``root`` overrides the strategies directory so a violation can be planted on
    a throwaway tree and the gate watched going red (non-negotiable #15).
    """
    root = STRATEGIES_ROOT if root is None else root
    violations: list[dict[str, str | int]] = []
    for py_file in sorted(root.rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue
        try:
            # Real tree: keep the src-relative key the allow-list is written in.
            rel = str(py_file.relative_to(REPO_ROOT / "src"))
        except ValueError:  # a planted tree lives outside the repo
            rel = str(py_file.relative_to(root))
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name not in TARGET_METHODS:
                continue
            finder = _BannedCallFinder()
            implicit = _ImplicitSyncFinder()
            # Visit only the *body* of the method (not nested class defs)
            for child in node.body:
                finder.visit(child)
                implicit.visit(child)
            for call, lineno in finder.found + implicit.found:
                violations.append(
                    {
                        "file": rel,
                        "method": node.name,
                        "call": call,
                        "line": lineno,
                    }
                )
    return violations


# ---------------------------------------------------------------------------
# Allowlist helpers
# ---------------------------------------------------------------------------


def _load_known() -> list[dict]:
    if not VIOLATIONS_FILE.exists():
        return []
    data = json.loads(VIOLATIONS_FILE.read_text())
    return data.get("training_loop", [])


def _vkey(v: dict) -> tuple[str, str, str]:
    # (file, method, call) — line numbers shift, so we don't include them
    return (v["file"], v["method"], v["call"])


def _allowance(v: dict) -> int:
    """How many occurrences this entry forgives. Absent ``count`` means one."""
    return int(v.get("count", 1))


def _new_only(found: list[dict], known: list[dict]) -> list[dict]:
    """Violations beyond what the allow-list records.

    Keyed on identity AND COUNT. Identity alone leaves the hole non-negotiable
    #20 names for the LOC ratchet: growth inside an already-recorded entry is
    invisible, so a second ``.item()`` added to a method that already has one
    lands green forever. The allow-list forgives the occurrences it recorded,
    not the method.
    """
    budget: dict[tuple[str, str, str], int] = {}
    for k in known:
        budget[_vkey(k)] = budget.get(_vkey(k), 0) + _allowance(k)
    new: list[dict] = []
    for v in sorted(found, key=lambda x: (x["file"], x["method"], x["line"])):
        if budget.get(_vkey(v), 0) > 0:
            budget[_vkey(v)] -= 1
        else:
            new.append(v)
    return new


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.architecture
def test_no_new_training_loop_gpu_sync_calls() -> None:
    """Gate: fail if a NEW GPU-sync call appears in a training-step method.

    Pre-existing violations are tracked in
    ``tests/architecture/_known_violations.json["training_loop"]``.

    Allowed exception: calls in helper methods that are NOT themselves named
    ``train_step`` / ``training_step`` / ``optimizer_step`` / ``compute_loss``
    are not checked here.  Move metric-extraction logic to a ``_metrics()``
    helper to keep the hot path clean.
    """
    found = _scan_violations()
    known = _load_known()
    new = _new_only(found, known)
    if new:
        lines = [
            "NEW GPU-sync calls in training-step methods "
            '(not in _known_violations.json["training_loop"]):'
        ]
        for v in new:
            lines.append(f"  {v['file']}::{v['method']}:{v['line']}  ->  {v['call']}")
        lines.append(
            "\nFix: extract scalar logging into a post-step helper that runs "
            "outside the hot path.  See CLAUDE.md training-loop rules."
        )
        pytest.fail("\n".join(lines))


@pytest.mark.architecture
def test_training_loop_allowlist_has_no_stale_entries() -> None:
    """Hard gate: every recorded exemption must still describe a real sync call.

    A stale entry makes the allowlist lie about the codebase and silently
    re-exempts the method if a ``.item()`` reappears there. Actionable today, so
    unlike the debt report below this stays a hard failure (#629).
    """
    found = {_vkey(v) for v in _scan_violations()}
    stale = [v for v in _load_known() if _vkey(v) not in found]
    assert not stale, (
        f"{len(stale)} stale training_loop entries in _known_violations.json "
        "(sync call no longer present — remove them):\n"
        + "\n".join(f"  {v['file']}::{v['method']}  {v['call']}" for v in stale)
    )


@pytest.mark.architecture
@pytest.mark.slow
@pytest.mark.debt_tracker
@pytest.mark.xfail(
    strict=False,
    reason="Debt report: red until every recorded GPU-sync call is out of the hot "
    "path. XPASS means the debt is paid — delete this marker.",
)
def test_no_known_training_loop_violations_remain() -> None:
    """Debt report: how many recorded GPU-sync violations are left.

    Run ``pytest -m debt_tracker -rx`` to read them.
    """
    found = _scan_violations()
    known = _load_known()
    still_present = [v for v in known if any(_vkey(v) == _vkey(f) for f in found)]
    if still_present:
        msg = (
            f"{len(still_present)} known training-loop violations still present "
            "(remove from _known_violations.json once fixed):\n"
            + "\n".join(f"  {v['file']}::{v['method']}  {v['call']}" for v in still_present)
        )
        pytest.fail(msg)


# ---------------------------------------------------------------------------
# Planted violations (non-negotiable #15)
# ---------------------------------------------------------------------------
#
# Until 2026-08 this detector had none, and that is precisely how it stayed
# green for months while auditing three method names that were defined nowhere.
# Every rule below is watched FAILING on a tree built for the purpose: one plant
# per banned call, one per shape the derivation is supposed to have unlocked,
# and one negative per way the gate could go loud-wrong.


# Assembled rather than written literally. ``tests/conftest.py`` auto-marks a
# test ``gpu`` when the pattern ``torch.cuda.empty_cache`` appears anywhere in
# its source OR its enclosing CLASS's source, and the CPU lane then skips it.
# Spelling the banned call out here would silently disable every plant in the
# class below — a detector test for a CUDA rule cannot contain the CUDA rule.
_EMPTY_CACHE = "torch.cuda." + "empty_cache()"


def _plant_strategy(tmp_path: Path, body: str, name: str = "planted_strategy.py") -> Path:
    """Write a throwaway strategy module and return the root to scan."""
    (tmp_path / name).write_text(body, encoding="utf-8")
    return tmp_path


class TestDetectorFiresOnPlantedViolations:
    """A gate nobody has watched fail is not a gate."""

    @pytest.mark.architecture
    @pytest.mark.parametrize(
        ("call", "line"),
        [
            ("item()", "        x = loss.item()"),
            ("cpu()", "        x = loss.cpu()"),
            ("tolist()", "        x = loss.tolist()"),
            ("numpy()", "        x = loss.numpy()"),
            (_EMPTY_CACHE, f"        {_EMPTY_CACHE}"),
        ],
    )
    def test_each_banned_call_is_caught(self, tmp_path, call, line) -> None:
        root = _plant_strategy(tmp_path, f"class S:\n    def train_step(self):\n{line}\n")
        found = _scan_violations(root)
        assert [v["call"] for v in found] == [call]

    @pytest.mark.architecture
    def test_the_real_per_step_hook_is_audited(self, tmp_path) -> None:
        """The regression pin for the whole rewrite.

        ``_compute_losses_impl`` is defined in **134 files** (145 definitions)
        under ``strategies/`` -- re-measured 2026-08-23 -- and was
        invisible to the enumerated set. If ``TARGET_METHODS`` ever reverts to a
        literal that omits it, this plant goes green and the test fails.
        """
        root = _plant_strategy(
            tmp_path,
            "class S:\n    def _compute_losses_impl(self):\n        return loss.item()\n",
        )
        found = _scan_violations(root)
        assert len(found) == 1, "the real per-step hook is not being audited"
        assert found[0]["method"] == "_compute_losses_impl"

    @pytest.mark.architecture
    def test_a_method_off_the_per_step_path_is_not_audited(self, tmp_path) -> None:
        """Negative control: setup code may sync freely, and must not be flagged.

        Without this, "audit everything" would pass every positive plant above
        while making the gate useless.
        """
        root = _plant_strategy(
            tmp_path,
            "class S:\n"
            "    def _setup_strategy_specific_components(self):\n"
            "        return cfg.item()\n",
        )
        assert _scan_violations(root) == []

    @pytest.mark.architecture
    def test_docstring_mention_is_not_a_call(self, tmp_path) -> None:
        root = _plant_strategy(
            tmp_path,
            'class S:\n    def train_step(self):\n        """Never call .item() here."""\n',
        )
        assert _scan_violations(root) == []

    @pytest.mark.architecture
    def test_a_second_occurrence_in_a_recorded_method_is_new(self) -> None:
        """The NN20 hole: identity alone forgives a method, not an occurrence.

        One recorded ``.item()`` must not pre-forgive the next one added beside
        it. Without the count, this is the shape that lands green forever.
        """
        known = [{"file": "f.py", "method": "train_step", "call": "item()", "count": 1}]
        twice = [
            {"file": "f.py", "method": "train_step", "call": "item()", "line": 10},
            {"file": "f.py", "method": "train_step", "call": "item()", "line": 11},
        ]
        assert len(_new_only(twice, known)) == 1
        assert _new_only(twice[:1], known) == []

    @pytest.mark.architecture
    def test_entry_without_count_still_forgives_one(self) -> None:
        """Back-compat: the schema predates ``count`` and older entries omit it."""
        known = [{"file": "f.py", "method": "train_step", "call": "item()"}]
        one = [{"file": "f.py", "method": "train_step", "call": "item()", "line": 10}]
        assert _new_only(one, known) == []


class TestTheImplicitSyncVisitorFiresOnPlantedViolations:
    """``D12#5``: the syncs that no ``.item()`` spelling makes visible.

    Every rule the visitor claims is watched failing here, and every shape that
    rule can take -- attribute ``.any()``/``.all()``, the ``torch.isnan``
    predicate family, the ``or``-chained pair, and ``elif``. The negatives
    matter as much: this visitor's one way to be loud-wrong is to flag the
    Python builtins, and three real sites in the audited methods would break.
    """

    @pytest.mark.architecture
    @pytest.mark.parametrize(
        ("expected", "test_src"),
        [
            ("if any()", "flags.any()"),
            ("if all()", "flags.all()"),
            ("if isnan()", "torch.isnan(x)"),
            ("if isinf()", "torch.isinf(x)"),
            ("if isfinite()", "not torch.isfinite(x)"),
            # The bare-name predicate form (``from torch import isnan``). No
            # strategy writes it today; it is matched so that adding the import
            # cannot quietly reopen the hole.
            ("if isnan()", "isnan(x)"),
        ],
    )
    def test_each_truth_forcing_call_is_caught(self, tmp_path, expected, test_src) -> None:
        root = _plant_strategy(
            tmp_path,
            f"class S:\n    def _compute_losses_impl(self):\n        if {test_src}:\n            pass\n",
        )
        found = _scan_violations(root)
        assert [v["call"] for v in found] == [expected]

    @pytest.mark.architecture
    def test_one_violation_per_if_not_per_matching_call(self, tmp_path) -> None:
        """The real ``graph_cold_diffusion_strategy`` shape, in miniature.

        ``torch.isnan(p).any() or torch.isinf(p).any()`` matches four times and
        is **one** branch and one sync point. Counting calls would have made the
        debt report read 7 sites where the tree has 4 -- a detector that
        overstates is as untrustworthy as one that misses.
        """
        root = _plant_strategy(
            tmp_path,
            "class S:\n    def _compute_losses_impl(self):\n"
            "        if torch.isnan(p).any() or torch.isinf(p).any():\n            pass\n",
        )
        found = _scan_violations(root)
        assert len(found) == 1, f"expected one violation per if-test, got {found}"
        assert found[0]["call"] == "if any()"

    @pytest.mark.architecture
    def test_each_elif_is_its_own_branch(self, tmp_path) -> None:
        """An ``elif`` is a separate ``ast.If`` and a separate sync -- both count."""
        root = _plant_strategy(
            tmp_path,
            "class S:\n    def _compute_losses_impl(self):\n"
            "        if a.any():\n            pass\n"
            "        elif b.all():\n            pass\n",
        )
        found = _scan_violations(root)
        assert [v["call"] for v in found] == ["if any()", "if all()"]

    @pytest.mark.architecture
    @pytest.mark.parametrize(
        "test_src",
        [
            # The Python builtins over a generator: no tensor is touched.
            # ``cycle_bloch_digital_twin_strategy`` has two of these and
            # ``domain_adaptation`` one; flagging them is the loud-wrong answer.
            "all(k in params for k in ('M0', 'T1', 'T2'))",
            "any(isinstance(v, dict) for v in batch.values())",
        ],
    )
    def test_the_builtin_over_a_generator_is_not_a_sync(self, tmp_path, test_src) -> None:
        root = _plant_strategy(
            tmp_path,
            f"class S:\n    def _compute_losses_impl(self):\n        if {test_src}:\n            pass\n",
        )
        assert _scan_violations(root) == []

    @pytest.mark.architecture
    @pytest.mark.parametrize(
        "src",
        [
            # Outside an ``if`` test: assigning the result syncs nothing until
            # something booleans it, and the row scopes the rule to ``ast.If``.
            "        bad = flags.any()\n",
            # ``while`` and the ``IfExp`` ternary sync the same way but are a
            # separate decision with their own blast radius -- not widened into
            # here without their own measurement.
            "        while flags.any():\n            break\n",
            "        y = 1 if flags.any() else 2\n",
        ],
    )
    def test_the_rule_does_not_reach_past_its_scope(self, tmp_path, src) -> None:
        root = _plant_strategy(tmp_path, f"class S:\n    def _compute_losses_impl(self):\n{src}")
        assert _scan_violations(root) == []

    @pytest.mark.architecture
    def test_a_method_off_the_per_step_path_is_not_audited(self, tmp_path) -> None:
        """Negative control: setup code may boolean a tensor freely."""
        root = _plant_strategy(
            tmp_path,
            "class S:\n    def build_dataset(self):\n        if flags.any():\n            pass\n",
        )
        assert _scan_violations(root) == []

    @pytest.mark.architecture
    def test_removing_the_visitor_makes_the_plant_go_uncaught(self, tmp_path, monkeypatch) -> None:
        """Mutation: prove the plants above are caught by *this* visitor.

        Without this, a plant that some other rule happens to catch reads as
        evidence for a visitor that could be doing nothing.
        """
        monkeypatch.setattr(sys.modules[__name__], "IMPLICIT_SYNC_ATTRS", frozenset(), raising=True)
        monkeypatch.setattr(sys.modules[__name__], "IMPLICIT_SYNC_NAMES", frozenset(), raising=True)
        root = _plant_strategy(
            tmp_path,
            "class S:\n    def _compute_losses_impl(self):\n        if flags.any():\n            pass\n",
        )
        assert _scan_violations(root) == [], "the plant is being caught by some other rule"


class TestTargetMethodDerivation:
    """The audited set is read off the code, not typed into this file."""

    @pytest.mark.architecture
    def test_derivation_reaches_the_documented_hook(self) -> None:
        assert "_compute_losses_impl" in TARGET_METHODS
        assert STEP_ENTRY_POINT in TARGET_METHODS

    @pytest.mark.architecture
    def test_derivation_is_wider_than_the_literal_it_replaced(self) -> None:
        """The three names that were audited while being defined zero times.

        Keeping them would not be harmless -- it is what made the set look
        maintained. They are gone because nothing defines them, and the
        assertion records that rather than trusting it.
        """
        retired = {"training_step", "optimizer_step", "compute_loss"}
        defined = {
            node.name
            for py in STRATEGIES_ROOT.rglob("*.py")
            if "__pycache__" not in str(py)
            for node in ast.walk(ast.parse(py.read_text(encoding="utf-8")))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert not (retired & defined), (
            "a retired name is defined again — re-run the derivation reasoning"
        )
        assert len(TARGET_METHODS) > 4

    @pytest.mark.architecture
    def test_a_renamed_entry_point_raises_instead_of_emptying(self, tmp_path, monkeypatch) -> None:
        """Absent is a state to REPORT, never one to infer (#18).

        If ``train_step`` is renamed and the derivation silently returns the
        empty set, every violation disappears and the gate prints green — a
        strictly worse failure than the one being fixed.
        """
        (tmp_path / "base.py").write_text(
            "class BaseTrainingStrategy:\n    def step(self):\n        pass\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(sys.modules[__name__], "STRATEGIES_ROOT", tmp_path, raising=True)
        with pytest.raises(RuntimeError, match="not defined in"):
            _derive_target_methods()
