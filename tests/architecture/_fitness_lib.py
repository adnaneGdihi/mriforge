"""Shared helpers for the architecture fitness functions (cached-cascade Phase 0).

AST-level scanners + a ratchet-baseline mechanism. Each fitness test reports the
*current* set of offenders, compares against a committed baseline under
``tests/architecture/baselines/``, and fails only on NEW offenders (regressions)
-- the same ratchet the layering gate uses, so a guard can be added to a large
legacy tree without a flag-day refactor.

Regenerate baselines after an intentional change::

    SPECTRAMR_UPDATE_ARCH_BASELINE=1 pytest tests/architecture/ -q

Not a test module (underscore-prefixed) -- pytest does not collect it.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

# tests/architecture/_fitness_lib.py -> parents[2] == repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "spectramr"
BASELINE_DIR = Path(__file__).resolve().parent / "baselines"

# Canonical step signatures (regular param names, excluding self/**kwargs).
CANONICAL_TRAIN_STEP = ("batch", "epoch", "input_batch", "target_batch")
CANONICAL_VALIDATION_STEP = ("input_batch", "target_batch")

# Paths that are legitimate dispatch homes (registries / factories) -- a string
# dispatch table there is the SSOT, not a smell.
_DISPATCH_HOME_MARKERS = (
    "registry",
    "scheduler_system",
    "strategy_factory",
    "/enums.py",
)


def should_update_baseline() -> bool:
    return os.environ.get("SPECTRAMR_UPDATE_ARCH_BASELINE") == "1"


def should_raise_ceiling() -> bool:
    """Whether a regeneration may record a HIGHER measurement than the last one.

    Deliberately a second, separate flag. ``SPECTRAMR_UPDATE_ARCH_BASELINE=1``
    accepts everything currently present, so a routine re-baseline — run to
    record one new offender — silently raises the ceiling on every other entry
    that grew since. 107 entries would move up today. Non-negotiable 20 says a
    baseline regeneration must not raise a ceiling; this is the mechanism that
    makes that true by default rather than by reviewer vigilance.
    """
    return os.environ.get("SPECTRAMR_RAISE_ARCH_CEILING") == "1"


def iter_src_files() -> list[Path]:
    return sorted(p for p in SRC_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return None


# --------------------------------------------------------------------------- #
# Baseline IO (sorted text, one entry per line -- easy to diff)
# --------------------------------------------------------------------------- #
def load_baseline(name: str) -> set[str]:
    path = BASELINE_DIR / name
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def write_baseline(name: str, entries: set[str], header: str = "") -> None:
    """Persist ``entries`` as ``<identity>  # <measurement>`` lines.

    Demoting the measurement to a trailing comment is not cosmetic: it is what
    stops anyone re-introducing the equality bug the ``#`` makes obvious.
    """
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    path = BASELINE_DIR / name
    body = "\n".join(sorted(_as_baseline_line(e) for e in entries))
    prefix = f"# {header}\n" if header else ""
    path.write_text(prefix + body + ("\n" if body else ""), encoding="utf-8")


# A trailing ``(<int> <word>)`` / ``(<word> <int>)`` group -- "(371 loc)",
# "(5 branches)", "(depth 3)". Deliberately does NOT match a param tuple such
# as "('batch', 'epoch')", where the parenthesised part IS the identity.
_MEASUREMENT_RE = re.compile(r"\s*\((?:\d+\s+\w+|\w+\s+\d+)\)$")
# The same measurement once demoted to a trailing comment by write_baseline.
_TRAILING_COMMENT_RE = re.compile(r"\s*#.*$")
# ...and that comment form read back as a measurement: "foo.py  # 371 loc".
_COMMENT_MEASUREMENT_RE = re.compile(r"\s*#\s*(?:\d+\s+\w+|\w+\s+\d+)\s*$")


def _as_baseline_line(entry: str) -> str:
    """``foo.py (371 loc)`` -> ``foo.py  # 371 loc``; other entries unchanged."""
    m = _MEASUREMENT_RE.search(entry)
    if not m:
        return entry
    return f"{entry[: m.start()]}  # {m.group().strip()[1:-1]}"


def baseline_identity(entry: str) -> str:
    """The stable half of a baseline entry, used as the ratchet key.

    Entries are ``<identity>`` plus a measurement -- a count that moves on any
    ordinary edit. Keying the ratchet on the whole string made every
    long-standing offender re-report as NEW the moment anyone touched it
    (``bootstrap.py (336 loc)`` -> ``(371 loc)``), so the gate could not stay
    green through normal development -- see issue #629. Accepts both the
    scanner's ``(371 loc)`` form and the baseline file's ``# 371 loc`` form.
    Signature-drift entries carry a param tuple rather than a count and are left
    whole, because there the params ARE the thing being watched.
    """
    return _MEASUREMENT_RE.sub("", _TRAILING_COMMENT_RE.sub("", entry)).strip()


def baseline_measurement(entry: str) -> int | None:
    """The integer a baseline entry records, or ``None`` if it records none.

    ``foo.py  # 371 loc`` -> ``371``; ``foo.py (371 loc)`` -> ``371``;
    ``Cls (depth 3)`` -> ``3``; a signature-drift entry -> ``None``.
    """
    m = _MEASUREMENT_RE.search(entry) or _COMMENT_MEASUREMENT_RE.search(entry)
    if m is None:
        return None
    digits = [tok for tok in m.group().strip("() #").split() if tok.isdigit()]
    return int(digits[0]) if digits else None


def stale_entries(name: str, current: set[str]) -> set[str]:
    """Baselined identities that are no longer offenders at all.

    A stale entry is not harmless bookkeeping: because :func:`ratchet` decides
    membership on identity alone, an entry naming a **deleted** file keeps that
    path pre-exempted, so re-creating it at 900 LOC lands green. Three of
    ``large_files.txt``'s entries name directors deleted in the builder
    consolidation, which is exactly that hole standing open.

    Mirrors ``test_data_io_allowlist_has_no_stale_entries``, which has been a
    hard gate on the sibling allowlist since #629; the ratchet baselines never
    grew one.
    """
    return stale_identities(load_baseline(name), current)


def stale_identities(baselined: set[str], current: set[str]) -> set[str]:
    """Pure core of :func:`stale_entries` — no baseline IO, so it can be planted."""
    live = {baseline_identity(e) for e in current}
    return {baseline_identity(e) for e in baselined} - live


def grown_entries(
    name: str, current: set[str], slack: int = 0
) -> dict[str, tuple[int, int]]:
    """``identity -> (recorded, now)`` for offenders that got worse since baselining.

    The ratchet keys on identity alone, deliberately: keying on the whole string
    made every long-standing offender re-report as NEW the moment anyone touched
    it (``bootstrap.py (336 loc)`` -> ``(371 loc)``), so the gate could not stay
    green through ordinary development (#629). The cost of that correct fix is
    that growth *inside* an already-baselined entry is invisible, which is what
    non-negotiable 20 names.

    This is report-only, and that is not timidity. A hard gate against the
    recorded values is a 107-file flag day; a hard gate against *refreshed*
    values would first have to refresh the measurements, which is the ceiling
    raise NN20's review rule tells readers to reject — and would then re-create
    #629 exactly. Report-only is the only tightening available today that is
    lawful in both directions.
    """
    return grown_measurements(load_baseline(name), current, slack=slack)


def grown_measurements(
    baselined: set[str], current: set[str], slack: int = 0
) -> dict[str, tuple[int, int]]:
    """Pure core of :func:`grown_entries` — no baseline IO, so it can be planted.

    ``slack`` is a FLAT allowance, never proportional. `00_MASTER.md` §5 is
    explicit about why: a 10 % rule on a 12,745-LOC file waves through ~1,275
    lines, which would have permitted the exact +481 growth the row exists to
    catch. 25 for LOC, 0 for counts.
    """
    recorded = {}
    for entry in baselined:
        m = baseline_measurement(entry)
        if m is not None:
            recorded[baseline_identity(entry)] = m
    out: dict[str, tuple[int, int]] = {}
    for entry in current:
        ident = baseline_identity(entry)
        now = baseline_measurement(entry)
        if now is None or ident not in recorded:
            continue
        if now - recorded[ident] > slack:
            out[ident] = (recorded[ident], now)
    return out


def with_measurement(entry: str, value: int) -> str:
    """``src/a.py (400 loc)`` + 300 -> ``src/a.py (300 loc)``.

    Entries with no measurement (signature drift) are returned unchanged.
    """
    m = _MEASUREMENT_RE.search(entry) or _COMMENT_MEASUREMENT_RE.search(entry)
    if m is None:
        return entry
    parts = [
        str(value) if tok.isdigit() else tok
        for tok in m.group().strip("() #").split()
    ]
    return f"{entry[: m.start()]} ({' '.join(parts)})"


def clamp_to_recorded(baselined: set[str], current: set[str]) -> set[str]:
    """``current``, with every grown measurement pulled back to its recorded one.

    The anti-laundering half of non-negotiable 20. Growth is preserved as a
    *finding* (:func:`grown_measurements` still reports it against the clamped
    value); what is refused is quietly writing the larger number down as the new
    ceiling. Shrinkage is recorded as-is, so the ratchet can only tighten.
    """
    recorded = {baseline_identity(e): baseline_measurement(e) for e in baselined}
    out: set[str] = set()
    for entry in current:
        now = baseline_measurement(entry)
        old = recorded.get(baseline_identity(entry))
        if now is None or old is None or now <= old:
            out.add(entry)
        else:
            out.add(with_measurement(entry, old))
    return out


def ratchet(name: str, current: set[str], header: str = "") -> set[str]:
    """Return NEW offenders (current - baseline), updating the baseline if asked.

    Membership is decided on :func:`baseline_identity`, so growing an
    already-baselined file does not read as a new violation -- only a
    previously-unseen offender does. In update mode, writes ``current`` to the
    baseline (with fresh measurements) and returns an empty set.
    """
    if should_update_baseline():
        to_write = (
            current
            if should_raise_ceiling()
            else clamp_to_recorded(load_baseline(name), current)
        )
        write_baseline(name, to_write, header=header)
        return set()
    baselined = {baseline_identity(e) for e in load_baseline(name)}
    return {e for e in current if baseline_identity(e) not in baselined}


# --------------------------------------------------------------------------- #
# Detector 1: string-literal dispatch chains (dispatch-hell)
# --------------------------------------------------------------------------- #
def _is_dispatch_home(path: Path) -> bool:
    s = rel(path).replace("\\", "/").lower()
    return any(m in s for m in _DISPATCH_HOME_MARKERS)


def _string_eq_target(node: ast.Compare) -> str | None:
    """If ``node`` is ``<name/attr> == "literal"`` (or reversed), return the
    normalized target name; else None."""
    if len(node.ops) != 1 or not isinstance(node.ops[0], ast.Eq):
        return None
    left, right = node.left, node.comparators[0]

    def _target(n: ast.expr) -> str | None:
        if isinstance(n, ast.Name):
            return n.id
        if isinstance(n, ast.Attribute):
            return n.attr
        return None

    def _is_str_const(n: ast.expr) -> bool:
        return isinstance(n, ast.Constant) and isinstance(n.value, str)

    if _is_str_const(right):
        return _target(left)
    if _is_str_const(left):
        return _target(right)
    return None


def _iter_qualified_functions(
    node: ast.AST, prefix: str = ""
) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Every function in ``node`` paired with its dotted qualname.

    A bare ``fn.name`` collides across classes -- ``kspace_cold_diffusion_generator.py``
    defines eight ``__init__``s, so ``path::__init__`` cannot say WHICH one is
    baselined and, once the ratchet stopped keying on the branch count, one
    baselined entry would have exempted all of them.
    """
    out: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            out += _iter_qualified_functions(child, f"{prefix}{child.name}.")
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append((f"{prefix}{child.name}", child))
            out += _iter_qualified_functions(child, f"{prefix}{child.name}.")
    return out


def find_string_dispatch(
    tree: ast.Module, min_branches: int = 3
) -> list[tuple[str, int]]:
    """Return (qualname, target_count) for functions containing a string-literal
    dispatch chain of >= ``min_branches`` comparisons against the same target."""
    hits: list[tuple[str, int]] = []
    for qualname, fn in _iter_qualified_functions(tree):
        counts: dict[str, int] = {}
        for cmp_node in ast.walk(fn):
            if isinstance(cmp_node, ast.Compare):
                tgt = _string_eq_target(cmp_node)
                if tgt is not None:
                    counts[tgt] = counts.get(tgt, 0) + 1
        worst = max(counts.values(), default=0)
        if worst >= min_branches:
            hits.append((qualname, worst))
    return hits


def scan_dispatch_hell(min_branches: int = 3) -> set[str]:
    offenders: set[str] = set()
    for path in iter_src_files():
        if _is_dispatch_home(path):
            continue
        tree = _parse(path)
        if tree is None:
            continue
        for fn_name, n in find_string_dispatch(tree, min_branches=min_branches):
            offenders.add(f"{rel(path)}::{fn_name} ({n} branches)")
    return offenders


# --------------------------------------------------------------------------- #
# Detector 2: step-method + builder-__init__ signature drift
# --------------------------------------------------------------------------- #
def _regular_params(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    a = fn.args
    names = [p.arg for p in (*a.posonlyargs, *a.args)]
    if names and names[0] == "self":
        names = names[1:]
    return tuple(names)


def _required_params(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    """Regular params a caller MUST supply (those without a default).

    ``args.defaults`` covers the trailing N of ``posonlyargs + args``; ``self``
    never carries one, so stripping it in :func:`_regular_params` keeps the
    alignment.
    """
    names = _regular_params(fn)
    n_defaults = len(fn.args.defaults)
    return names[: len(names) - n_defaults] if n_defaults else names


def accepts_canonical_call(
    params: tuple[str, ...], required: tuple[str, ...], canonical: tuple[str, ...]
) -> bool:
    """Whether ``strategy.step(*canonical)`` is still a valid call.

    Drift means the canonical call broke, not that the signature grew. The
    field/ULF cohort appends *defaulted* conditioning params
    (``field_strength_target=None``) and forwards through ``super()``, which
    keeps the contract intact; a *required* extra param does not. Comparing
    param tuples for equality conflated the two and flagged 16 compatible
    extensions (#629).
    """
    return params[: len(canonical)] == canonical and len(required) <= len(canonical)


def _decorator_names(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    out: set[str] = set()
    for dec in fn.decorator_list:
        node = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
    return out


def collect_step_methods() -> list[dict[str, object]]:
    """Every train_step/validation_step under strategies/ with its params +
    whether it carries the @accepts_step_io migration marker."""
    strat_root = SRC_ROOT / "infrastructure" / "training" / "strategies"
    out: list[dict[str, object]] = []
    for path in sorted(strat_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = _parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and node.name in (
                "train_step",
                "validation_step",
            ):
                out.append(
                    {
                        "path": rel(path),
                        "method": node.name,
                        "params": _regular_params(node),
                        "required": _required_params(node),
                        "decorated": "accepts_step_io" in _decorator_names(node),
                    }
                )
    return out


def collect_builder_inits() -> list[dict[str, object]]:
    """Every __init__ of a class named *Builder/*Director with its params +
    whether it carries the @accepts_builder_context marker."""
    out: list[dict[str, object]] = []
    for sub in ("infrastructure/training/builders", "infrastructure/builders"):
        root = SRC_ROOT / sub
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            tree = _parse(path)
            if tree is None:
                continue
            for cls in ast.walk(tree):
                if not isinstance(cls, ast.ClassDef):
                    continue
                if not (cls.name.endswith("Builder") or cls.name.endswith("Director")):
                    continue
                for item in cls.body:
                    if (
                        isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and item.name == "__init__"
                    ):
                        out.append(
                            {
                                "path": rel(path),
                                "cls": cls.name,
                                "params": _regular_params(item),
                                "decorated": "accepts_builder_context"
                                in _decorator_names(item),
                            }
                        )
    return out


def step_method_key(rec: dict[str, object]) -> str:
    return f"{rec['path']}::{rec['method']}{rec['params']}"


def builder_key(rec: dict[str, object]) -> str:
    return f"{rec['path']}::{rec['cls']}.__init__{rec['params']}"


# --------------------------------------------------------------------------- #
# Detector 3: structure guard (file LOC; local inheritance depth)
# --------------------------------------------------------------------------- #
def scan_large_files(max_loc: int = 300) -> set[str]:
    offenders: set[str] = set()
    for path in iter_src_files():
        try:
            loc = sum(1 for _ in path.open(encoding="utf-8"))
        except UnicodeDecodeError:
            continue
        if loc > max_loc:
            offenders.add(f"{rel(path)} ({loc} loc)")
    return offenders


def _build_class_base_map() -> dict[str, list[str]]:
    """Map ``module.ClassName`` -> list of base class simple names, across src.

    Only simple-name bases (resolvable within our class universe) are kept;
    external bases like ``nn.Module`` terminate a chain (which is what we want --
    we measure OUR inheritance towers, not depth from a framework base).
    """
    cls_bases: dict[str, list[str]] = {}
    for path in iter_src_files():
        tree = _parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
                cls_bases[node.name] = bases
    return cls_bases


def scan_deep_inheritance(max_depth: int = 2) -> set[str]:
    """Flag classes whose inheritance chain through OUR classes exceeds
    ``max_depth`` (depth = number of our-class ancestors)."""
    cls_bases = _build_class_base_map()
    offenders: set[str] = set()

    def depth(name: str, seen: frozenset[str]) -> int:
        if name in seen or name not in cls_bases:
            return 0
        bases = cls_bases.get(name, [])
        local = [b for b in bases if b in cls_bases]
        if not local:
            return 0
        return 1 + max(depth(b, seen | {name}) for b in local)

    for name in cls_bases:
        d = depth(name, frozenset())
        if d > max_depth:
            offenders.add(f"{name} (depth {d})")
    return offenders


# --------------------------------------------------------------------------- #
# Detector 5: signature dispatch via ``except TypeError`` (SAQ-001)
# --------------------------------------------------------------------------- #
def _callee_name(node: ast.AST) -> str | None:
    """The name a ``Call`` targets: ``gen(...)`` -> ``gen``, ``m.f(...)`` -> ``f``."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _catches_type_error(handler: ast.ExceptHandler) -> bool:
    node = handler.type
    if node is None:  # bare ``except:`` -- not this detector's business
        return False
    names = node.elts if isinstance(node, ast.Tuple) else [node]
    return any(isinstance(n, ast.Name) and n.id == "TypeError" for n in names)


def _call_arity(node: ast.Call) -> int:
    return len(node.args) + len(node.keywords)


def _try_owners(tree: ast.Module) -> dict[int, str]:
    """Map each ``ast.Try`` to the qualname of its INNERMOST enclosing function.

    ``_iter_qualified_functions`` yields a parent before its nested functions, so
    assigning in order lets the deeper owner overwrite the shallower one. Without
    this, a ``try`` inside a nested ``_fwd`` closure would be attributed to both
    and counted twice.
    """
    owners: dict[int, str] = {}
    for qualname, fn in _iter_qualified_functions(tree):
        for node in ast.walk(fn):
            if isinstance(node, ast.Try):
                owners[id(node)] = qualname
    return owners


def find_exception_dispatch(tree: ast.Module) -> list[tuple[str, str, int]]:
    """Return ``(qualname, callee, count)`` for handlers that RETRY a call more cheaply.

    The shape being caught is ``try: f(x, t) / except TypeError: f(x)`` -- using
    the exception as a signature probe. It is unsound because ``TypeError`` does
    not distinguish "this callable has no such parameter" (raised at the call
    boundary) from "the body raised a TypeError three frames down"; the second
    is a real bug and the retry silently answers it by running a degraded call.

    Deliberately narrow, so the flag means one thing:

    * the handler must call the SAME callee as the ``try`` body, and
    * with strictly FEWER arguments -- the degraded retry.

    A handler that pops one kwarg and ``continue``s a bounded loop (the
    progressive-strip retry in ``physics_driven_strategy``) makes no call of its
    own and is therefore not flagged: its retry is the loop, and it re-raises
    anything whose message it cannot account for.

    The entry is keyed on ``qualname::callee`` and measured by a COUNT, never a
    line number: a line moves on any edit above it, which is the #629 failure
    that made these gates unable to stay green. The qualname is required for
    uniqueness -- ``ddim_sampler.py`` retries ``model`` three times in one
    sampler, and ``path::model`` alone cannot say which site is baselined.
    """
    owners = _try_owners(tree)
    counts: dict[tuple[str, str], int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        tried: dict[str, int] = {}
        for sub in ast.walk(ast.Module(body=node.body, type_ignores=[])):
            name = _callee_name(sub)
            if name is not None:
                tried[name] = max(tried.get(name, 0), _call_arity(sub))
        for handler in node.handlers:
            if not _catches_type_error(handler):
                continue
            for sub in ast.walk(ast.Module(body=handler.body, type_ignores=[])):
                name = _callee_name(sub)
                if name in tried and _call_arity(sub) < tried[name]:
                    key = (owners.get(id(node), "<module>"), name)
                    counts[key] = counts.get(key, 0) + 1
    return [(qualname, callee, n) for (qualname, callee), n in counts.items()]


def scan_exception_dispatch() -> set[str]:
    offenders: set[str] = set()
    for path in iter_src_files():
        tree = _parse(path)
        if tree is None:
            continue
        for qualname, callee, n in find_exception_dispatch(tree):
            offenders.add(f"{rel(path)}::{qualname}::{callee} ({n} retries)")
    return offenders
