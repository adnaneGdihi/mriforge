"""Every constrained field in the reference template lists its real options.

``v1.0_reference.yaml`` is the file people copy. A value it advertises that the
schema rejects is worse than no documentation at all: it produces a config that
does not load, and the reader trusts the reference over the error.

That is not hypothetical. ``experiments/templates/categorical_values_reference.yaml``
is the hand-written catalogue this module exists to prevent a repeat of. It was
generated once, by hand, and never re-derived: on ``training_mode`` alone it
advertises three members the enum no longer has (``swarm_learning``,
``federated``, ``test_time_adaptation``) and omits six it does. Its header even
cites ``src/config/validation_constants.py`` — a path that stopped existing at
the 2026-05 ``src→spectramr`` refactor. Nothing failed when it rotted, because
nothing checked it.

So the option lists here are not maintained by hand either. They are compared,
per path, against the annotation the loader actually validates against, and the
four checks below are deliberately different shapes of the same question:

``test_option_lists_match_the_schema``
    a list that drifts is caught (the ``categorical_values_reference`` failure).
``test_every_constrained_field_is_documented``
    a field added WITHOUT documentation is caught. This is a full ratchet at
    complete, not a spot check — a new ``Literal`` field fails here on arrival.
``test_no_phantom_keys``
    a key the template declares that the schema has no field for. Note that
    ``extra="forbid"`` does NOT cover this: it protects live keys only, and the
    template's documentation is mostly COMMENTED. It also does not cover blocks
    that are ``extra="ignore"``, where a wrong key loads and does nothing.
``test_registry_pointers_resolve``
    for the fields whose value set is a registry rather than an annotation, the
    named symbol must still import. This is the exact failure the stale
    catalogue's header shows.

The parser is the load-bearing part and is written to one convention: a
commented key is ``# `` followed by the exact line it would be live. The comment
marker contributes NO depth — counting its two columns is what makes ``  # k:``
collide with ``    k:`` and silently reparent half a block.
"""

from __future__ import annotations

import ast
import enum
import importlib
import re
import types
import typing
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel

from spectramr.config.settings import TrainingSettings
from spectramr.config.schemas.introspection import nested_models

TEMPLATE = (
    Path(__file__).resolve().parents[3]
    / "src/spectramr/config/schemas/templates/v1.0_reference.yaml"
)

# Indentation may sit on either side of the comment marker; both spellings are
# in the file. `- ` starts a sequence item, whose keys are two columns right.
_KEY = re.compile(r"^( *)((?:# ?)?)( *)(- )?([a-z_][a-z0-9_]*) *:(.*)$")
_OPT = re.compile(r"#\s*options:\s*(.*)$")
_CONT = re.compile(r"^\s*#\s{2,}(\S.*)$")
# `a/b/c.Symbol` -- a pointer written as a path rather than after "see".
_PATH_POINTER = re.compile(r"(?:[a-z_]+/)+[a-z_]+\.[A-Za-z_]\w*")

# Value sets that live in a registry, not in an annotation. Each maps the text
# the template points at to the symbol that must still exist.
_POINTERS = {
    "MODEL_REGISTRY": ("spectramr.models.registry", "MODEL_REGISTRY"),
    "TrainingStrategyFactory.STRATEGY_CLASS_PATHS": (
        "spectramr.infrastructure.training.strategy_factory",
        "TrainingStrategyFactory",
    ),
    "KANType": ("spectramr.config.schemas.enums", "KANType"),
    # The template used to point at MaskType here. It now deliberately points
    # at the accelerator registry and says "NOT MaskType" -- the two vocabularies
    # overlap but are different sets -- so the pointer that must still resolve is
    # the registry. Replaced rather than deleted: dropping the entry would have
    # left the template's one live pointer unchecked.
    "infrastructure/physics/sampling._ACCELERATOR_REGISTRY": (
        "spectramr.infrastructure.physics.sampling",
        "_ACCELERATOR_REGISTRY",
    ),
}


# --------------------------------------------------------------------------
# schema introspection
# --------------------------------------------------------------------------
def _leaves(ann):
    """Concrete members of a possibly-Optional/Union annotation."""
    if typing.get_origin(ann) in (typing.Union, types.UnionType):
        for arg in typing.get_args(ann):
            if arg is not type(None):
                yield from _leaves(arg)
    else:
        yield ann


def _choices(ann) -> list | None:
    """The allowed values of an Enum/Literal annotation, else None.

    A plain ``bool`` is deliberately NOT constrained: '# options: true, false'
    is noise, and counting bools inflates the surface roughly six-fold.
    """
    vals, kinds = [], set()
    for leaf in _leaves(ann):
        if typing.get_origin(leaf) is typing.Literal:
            args = typing.get_args(leaf)
            kinds.add("bool" if set(args) <= {True, False} else "real")
            if set(args) - {True, False}:
                vals.extend(args)
        elif isinstance(leaf, type) and issubclass(leaf, enum.Enum):
            kinds.add("real")
            vals.extend(m.value for m in leaf)
    return vals if "real" in kinds else None


def _submodels(ann):
    """Every model this annotation can resolve to.

    Delegated to the SSOT. The hand-rolled version here unwrapped two levels,
    which covered `X`, `X | None` and `list[X]` -- every field in the tree until
    `training.diffusion` became a discriminated union. Its annotation is
    `Optional[Annotated[A | B | ..., FieldInfo(discriminator=...)]]`, three deep,
    so this yielded nothing and `test_no_phantom_keys` reported all ten
    documented `training.diffusion.*` keys as having no schema field behind
    them -- a confident, specific, entirely wrong list.
    """
    yield from nested_models(ann)


def _walk(model, prefix="", depth=0, seen=frozenset()):
    if depth > 12 or model in seen:
        return
    seen = seen | {model}
    for name, field in model.model_fields.items():
        path = f"{prefix}{name}"
        choices = _choices(field.annotation)
        yield (
            path,
            {
                "choices": choices,
                "optional": typing.get_origin(field.annotation) in (typing.Union, types.UnionType)
                and type(None) in typing.get_args(field.annotation),
                "is_dict": str(field.annotation).startswith("dict["),
                # the alias is the key YAML is allowed to use
                "alias": (
                    f"{prefix}{field.validation_alias}"
                    if isinstance(field.validation_alias, str)
                    else None
                ),
            },
        )
        for sub in _submodels(field.annotation):
            yield from _walk(sub, f"{path}.", depth + 1, seen)


@pytest.fixture(scope="module")
def schema() -> dict[str, dict]:
    """Dotted path -> field facts, including alias spellings.

    Paths are MERGED, not overwritten. A discriminated union yields the same
    dotted path once per variant -- `training.diffusion.type` arrives eight
    times, each carrying that variant's single-member `Literal`. Plain
    `dict(...)` keeps whichever came last, so the field's documented vocabulary
    became `['chi_square']` purely because `ChiSquareParams` is last in the
    union. The honest answer is every tag the discriminator accepts.
    """
    out: dict[str, dict] = {}
    for path, info in _walk(TrainingSettings):
        existing = out.get(path)
        if existing is None:
            out[path] = info
            continue
        if existing.get("choices") and info.get("choices"):
            merged = list(existing["choices"])
            merged += [c for c in info["choices"] if c not in merged]
            existing["choices"] = merged
        elif info.get("choices"):
            existing["choices"] = info["choices"]
    for info in list(out.values()):
        if info["alias"]:
            out.setdefault(info["alias"], info)
    return out


# --------------------------------------------------------------------------
# template parsing
# --------------------------------------------------------------------------
def _is_key(line: str) -> bool:
    """True for a real (live or commented) YAML key line.

    Prose inside a comment can look like one, in two different ways, and both
    occur in this file:

    * ``#   torch: adam, adamw, sgd`` sits under ``optimizer:`` and would
      register as a key named ``torch`` -- rejected because the value is not
      loadable as a single YAML node;
    * a wrapped sentence whose line break lands just before a colon
      (``# documentation: plain bool fields ...``) IS loadable, as a bare
      string -- rejected because a real value is a scalar, not a phrase.

    So the rule is: empty, or a value that both parses and is scalar-shaped
    (quoted, numeric, bool, null, a literal list/dict, or one bare token).
    """
    m = _KEY.match(line)
    if not m or m.group(5) == "options":
        return False
    value = m.group(6).split("#")[0].strip()
    if not value:
        return True
    try:
        yaml.safe_load(value)
    except yaml.YAMLError:
        return False
    if value[0] in "'\"[{":
        return True
    return len(value.split()) == 1


@pytest.fixture(scope="module")
def template() -> dict[str, dict]:
    """Dotted path -> {line, live, options} for every key the template declares."""
    lines = TEMPLATE.read_text().splitlines()
    opt_at: dict[int, str] = {}

    i = 0
    while i < len(lines):
        m = _OPT.search(lines[i])
        if not m:
            i += 1
            continue
        text, j = m.group(1), i + 1
        # A continuation is a bare comment line -- never the next commented KEY.
        # '#   mae:' reads as an indented comment and would otherwise be
        # swallowed into the previous field's option list.
        while (
            j < len(lines)
            and _CONT.match(lines[j])
            and not _OPT.search(lines[j])
            and not _is_key(lines[j])
        ):
            text += " " + _CONT.match(lines[j]).group(1)
            j += 1
        target = i if _is_key(lines[i]) else None
        if target is None:
            target = next((k for k in range(j, len(lines)) if _is_key(lines[k])), None)
        if target is not None:
            opt_at[target] = text
        i = j

    stack: list[tuple[int, str]] = []
    out: dict[str, dict] = {}
    for i, line in enumerate(lines):
        m = _KEY.match(line)
        if not m or line.lstrip().startswith("# =") or not _is_key(line):
            continue
        indent, hashed, indent2, dash, name, _ = m.groups()
        depth = len(indent) + len(indent2) + (2 if dash else 0)
        while stack and stack[-1][0] >= depth:
            stack.pop()
        out[".".join([s[1] for s in stack] + [name])] = {
            "line": i + 1,
            "live": not hashed,
            "options": opt_at.get(i),
        }
        stack.append((depth, name))
    return out


def _parse_values(text: str | None) -> list | None:
    """An option-comment body -> its values, or None when it is a pointer."""
    if text is None or "see " in text or "etc." in text:
        return None
    text = re.sub(r"\((?![^)]*\bor\b)[^)]*\)", "", text)  # drop parenthetical prose
    out = []
    for token in text.split(","):
        token = token.strip().rstrip(".")
        if not token or token.startswith("..."):
            continue
        if token in ("null", "None"):
            out.append(None)
            continue
        try:
            out.append(ast.literal_eval(token))
        except (ValueError, SyntaxError):
            return None
    return out


def _constrained(schema: dict) -> dict[str, dict]:
    return {p: i for p, i in schema.items() if i["choices"]}


# --------------------------------------------------------------------------
# the contracts
# --------------------------------------------------------------------------
def test_template_still_loads():
    """The premise of every other assertion here."""
    assert TrainingSettings.from_yaml(str(TEMPLATE)) is not None


def test_option_lists_match_the_schema(schema, template):
    """Each '# options:' list equals the set the loader will accept.

    This is the ``categorical_values_reference.yaml`` failure mode: a list that
    was right once, drifted, and had nothing checking it.
    """
    wrong = []
    for path, entry in template.items():
        info = schema.get(path)
        if not info or not info["choices"]:
            continue
        got = _parse_values(entry["options"])
        assert got is not None, (
            f"{path} (line {entry['line']}) is a closed enum with no readable "
            f"'# options:' list. Every constrained field must carry one."
        )
        want = ([None] if info["optional"] else []) + list(info["choices"])
        if sorted(map(str, got)) != sorted(map(str, want)):
            wrong.append(
                f"  {path} (line {entry['line']})\n"
                f"    template: {sorted(map(str, got))}\n"
                f"    schema  : {sorted(map(str, want))}\n"
                f"    invented: {sorted(set(map(str, got)) - set(map(str, want)))}\n"
                f"    missing : {sorted(set(map(str, want)) - set(map(str, got)))}"
            )
    assert not wrong, "option lists disagree with the schema:\n" + "\n".join(wrong)


def test_every_constrained_field_is_documented(schema, template):
    """A full ratchet, not a sample: ALL of them present and annotated.

    Scoped to Enum/Literal on purpose. Adding one of those is adding a closed
    vocabulary, and a vocabulary nobody can discover is the reason the flags in
    ``metrics.compute_*`` went 144-deep before anyone counted.
    """
    constrained = _constrained(schema)
    missing = sorted(set(constrained) - set(template))
    assert not missing, (
        f"{len(missing)} constrained field(s) absent from the reference "
        f"template — a closed vocabulary nobody can find:\n  "
        + "\n  ".join(f"{p}: {constrained[p]['choices']}" for p in missing)
    )
    unannotated = sorted(p for p in constrained if _parse_values(template[p]["options"]) is None)
    assert not unannotated, "documented but with no readable '# options:' list:\n  " + "\n  ".join(
        f"{p} (line {template[p]['line']})" for p in unannotated
    )


def test_no_phantom_keys(schema, template):
    """Every key the template declares maps to a real field.

    ``extra="forbid"`` does not cover this. It sees LIVE keys only, and the
    documentation added for the optional blocks is commented; and it is not even
    in force everywhere, since blocks like ``DigitalTwinConfig`` are
    ``extra="ignore"``, where a wrong key loads and quietly does nothing.
    """
    # below a dict[...] field the keys are DATA, not schema
    dict_prefixes = tuple(p + "." for p, i in schema.items() if i["is_dict"])
    # consumed by the loader and folded onto run.config_version
    loader_keys = {"config_version"}
    phantom = sorted(
        p
        for p in template
        if p not in schema and p not in loader_keys and not p.startswith(dict_prefixes)
    )
    assert not phantom, "keys with no schema field behind them:\n  " + "\n  ".join(
        f"{p} (line {template[p]['line']}, live={template[p]['live']})" for p in phantom
    )


def test_every_top_level_block_is_documented(schema, template):
    """A block absent from the reference is a feature nobody knows exists."""
    tops = {p for p in schema if "." not in p}
    missing = sorted(tops - set(template))
    assert not missing, f"top-level blocks the reference never mentions: {missing}"


@pytest.mark.parametrize("text,target", sorted(_POINTERS.items()))
def test_registry_pointers_resolve(text, target):
    """A pointer to a registry must name something that still exists.

    The stale catalogue cites ``src/config/validation_constants.py``, a path
    retired by the ``src→spectramr`` refactor. A pointer is documentation too.
    """
    module_name, symbol = target
    assert TEMPLATE.read_text().count(text) >= 1, (
        f"pointer {text!r} is no longer in the template; drop it from _POINTERS"
    )
    module = importlib.import_module(module_name)
    assert hasattr(module, symbol), f"{module_name}.{symbol} no longer exists"


def test_every_pointer_is_registered():
    """A NEW ``see X`` pointer must be added to ``_POINTERS``, not just written.

    Without this the pointer leg has the opposite polarity to the options leg:
    it catches an entry that leaves the template, but a pointer someone adds
    later would go unchecked forever -- which is precisely how the stale
    catalogue came to cite a module path that no longer exists.

    Two phrasings, not one. The original scan required the literal word
    ``see``, so ``# options: the keys of infrastructure/physics/sampling.
    _ACCELERATOR_REGISTRY`` -- the template's only live pointer -- was invisible
    to it, and the entry that WAS registered had gone stale unnoticed. A gate is
    only a gate for the violation shape it has been watched to fail on.
    """
    seen = set()
    for line in TEMPLATE.read_text().splitlines():
        m = _OPT.search(line)
        if not m:
            continue
        body = m.group(1)
        if "see " in body:
            seen.add(body.split("see ", 1)[1].split(" (")[0].strip().rstrip("."))
        # A slashed module path with a dotted symbol, however the sentence is
        # worded around it.
        seen.update(_PATH_POINTER.findall(body))
    unregistered = sorted(
        s for s in seen if not any(s.startswith(k) or k.startswith(s) for k in _POINTERS)
    )
    assert not unregistered, (
        "'# options: see X' pointers with no _POINTERS entry, so nothing checks "
        f"that X still exists: {unregistered}"
    )
