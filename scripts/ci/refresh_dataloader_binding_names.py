"""Regenerate the DataLoader binding-name vocabulary the SSOT gate matches against.

Split out from ``check_dataloader_construction_ssot.py`` on purpose. The gate runs
in the ``guards`` job of ``.github/workflows/pr-required.yml``, which does
``checkout`` + ``setup-python`` and **no pip install** -- so the gate itself may
import nothing beyond ``ast`` / ``pathlib``. Discovery needs the opposite: torch,
torchio and monai must be importable to ask them what actually subclasses
``DataLoader``. Keeping the two apart is what lets the vocabulary be *discovered*
rather than enumerated while the gate stays dependency-free.

Why binding names and not class ``__name__``s
---------------------------------------------
The obvious vocabulary -- the ``__name__`` of every subclass -- is one short.
``monai.transforms.inverse_batch_transform`` re-exports the torch loader as
``TorchDataLoader``, so ``from monai.transforms.inverse_batch_transform import
TorchDataLoader`` binds a name that appears in no class ``__name__``. That is the
gate docstring's own "enumerating two of the three reports clean while the third
walks past", one level up. So this records every **attribute name a subclass is
reachable under**, across every imported module.

Usage:
    python scripts/ci/refresh_dataloader_binding_names.py           # rewrite the file
    python scripts/ci/refresh_dataloader_binding_names.py --check   # verify freshness
"""

from __future__ import annotations

import argparse
import ast
import importlib
import sys
from pathlib import Path

# The gate owns the vocabulary contract (filename + reader); this script owns
# only its regeneration. Importing that direction keeps ONE parser rather than
# two that must be kept in sync (non-negotiable #17).
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_dataloader_construction_ssot import (
    VOCABULARY_FILENAME,
    read_vocabulary,
)

#: Namespaces the generating process itself creates; never real import targets.
_SYNTHETIC_MODULES = {"__main__", "__mp_main__"}


def repo_third_party_modules(root: Path) -> set[str]:
    """Every non-stdlib, non-``mriforge`` module ``src/mriforge`` imports anywhere.

    Function-local imports included -- ``ast.walk`` does not care about depth, and
    a loader imported inside a function is exactly the shape a top-of-file scan
    misses (the same blind spot as ``check_layering.sh``'s ``^``-anchored greps).
    """
    modules: set[str] = set()
    for path in (root / "src" / "mriforge").rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module)
    stdlib = sys.stdlib_module_names
    return {m for m in modules if not m.startswith("mriforge") and m.split(".")[0] not in stdlib}


def discover(root: Path) -> tuple[set[str], list[str]]:
    """Return ``(binding names, unimportable modules)``.

    The unimportable list is returned rather than swallowed: non-negotiable #18 --
    absent is a state to report, never a state to infer. A vocabulary built while
    torchio failed to import is not a clean vocabulary, it is a blind one.
    """
    import torch.utils.data

    unimportable: list[str] = []
    for name in sorted(repo_third_party_modules(root)):
        try:
            importlib.import_module(name)
        except BaseException:
            unimportable.append(name)

    # Transitive closure: a subclass of a subclass is still a DataLoader.
    closure: set[type] = {torch.utils.data.DataLoader}
    frontier = [torch.utils.data.DataLoader]
    while frontier:
        for sub in frontier.pop().__subclasses__():
            if sub not in closure:
                closure.add(sub)
                frontier.append(sub)

    names: set[str] = set()
    for mod_name, module in list(sys.modules.items()):
        if module is None or mod_name in _SYNTHETIC_MODULES:
            continue
        for attr, value in vars(module).items():
            if isinstance(value, type) and value in closure:
                names.add(attr)
    return names, unimportable


def render(names: set[str], unimportable: list[str]) -> str:
    header = [
        "# DataLoader binding names -- GENERATED, do not hand-edit.",
        "# Regenerate: python scripts/ci/refresh_dataloader_binding_names.py",
        "#",
        "# Every attribute name under which a torch.utils.data.DataLoader subclass",
        "# is reachable in this environment. check_dataloader_construction_ssot.py",
        "# matches call sites against this list, so a new loader class in a new",
        "# dependency is picked up by regenerating rather than by editing a gate.",
        "#",
        f"# {len(unimportable)} module(s) were not importable when this was generated,",
        "# so any loader they alone expose is NOT represented here:",
    ]
    header += [f"#   {m}" for m in unimportable] or ["#   (none)"]
    return "\n".join(header) + "\n" + "\n".join(sorted(names)) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if the committed vocabulary differs from a live rediscovery.",
    )
    args = parser.parse_args()

    target = args.root / "scripts" / "ci" / VOCABULARY_FILENAME
    names, unimportable = discover(args.root)

    if args.check:
        committed = read_vocabulary(target)
        if committed != names:
            print(f"{target.name} is stale.")
            print(f"  missing (discovered, not committed): {sorted(names - committed)}")
            print(f"  extra   (committed, not discovered): {sorted(committed - names)}")
            return 1
        print(f"{target.name}: fresh ({len(names)} binding name(s)).")
        return 0

    target.write_text(render(names, unimportable), encoding="utf-8")
    print(f"Wrote {target} with {len(names)} binding name(s): {sorted(names)}")
    if unimportable:
        print(f"{len(unimportable)} module(s) not importable here: {unimportable}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
