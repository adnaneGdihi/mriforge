#!/usr/bin/env python3
"""Report every config key the schema SILENTLY DISCARDS, across a corpus.

57 of ~163 mounted schema classes are ``extra="ignore"``. Inside one of those, a
key the schema does not declare is dropped with no error and no warning: the YAML
still shows it, the run never sees it, and the arm trains on the default. That is
the issue #550 mechanism, and it is invisible to every other gate the repo has --
the config loads, the audit passes, and the unconsumed-key ratchet cannot see it
because the key is not a field to begin with.

Two open issues are the same finding at cohort scale: **#675** (``logging:``
discards 26 names across 1,154 declarations -- 419 arms set ``project_name`` and
417 set ``enable_wandb``, neither of which exists) and **#681**
(``undersampling:``, 125 declarations over 7 names, including
``num_accumulation_steps`` -- a *third* spelling of the accumulation knob).

**This adds no detection logic.** ``diff_declared_vs_resolved``
(``core/execution_ledger.py``) already classifies exactly this as
``extra_ignore_dropped``; it just only runs when a ledger is armed, i.e. during a
real run. This script arms one per arm and aggregates.

It deliberately does NOT live in ``config_health_checker``. That checker validates
the RESOLVED config on purpose -- the no-``config_path`` design exists so that
what is checked is what runs -- and a discarded key is by definition absent from
the resolved config. The ledger, which sees the declaration and the model
together at ``from_yaml`` time, is the right seam.

Usage::

    python scripts/ci/report_discarded_config_keys.py experiments/inprogress
    python scripts/ci/report_discarded_config_keys.py experiments/ --by-arm
    python scripts/ci/report_discarded_config_keys.py experiments/ --max 0   # gate
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

warnings.filterwarnings("ignore")

from spectramr.config.settings import TrainingSettings  # noqa: E402
from spectramr.core.execution_ledger import ExecutionLedger  # noqa: E402

_DROPPED = "extra_ignore_dropped"


def discarded_keys(path: Path) -> list[tuple[str, str]] | None:
    """``(block, key)`` for every key this arm declares and the schema drops.

    ``None`` -- distinct from ``[]`` -- when the config does not load at all.

    A load failure genuinely is not our business: ``check_experiment_configs_load``
    owns it, walks the same trees, and reports every skip by category rather than
    swallowing it. But *this* script must still not present a file it never read
    as one it read and found clean. Returning ``[]`` for both made the roll-up's
    "scanned N" ambiguous in the one direction that matters -- "0 discarded" over
    a root where nothing loaded is indistinguishable from a clean root, and the
    quieter reading is the reassuring one.
    """
    ledger = ExecutionLedger.begin_run(source=str(path))
    try:
        TrainingSettings.from_yaml(str(path))
    except Exception:
        return None
    out: list[tuple[str, str]] = []
    for rec in ledger.to_dict(run_id="scan").get("substitutions", ()):
        if rec.get("class_id") != _DROPPED:
            continue
        # `path` on the record is the dotted location of the discarded key.
        dotted = str(rec.get("path") or rec.get("key") or "")
        if not dotted:
            continue
        block, _, leaf = dotted.rpartition(".")
        out.append((block or "<root>", leaf))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="*", default=["experiments/inprogress"])
    ap.add_argument("--by-arm", action="store_true", help="list every arm")
    ap.add_argument(
        "--max",
        type=int,
        default=None,
        metavar="N",
        help="exit 1 if more than N declarations are discarded (0 = gate)",
    )
    args = ap.parse_args(argv)

    per_key: Counter[str] = Counter()
    per_arm: dict[str, list[str]] = defaultdict(list)
    scanned = 0
    unread: list[str] = []
    for root in args.roots:
        for path in sorted(Path(root).rglob("*.yaml")):
            hits = discarded_keys(path)
            scanned += 1
            if hits is None:
                unread.append(str(path))
                continue
            for block, leaf in hits:
                per_key[f"{block}.{leaf}"] += 1
                per_arm[str(path)].append(f"{block}.{leaf}")

    total = sum(per_key.values())
    print(
        f"scanned {scanned} config(s) under {', '.join(args.roots)}: "
        f"{scanned - len(unread)} read, {len(unread)} did not load"
    )
    print(
        f"{total} declaration(s) SILENTLY DISCARDED across {len(per_key)} key(s), "
        f"in {len(per_arm)} arm(s)\n"
    )
    if per_key:
        width = max(len(k) for k in per_key)
        for key, n in per_key.most_common():
            print(f"  {n:5d}  {key:<{width}}")
    if args.by_arm:
        print()
        for arm in sorted(per_arm):
            print(f"  {arm}\n      {', '.join(sorted(set(per_arm[arm])))}")
    if not per_key:
        print("  (none)")

    if unread:
        print(
            f"\n{len(unread)} config(s) DID NOT LOAD and were therefore never "
            "inspected for discarded keys. They are not clean -- they are unread. "
            "check_experiment_configs_load owns the load failure itself:"
        )
        for arm in unread:
            print(f"  {arm}")

    print(
        "\nEach of these is declared in YAML and dropped at parse time: the arm "
        "trains on the schema default.\nFix per key: declare it, or delete it "
        "from the arms. See issues #675 and #681."
    )
    if args.max is not None and total > args.max:
        print(f"\nFAIL: {total} discarded declaration(s) exceeds --max {args.max}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
