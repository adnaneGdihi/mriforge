"""Which arm currently drops which declared key — recorded debt, per arm.

The third of the three owners the smoke sweeps share, after
:mod:`tests.utils.corpus` (*which arms are swept*) and
:mod:`tests.utils.config_load_baseline` (*which of them cannot construct at
all*). This one owns the next question down: **which keys does a loadable arm
declare that the loader silently discards, and which of those are already
known.**

Why it exists
-------------
``tests/smoke/test_deep_config_integrity.py`` enumerated
``experiments/in_progress`` — a directory that has never existed — so it swept
zero of the 647 committed ``inprogress/`` arms (#1053). Correcting the path
grew its subject from 128 parameters to 2069 and surfaced **337 arms declaring
86 distinct keys the loader drops** (1494 arm/key pairs): pitfall-#15 debt that
was always there and never visible.

The measurement is corroborated: restricted to ``data.*`` on ``inprogress/``
arms it reports 244 arms / 620 declarations / 29 paths, matching issue #1012's
independently-derived census exactly once its two ``KNOWN_DROPPED_KEYS``
members are added back (``data.slice_aware`` on 39 arms).

That debt is real and must not be waived. It is also far too large to fix in
the PR that merely corrects a path, and several of the keys are already tracked
with their own fixes pending (``checkpoint.save_dir`` is #1061,
``data.*`` drops are #1012, and ``data.enable_geometric_standardization`` is the
worked example in ``scripts/ci/check_getattr_names_a_real_field.py``). So it is
recorded here and ratcheted: **an arm/key pair that is not recorded still fails
the sweep.**

Why this is NOT ``KNOWN_DROPPED_KEYS``
--------------------------------------
The sweep already has a key-level list, and the two say different things:

``KNOWN_DROPPED_KEYS``
    a claim about a KEY: *"this key is genuinely dead corpus-wide"* — the
    sweep's own failure message demands evidence for entry, and a stale-entry
    ratchet polices it.

this file
    a claim about an ARM: *"this arm drops this key today"* — a measurement,
    asserting nothing about whether the key should be wired, deleted or
    respelled.

Folding 86 keys into ``KNOWN_DROPPED_KEYS`` to make the sweep green would have
asserted 86 things nobody verified, and would have permanently blessed keys
whose issues call for **wiring** them. Recording arms asserts only what was
measured.

Keyed per ``(arm, key)``, not per key
-------------------------------------
Deliberate, and the same argument as ``scripts/ci/model_kwargs_baseline.txt``:
a key-level record lets ``data.volume_format`` reach a 216th arm for free,
which is exactly the spread a ratchet exists to stop. A new arm declaring an
already-recorded key is a NEW pair, and fails.

Regeneration
------------
Never hand-edit. The file is generated from the sweep's own resolver, so it
cannot drift from what the tests measure::

    python -m tests.utils.dropped_key_baseline --update

Removing a pair is the goal; adding one needs a reason in the PR that adds it —
the same contract ``scripts/ci/config_load_baseline.txt`` states in its header.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from tests.utils.corpus import repo_root

__all__ = ["baseline_path", "recorded_dropped_keys", "recorded_pairs"]


def baseline_path() -> Path:
    """Location of the record.

    Co-located with its siblings under ``scripts/ci/`` rather than beside this
    module: ``config_load_baseline.txt``, ``getattr_field_baseline.txt`` and
    ``model_kwargs_baseline.txt`` all live there, and a baseline that is
    findable only by reading the module that consumes it is a baseline nobody
    ratchets down.
    """
    return repo_root() / "scripts" / "ci" / "dropped_key_baseline.txt"


@lru_cache(maxsize=1)
def recorded_pairs() -> dict[str, frozenset[str]]:
    """``repo-relative arm -> frozenset of dropped keys`` recorded for it."""
    path = baseline_path()
    if not path.exists():
        return {}
    pairs: dict[str, set[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        arm, _, key = entry.rpartition(" ")
        if not arm or not key:
            continue
        pairs.setdefault(arm, set()).add(key)
    return {arm: frozenset(keys) for arm, keys in pairs.items()}


def recorded_dropped_keys(config_path: Path | str) -> frozenset[str]:
    """Keys recorded as already-dropped for ``config_path``.

    Accepts absolute or repo-relative paths, for the reason
    :func:`tests.utils.config_load_baseline.is_baselined` documents: the sweeps
    disagree about which they hold, since ``tests.utils.corpus`` returns
    absolute paths while older ``glob`` call sites carry relative strings.
    """
    candidate = Path(config_path)
    if candidate.is_absolute():
        try:
            candidate = candidate.resolve().relative_to(repo_root())
        except ValueError:
            return frozenset()
    return recorded_pairs().get(candidate.as_posix(), frozenset())


def _regenerate() -> int:
    """Rewrite the baseline from the sweep's own resolver.

    Imported lazily and only here: the smoke module imports ``TrainingSettings``
    at module scope, and making that a cost of *reading* the baseline would
    couple every sweep's collection to the schema importing cleanly — in a tree
    where a broken schema import is one of the things these sweeps exist to
    detect.
    """
    from tests.smoke.test_deep_config_integrity import (
        ALL_CONFIGS,
        KNOWN_DROPPED_KEYS,
        dropped_keys,
    )
    from tests.utils.config_load_baseline import is_baselined

    lines: list[str] = []
    for config_path in ALL_CONFIGS:
        # A baselined arm does not construct, so it yields no measurement at
        # all -- recording it would be inventing one.
        if is_baselined(config_path):
            continue
        arm = Path(config_path).resolve().relative_to(repo_root()).as_posix()
        for key in sorted(dropped_keys(config_path) - KNOWN_DROPPED_KEYS):
            lines.append(f"{arm} {key}")

    header = (
        "# Arm/key pairs where the YAML declares a key the loader silently drops.\n"
        "# Recorded DEBT, not permission: removing a pair is the goal, adding one\n"
        "# needs a reason in the PR that adds it. See tests/utils/\n"
        "# dropped_key_baseline.py. Regenerate with:\n"
        "#     python -m tests.utils.dropped_key_baseline --update\n"
    )
    baseline_path().write_text(header + "\n".join(sorted(lines)) + "\n", encoding="utf-8")
    print(f"Baseline rewritten: {len(lines)} (arm, key) pair(s).")
    return 0


if __name__ == "__main__":  # pragma: no cover - operator entry point
    import sys

    if "--update" not in sys.argv:
        print(__doc__)
        sys.exit(2)
    sys.exit(_regenerate())
