"""Prove a component is REACHABLE, not merely registered.

The distinction this script exists for. Three claims get collapsed into "it's
registered", and only the first is usually established:

  registered  the name is in the registry once *something* imports its module
  reachable   the name is in the registry after a COLD import of the documented
              entry point alone -- which is all a YAML-driven run performs
  fires       the mechanism actually does work on a batch (not covered here)

A decorator that only fires because an unrelated test happened to import its module
is registered and not reachable. ``models/init_registry.py:73-190`` carries six dated
comments recording exactly that, e.g. "``gans`` registers 9 models ... Before this
entry, only the last 3 fired (transitively via ``generators/__init__.py``); the other
6 decorators were dead."

Method. Two subprocesses per registry, so nothing this process already imported can
contaminate the answer:

  COLD  import the documented entry point, read the registry
  WARM  pkgutil-walk every module under the component home, then read the registry

``WARM - COLD`` is the defect set: those names exist, look correct in source, pass a
same-process unit test -- and cannot be resolved from a config.

The two depths genuinely differ, so a clean result is not clean by construction:
``populate_model_registry()`` walks a HAND-CURATED list of 44 packages, while the warm
probe walks all 64 subdirectories under ``mriforge/models/`` -- a package missing from the
curated list is exactly what shows up in the gap. Likewise ``data/transforms/__init__.py``
curates 9 imports by hand out of everything in that directory.

Usage::

    python scripts/maintenance/prove_reachable.py --audit            # all registries
    python scripts/maintenance/prove_reachable.py --name unet_generator
    python scripts/maintenance/prove_reachable.py --audit --json

Exit codes: 0 nothing unreachable, 1 at least one name is registered-but-unreachable
(or, with ``--name``, the name is not reachable). ``--warn-only`` always exits 0.

Note this measures THIS interpreter's environment. Run it in the environment the run
will use; a local `.venv` result is not a claim about the cluster.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Registry:
    """One registry, and the two import depths used to probe it."""

    kind: str
    #: Documented entry point — what a config-driven run actually triggers.
    cold: str
    #: Package whose modules get walked to find everything that *could* register.
    home: str
    #: Expression returning an iterable of registered names, given the imports above.
    names: str


REGISTRIES: tuple[Registry, ...] = (
    Registry(
        kind="transform",
        cold="import mriforge.data.transforms",
        home="mriforge.data.transforms",
        names="__import__('mriforge.data.transforms.registry', fromlist=['x']).list_transforms()",
    ),
    Registry(
        kind="metric",
        cold="import mriforge.core.metrics",
        home="mriforge.core.metrics",
        names="list(__import__('mriforge.core.metrics', fromlist=['x']).MetricsRegistry._metrics)",
    ),
    Registry(
        kind="loss",
        cold="import mriforge.models.losses",
        home="mriforge.models.losses",
        names=(
            "list(__import__('mriforge.models.losses.registry', fromlist=['x'])"
            ".LossRegistry._custom_losses)"
        ),
    ),
    Registry(
        kind="model",
        # Models are the one registry with an explicit population step; a bare
        # `import mriforge.models` is deliberately NOT the documented entry point.
        cold=(
            "from mriforge.models.init_registry import populate_model_registry;"
            " populate_model_registry()"
        ),
        home="mriforge.models",
        names="list(__import__('mriforge.models.registry', fromlist=['x']).MODEL_REGISTRY)",
    ),
)

_PROBE = """
import json, sys, warnings, importlib, pkgutil
warnings.simplefilter("ignore")


{setup}
try:
    names = sorted({names})
except Exception as exc:                       # a registry that cannot even be read
    print(json.dumps({{"error": f"{{type(exc).__name__}}: {{exc}}"}}))
    sys.exit(0)
print(json.dumps({{"names": names}}))
"""


def _run(setup: str, names: str, *, timeout: int) -> tuple[list[str], str | None]:
    """Run one probe in a cold interpreter. Returns (names, error)."""
    code = _PROBE.format(setup=setup, names=names)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=REPO_ROOT,
        )
    except subprocess.TimeoutExpired:
        return [], f"probe timed out after {timeout}s"
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()
        return [], tail[-1] if tail else f"probe exited {proc.returncode}"
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return [], "probe produced no JSON"
    if "error" in payload:
        return [], payload["error"]
    return payload["names"], None


def probe(reg: Registry, *, timeout: int) -> dict[str, object]:
    cold, cold_err = _run(reg.cold, reg.names, timeout=timeout)
    warm_setup = (
        f"{reg.cold}\n"
        "import pkgutil as _pk, importlib as _il, warnings as _w\n"
        "_w.simplefilter('ignore')\n"
        f"_pkg = _il.import_module({reg.home!r})\n"
        "for _m in _pk.walk_packages(_pkg.__path__, _pkg.__name__ + '.'):\n"
        "    try:\n"
        "        _il.import_module(_m.name)\n"
        "    except Exception:\n"
        "        pass\n"
    )
    warm, warm_err = _run(warm_setup, reg.names, timeout=timeout * 4)
    return {
        "kind": reg.kind,
        "cold": cold,
        "warm": warm,
        "unreachable": sorted(set(warm) - set(cold)),
        "error": cold_err or warm_err,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0] or None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit", action="store_true", help="Probe every registry.")
    mode.add_argument("--name", help="Ask whether one component name is reachable.")
    parser.add_argument(
        "--kind",
        choices=[r.kind for r in REGISTRIES],
        help="Restrict to one registry (default: all).",
    )
    parser.add_argument("--timeout", type=int, default=120, help="Cold-probe timeout, seconds.")
    parser.add_argument("--json", action="store_true", help="Emit results as JSON.")
    parser.add_argument("--warn-only", action="store_true", help="Always exit 0.")
    args = parser.parse_args()

    selected = [r for r in REGISTRIES if not args.kind or r.kind == args.kind]
    results = [probe(r, timeout=args.timeout) for r in selected]

    if args.json:
        print(json.dumps(results, indent=2))
        any_problem = any(r["unreachable"] or r["error"] for r in results)
        return 0 if args.warn_only or not any_problem else 1

    if args.name:
        found = False
        for r in results:
            if args.name in r["cold"]:  # type: ignore[operator]
                print(f"REACHABLE   {args.name!r} resolves as a {r['kind']} from a cold import.")
                found = True
            elif args.name in r["warm"]:  # type: ignore[operator]
                print(
                    f"UNREACHABLE {args.name!r} is a registered {r['kind']}, but only after "
                    f"walking {r['kind']} modules. A config cannot resolve it — the import "
                    f"curation step is missing."
                )
        if not found and not any(args.name in r["warm"] for r in results):  # type: ignore[operator]
            print(f"UNKNOWN     {args.name!r} is in no probed registry, at either import depth.")
        return 0 if args.warn_only or found else 1

    # A probe that could not run and a probe that found nothing must never be
    # summarised the same way -- collapsing them is the soft-skip seam this script
    # exists to expose, and reporting "N names unreachable" for N failed probes
    # would be exactly the confidently-wrong answer it is meant to catch.
    unreachable_total = 0
    failed: list[str] = []
    for r in results:
        if r["error"]:
            print(f"{r['kind']:10} PROBE FAILED — {r['error']}")
            failed.append(str(r["kind"]))
            continue
        bad: list[str] = list(r["unreachable"])  # type: ignore[arg-type]
        cold, warm = list(r["cold"]), list(r["warm"])  # type: ignore[arg-type]
        print(f"{r['kind']:10} cold={len(cold):<4} walked={len(warm):<4} unreachable={len(bad)}")
        for name in bad:
            print(f"             - {name}")
        unreachable_total += len(bad)

    if failed:
        print(
            f"\n{len(failed)} probe(s) could not run ({', '.join(failed)}), so those registries "
            "were NOT checked.\nThis is not a clean result. Most often the interpreter lacks the "
            "package: run under the\nproject venv (`.venv/bin/python`), not a bare `python3`. From a "
            "git worktree there is no\nlocal `.venv`, so pass the primary checkout's interpreter "
            "explicitly."
        )
    if unreachable_total:
        print(
            f"\n{unreachable_total} name(s) are registered but not reachable from a cold import.\n"
            "Each is invisible to a config-driven run. Add the missing import-curation step —\n"
            "see the `reachability-contract` skill for which one applies per registry."
        )
    if not failed and not unreachable_total:
        print("\nEvery registered name is reachable from a cold import.")
    return 0 if args.warn_only or not (failed or unreachable_total) else 1


if __name__ == "__main__":
    sys.exit(main())
