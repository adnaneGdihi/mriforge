"""Per-array-task dispatch for the SLURM experiment array (WS-4).

This is the Python SSOT for the fragile body that used to live inline in
``scripts/training/dispatch_experiments.sbatch``: resolve the config for THIS
array index from a manifest, optionally cap the iteration budget (smoke runs),
run the Tier-0/1 audit pre-flight, then ``spectramr train``.

Why it moved out of bash
------------------------
The ``TRAIN_ITERS`` cap was a ``grep | sed | … || true`` pipeline under
``set -euo pipefail``. It aborted whole arrays in production **twice**: a config
that omits ``eval_interval`` makes the ``grep`` exit 1, and ``set -e`` then kills
the task right after the header, before training (job 7145522 — all 56
mrixfields tasks). Reimplemented here, the cap is a plain YAML parse:
``yaml.safe_load`` strips inline comments natively (no ``sed``) and
``dict.get → None`` handles a missing key explicitly (no ``set -e`` foot-gun).

Torch-free by design
--------------------
This module imports only stdlib + PyYAML, and ``spectramr.cli`` package init is
torch-free, so ``python -m spectramr.cli.manifest_dispatch`` runs on a torch-less
interpreter. The dispatcher's ``--dry-run`` path (the unit tests) therefore
needs no GPU/torch; the real audit + train are shelled out to
``python -m spectramr.cli`` AFTER the venv is activated, so the heavy imports
happen in the right interpreter.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

#: Default per-arm log/audit root, relative to the repo root (mirrors the old
#: ``DISPATCH_DIR`` default in the .sbatch).
_DEFAULT_DISPATCH_DIR = "experiments/results/dispatch"


def resolve_manifest_config(manifest_path: str | Path, index: int) -> str:
    """Return the config path on line ``index`` (0-based) of ``manifest_path``.

    Blank lines are ignored (a trailing newline is not a phantom arm). Raises
    :class:`IndexError` with an "out of range" message when ``index`` is outside
    ``[0, N)`` — the dispatcher turns that into a non-zero exit so SLURM marks
    the task failed rather than silently training nothing.
    """
    text = Path(manifest_path).read_text()
    configs = [ln.strip() for ln in text.splitlines() if ln.strip()]
    n = len(configs)
    if index < 0 or index >= n:
        raise IndexError(
            f"SLURM_ARRAY_TASK_ID={index} out of range [0, {n}) for manifest {manifest_path}"
        )
    return configs[index]


def _read_capped_key(
    cfg: dict,
    section: str,
    key: str,
    *,
    block: str | None = None,
    legacy_key: str | None = None,
) -> int | None:
    """Read a declared int out of ``cfg[section]``, or ``None`` if absent.

    ``(cfg.get(section) or {})`` tolerates a section that is missing OR present
    but null (``validation:`` with no body) — the two shapes that broke the bash
    grep. A present-but-non-int value is a config error and is left to surface
    downstream rather than silently coerced.

    ``block`` is the sub-block the leaf lives in today and is read first;
    ``legacy_key`` is its pre-decomposition flat spelling, read only if the
    canonical location is absent. Both are needed because this is a raw YAML
    read *on purpose* — the cap must see the DECLARATION, before the loader
    supplies a default — so nothing folds the legacy spelling on our behalf.

    ``validation.eval_interval`` became ``validation.schedule.interval_steps``
    on 2026-07-31, after which a migrated arm answered ``None`` here. That is the
    safe direction for the cap itself (absent reads as "cap it"), but it left the
    "already fires within the capped run" branch unreachable and turned the
    documented cap-not-inflate contract into an INFLATE for any arm declaring a
    smaller interval than the cap.
    """
    body = cfg.get(section) or {}
    if block is not None:
        nested = (body.get(block) or {}).get(key)
        if nested is not None:
            return int(nested)
    value = body.get(key)
    if value is None and legacy_key is not None:
        value = body.get(legacy_key)
    return None if value is None else int(value)


def compute_iter_cap_overrides(
    config_path: str | Path, train_iters: int
) -> tuple[list[str], list[str]]:
    """Compute the ``--override`` args for a ``TRAIN_ITERS`` smoke cap.

    Caps BOTH ``training.max_iterations`` AND ``validation.eval_interval`` so a
    short run actually reaches a validation step — capping iterations alone is a
    trap (most arms set ``eval_interval`` to 2500, so a 100-iter run never
    validates; pitfall #10). Both are CAPS, not forced values: an arm declaring a
    smaller budget (a 1-shot calibration) is never inflated.

    Returns ``(override_args, messages)`` — ``override_args`` is a flat
    ``["--override", "k=v", …]`` list ready to splat into the train command;
    ``messages`` are human ``[TRAIN] …`` lines explaining each cap decision.
    """
    cfg = yaml.safe_load(Path(config_path).read_text()) or {}
    cfg_iters = _read_capped_key(cfg, "training", "max_iterations")
    cfg_eval = _read_capped_key(
        cfg,
        "validation",
        "interval_steps",
        block="schedule",
        legacy_key="eval_interval",
    )

    overrides: list[str] = []
    messages: list[str] = []

    # max_iterations: cap-not-inflate.
    if cfg_iters is None or cfg_iters > train_iters:
        eff_iters = train_iters
        overrides += ["--override", f"training.max_iterations={eff_iters}"]
    else:
        eff_iters = cfg_iters
        messages.append(
            f"[TRAIN] keeping config max_iterations={cfg_iters} (<= cap "
            f"{train_iters}); short/calibration arm not inflated"
        )

    # eval_interval: shrink to the effective iteration count so validation fires
    # at least once (on the final step), unless it already fires within the run.
    if cfg_eval is None or cfg_eval > eff_iters:
        overrides += ["--override", f"validation.eval_interval={eff_iters}"]
        shown = "unset" if cfg_eval is None else cfg_eval
        messages.append(
            f"[TRAIN] capping validation.eval_interval {shown}->{eff_iters} so "
            "validation runs within the capped run"
        )
    else:
        messages.append(
            f"[TRAIN] keeping config eval_interval={cfg_eval} (<= {eff_iters}); "
            "already fires within the capped run"
        )
    return overrides, messages


def _build_train_args(config: str, overrides: list[str], resume: bool) -> list[str]:
    args = ["train", "--config", config]
    if resume:
        args += ["--resume", "auto"]
    args += overrides
    return args


def output_dir_override(output_base: str | Path, config: str | Path) -> list[str]:
    """Return the ``--override training.output_dir=<base>/<stem>`` args.

    Routes THIS task's checkpoints / CSVs into a campaign tree keyed by the
    config's file stem, so the campaign orchestrator's ``_discover_results``
    (which globs ``<output_dir>/checkpoints/*.pt``) finds them. The orchestrator
    records the identical ``<base>/<stem>`` as the arm's ``output_dir``; the two
    MUST agree, so the path is computed the same way on both sides.
    """
    stem = Path(config).stem
    return ["--override", f"training.output_dir={Path(output_base) / stem}"]


def _dispatch(
    *,
    manifest: str,
    index: int,
    train_iters: int | None,
    resume: bool,
    no_audit: bool,
    dispatch_dir: str,
    dry_run: bool,
    prod: bool = False,
    output_base: str | None = None,
) -> int:
    # --prod runs the actual experiment (FULL config max_iterations). A smoke cap
    # (--train-iters) is the opposite intent, so combining them is a contradiction
    # — fail loud rather than silently let one win (CLAUDE.md #9/#15).
    if prod and train_iters is not None:
        print(
            "FATAL: --prod is mutually exclusive with --train-iters "
            f"(got --train-iters={train_iters}). --prod runs the FULL config "
            "max_iterations (the actual experiment); drop the smoke cap to run "
            "production.",
            file=sys.stderr,
        )
        return 2

    try:
        config = resolve_manifest_config(manifest, index)
    except IndexError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1

    name = Path(config).stem
    print(f"[manifest_dispatch] task {index}: {config}")
    # Run-mode provenance (stamped into the SLURM .out — #15): make the
    # smoke-vs-production decision explicit and auditable, not inferred from the
    # absence of a cap.
    if prod:
        print("[MODE] PROD — actual experiment: full config max_iterations, no smoke cap")
    elif train_iters is not None:
        print(f"[MODE] SMOKE — capped at TRAIN_ITERS={train_iters}")
    else:
        print("[MODE] default — full config max_iterations (no --prod, no smoke cap)")

    if not Path(config).is_file():
        print(f"FATAL: config file missing: {config}", file=sys.stderr)
        return 1

    overrides: list[str] = []
    if train_iters is not None:
        overrides, messages = compute_iter_cap_overrides(config, train_iters)
        for msg in messages:
            print(msg)

    # Route output into the campaign tree (campaign array mode). Appended after
    # any smoke-cap overrides so both land; absent (None) leaves the config's own
    # output_dir untouched (standalone array dispatcher path).
    if output_base is not None:
        out_override = output_dir_override(output_base, config)
        overrides += out_override
        print(f"[OUTPUT] routing into {out_override[-1].split('=', 1)[1]}")

    if dry_run:
        print(f"[DRYRUN] would audit + train: {config}")
        if overrides:
            print(f"[DRYRUN] train overrides: {' '.join(overrides)}")
        return 0

    py = [sys.executable, "-m", "spectramr.cli"]

    if not no_audit:
        Path(dispatch_dir).mkdir(parents=True, exist_ok=True)
        audit_json = Path(dispatch_dir) / f"audit_{name}_{index}.json"
        print(f"[AUDIT] spectramr.cli audit {config} --strict")
        with audit_json.open("w") as fh:
            audit_rc = subprocess.run([*py, "audit", config, "--strict"], stdout=fh).returncode
        if audit_rc != 0:
            print(
                f"AUDIT FAILED for {name} - skipping train (see {audit_json})",
                file=sys.stderr,
            )
            return 2
        print("[AUDIT] passed")

    train_args = _build_train_args(config, overrides, resume)
    print(f"[TRAIN] spectramr.cli {' '.join(train_args)}")
    return subprocess.run([*py, *train_args]).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="manifest_dispatch",
        description="Resolve + audit + train one experiment from a SLURM array manifest.",
    )
    parser.add_argument("--manifest", required=True, help="Newline-separated config list.")
    parser.add_argument("--index", type=int, required=True, help="0-based array task index.")
    parser.add_argument(
        "--train-iters",
        type=int,
        default=None,
        help="Smoke cap on BOTH training.max_iterations and validation.eval_interval.",
    )
    parser.add_argument(
        "--prod",
        action="store_true",
        help=(
            "Run the actual experiment: the FULL config max_iterations with no "
            "smoke cap. Mutually exclusive with --train-iters (errors if both given)."
        ),
    )
    parser.add_argument("--resume", action="store_true", help="Append --resume auto.")
    parser.add_argument("--no-audit", action="store_true", help="Skip the audit pre-flight.")
    parser.add_argument("--dispatch-dir", default=_DEFAULT_DISPATCH_DIR)
    parser.add_argument(
        "--output-base",
        default=None,
        help=(
            "Route this task's training.output_dir to <output-base>/<config-stem> "
            "(campaign array mode). Default None keeps the config's own output_dir."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved config + planned cap/command, then exit 0 (no torch).",
    )
    args = parser.parse_args(argv)
    return _dispatch(
        manifest=args.manifest,
        index=args.index,
        train_iters=args.train_iters,
        resume=args.resume,
        no_audit=args.no_audit,
        dispatch_dir=args.dispatch_dir,
        dry_run=args.dry_run,
        prod=args.prod,
        output_base=args.output_base,
    )


if __name__ == "__main__":  # pragma: no cover - module CLI entry
    raise SystemExit(main())
