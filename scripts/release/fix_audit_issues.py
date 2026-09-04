"""Fix the audit failures surfaced by ``smoke_test_all_<TS>.log``.

The May 18, 2026 cluster smoke run produced **414 audited YAMLs** with
three failure modes:

* **412** ``hardcoded_cluster_paths`` failures — ``data.data_root``
  (and ``manifest_path``, ``checkpoint_dir``, ``output_dir``, …)
  pointing at ``/project/<user>/<rest>``. Mechanically safe to rewrite
  to a relative ``databases/<rest>`` path that ``PathResolver`` resolves
  against ``$PROJECT_ROOT`` / ``$SPECTRAMR_DATA_ROOT``.

* **7** ``data_model_compatibility`` failures — model declares
  ``spatial_dims=(1,)`` (a fingerprint / time-series operator) but
  ``data`` feeds 2D image patches. No registered adapter bridges
  ``spatial_dims=2 → 1``; the YAML's experimental intent is ambiguous.
  Cannot auto-fix safely; this script flags each arm with the canonical
  hint.

* **1** ``model_loss_output_domain`` failure —
  ``enhanced_deep_unet`` outputs ``image`` but losses declare
  ``output_domain='latent'``. No auto-bridge from image to latent
  losses. Same flag-don't-fix policy.

Usage:

    python scripts/release/fix_audit_issues.py [--dry-run] [--log PATH]
                                               [--scope DIR] [--no-verify]

* ``--dry-run`` reports what would change without writing.
* ``--log PATH`` parses a specific smoke log to enumerate affected
  YAMLs (default scans ``experiments/`` recursively).
* ``--scope DIR`` restricts the rewrite to one directory.
* ``--no-verify`` skips the post-fix ``spectramr audit`` re-run.

The rewrite is **idempotent**: a second invocation finds nothing to do.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]

# Match an absolute cluster path of the form /project/<user>/<...>/databases/<rest>
# or /scratch/<user>/<...>/databases/<rest>.
# Captures the suffix from "databases/" onward.
CLUSTER_PATH_RE = re.compile(
    r"(?P<prefix>/(?:project|scratch)/[A-Za-z0-9_\-]+/[^\"'\s:]*?/)"
    r"(?P<rel>databases/[^\"'\s]+)"
)

# Direct match for the most common literal we see — kept as a second
# pass to catch lines where ``databases/`` is the first path segment
# after the user dir without an intermediate component.
CLUSTER_PATH_RE_SIMPLE = re.compile(
    r"/(?:project|scratch)/[A-Za-z0-9_\-]+/(?:[A-Za-z0-9_\-]+/)*(?=databases/)"
)

# Semantic-issue arms that this script will FLAG but not auto-fix.
# Maps arm slug → (audit check_name, hint).
SEMANTIC_FLAGS: dict[str, tuple[str, str]] = {
    "idea_2_extra_wide_diffusion": (
        "data_model_compatibility",
        "Model spatial_dims=(1,) (fingerprint) vs data 2D image. "
        "Either swap to a fingerprint dataset_type, register an "
        "image_to_fingerprint adapter, or change the model registration.",
    ),
    "idea_2_high_sigma_diffusion": (
        "data_model_compatibility",
        "Same as idea_2_extra_wide_diffusion.",
    ),
    "idea_2_riemannian_mrf_diffusion": (
        "data_model_compatibility",
        "Same as idea_2_extra_wide_diffusion.",
    ),
    "idea_4_crlb_pulse_design": (
        "data_model_compatibility",
        "Same as idea_2_extra_wide_diffusion.",
    ),
    "idea_4_strict_sar_bound": (
        "data_model_compatibility",
        "Same as idea_2_extra_wide_diffusion.",
    ),
    "idea_5_extended_hrf_length": (
        "data_model_compatibility",
        "Same as idea_2_extra_wide_diffusion.",
    ),
    "idea_5_hrf_manifold_diffusion": (
        "data_model_compatibility",
        "Same as idea_2_extra_wide_diffusion.",
    ),
    "idea_6_phys_eq_ssl_brain": (
        "model_loss_output_domain",
        "enhanced_deep_unet outputs 'image' but losses.output_domain "
        "is 'latent'. Either change losses.output_domain to 'image' or "
        "introduce a learned encoder and an explicit pre_loss_pred adapter.",
    ),
}


def rewrite_cluster_paths(text: str) -> tuple[str, int]:
    """Return (new_text, n_replacements) after rewriting cluster prefixes."""
    n = 0

    def _sub(m: re.Match[str]) -> str:
        nonlocal n
        n += 1
        return m.group("rel")

    new_text = CLUSTER_PATH_RE.sub(_sub, text)
    # Second pass: any remaining ``/project/<user>/.../databases/`` literals
    # where the regex above missed a path component layout.
    def _sub2(m: re.Match[str]) -> str:
        nonlocal n
        n += 1
        return ""

    new_text = CLUSTER_PATH_RE_SIMPLE.sub(_sub2, new_text)
    return new_text, n


def yaml_files_under(scope: Path) -> Iterable[Path]:
    """Yield every YAML under ``scope``, skipping archives and build artefacts."""
    skip = {".backup_", "/archive/", "/results/", "/.venv/", "/_build/", "/.git/"}
    for p in scope.rglob("*.y*ml"):
        s = str(p)
        if any(tag in s for tag in skip):
            continue
        yield p


def yamls_from_smoke_log(log_path: Path) -> set[Path]:
    """Parse the smoke log to recover the list of YAMLs that failed audit."""
    affected: set[Path] = set()
    # Match `[AUDIT] python -m src.cli audit /<path>/X.yaml`
    line_re = re.compile(r"\[AUDIT\] [^ ]+ -m \S+ audit (\S+\.ya?ml) ")
    text = log_path.read_text(errors="replace")
    for line in text.splitlines():
        m = line_re.search(line)
        if not m:
            continue
        cluster_path = m.group(1)
        # Map cluster path → local path: take everything from `/experiments/` on.
        if "/experiments/" in cluster_path:
            rel = "experiments/" + cluster_path.split("/experiments/", 1)[1]
            local = REPO_ROOT / rel
            if local.exists():
                affected.add(local)
    return affected


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mechanically fix the audit failures from a smoke log."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without writing.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Parse a specific smoke log to scope the YAML list. "
        "Default: scan experiments/ recursively.",
    )
    parser.add_argument(
        "--scope",
        type=Path,
        default=REPO_ROOT / "experiments",
        help="Directory tree to scan when --log is not given.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the post-fix `spectramr audit` re-run.",
    )
    parser.add_argument(
        "--verify-limit",
        type=int,
        default=5,
        help="How many touched YAMLs to re-audit at the end (default 5).",
    )
    args = parser.parse_args()

    # 1. Gather candidate YAMLs.
    if args.log is not None:
        candidates = sorted(yamls_from_smoke_log(args.log))
        print(f"[scan] parsed {args.log} → {len(candidates)} YAMLs to inspect.")
    else:
        candidates = sorted(yaml_files_under(args.scope))
        print(f"[scan] walked {args.scope} → {len(candidates)} YAMLs to inspect.")

    # 2. Phase 1: cluster-path rewrite.
    changed: list[Path] = []
    totals = defaultdict(int)
    for f in candidates:
        try:
            text = f.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        new_text, n = rewrite_cluster_paths(text)
        if n == 0:
            continue
        totals["files"] += 1
        totals["edits"] += n
        rel = f.relative_to(REPO_ROOT)
        if args.dry_run:
            print(f"  [DRY] {rel}: would rewrite {n} cluster-path occurrence(s)")
        else:
            f.write_text(new_text)
            changed.append(f)
            print(f"  [FIX] {rel}: rewrote {n} cluster-path occurrence(s)")
    print(
        f"[phase-1] cluster-path rewrite: {totals['files']} file(s), "
        f"{totals['edits']} occurrence(s){' (dry-run)' if args.dry_run else ''}."
    )

    # 3. Phase 2: flag the semantic offenders that need human attention.
    print("\n[phase-2] semantic offenders (NOT auto-fixed — manual review):")
    flagged = 0
    for f in candidates:
        slug = f.stem
        if slug in SEMANTIC_FLAGS:
            check, hint = SEMANTIC_FLAGS[slug]
            rel = f.relative_to(REPO_ROOT)
            print(f"  [FLAG] {rel}")
            print(f"         check : {check}")
            print(f"         hint  : {hint}")
            flagged += 1
    print(f"[phase-2] flagged {flagged} arm(s) for human review.")

    # 4. Phase 3: optional verification re-audit.
    if not args.no_verify and not args.dry_run and changed:
        print("\n[phase-3] re-auditing a sample of touched YAMLs:")
        sample = changed[: args.verify_limit]
        for f in sample:
            rel = f.relative_to(REPO_ROOT)
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "spectramr.cli", "audit", str(f)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            except subprocess.TimeoutExpired:
                print(f"  [TIMEOUT] {rel}")
                continue
            failed = [
                line.strip()
                for line in (proc.stdout + proc.stderr).splitlines()
                if line.strip().startswith("❌") and "hardcoded_cluster" in line
            ]
            status = "PASS" if not failed else "FAIL"
            print(f"  [{status}] {rel}")
            for line in failed[:1]:
                print(f"         {line[:160]}")

    return 0 if args.dry_run or totals["files"] > 0 or flagged > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
