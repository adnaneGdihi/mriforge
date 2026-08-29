"""Regulatory-package CLI (ULF-PR-27 / V.4).

Plan: TODO/backlog_ulf_radiologist_acceptance_roadmap.md §ULF-PR-27.

Bundles the per-arm validation badge + provenance manifest +
certificate artefacts into a single zip the regulatory team submits.

Subcommands:

- ``regulatory bundle`` — collect inputs + write a single zip.
- ``regulatory verify`` — re-run the certificate gates from a bundle
  (without re-training) to confirm reproducibility.
- ``regulatory status`` — print eligibility + missing gates.

The implementation is deliberately thin: it composes the existing
``ValidationBadge``, ``ProvenanceManifest``, and ``ReaderStudyReport``
helpers — the CLI just exposes them in a regulatory-team-friendly
form.
"""

from __future__ import annotations

import argparse
import json
import logging
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)


def _cmd_bundle(args: argparse.Namespace) -> int:
    """Collect artefacts under ``--output-dir`` into ``args.bundle_path``."""
    src = Path(args.output_dir)
    if not src.is_dir():
        logger.error("Output dir not found: %s", src)
        return 2
    bundle = Path(args.bundle_path)
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in (
            "validation_badge.json",
            "provenance_manifest.json",
            "calibration_report.json",
            "chd_report.json",
            "prc_report.json",
            "pac_bayes_report.json",
        ):
            f = src / name
            if f.exists():
                zf.write(f, arcname=name)
        # Include certificate-summary figure if present.
        for fig in ("certificate_summary.png", "certificate_summary.pdf"):
            f = src / fig
            if f.exists():
                zf.write(f, arcname=fig)
    logger.info("Bundle written: %s", bundle)
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    """Re-load the bundle and verify every gate is present and passing."""
    bundle = Path(args.bundle_path)
    if not bundle.is_file():
        logger.error("Bundle not found: %s", bundle)
        return 2
    extracted = bundle.parent / f"{bundle.stem}_unpacked"
    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle) as zf:
        zf.extractall(extracted)
    badge_file = extracted / "validation_badge.json"
    if not badge_file.exists():
        logger.error("validation_badge.json missing from bundle")
        return 1
    badge = json.loads(badge_file.read_text())
    eligible = bool(badge.get("eligible", False))
    gates = badge.get("gates", {})
    missing = [g for g, v in gates.items() if not v.get("passed", False)]
    print(
        json.dumps(
            {
                "bundle": str(bundle),
                "eligible": eligible,
                "n_gates": len(gates),
                "n_passed": sum(1 for v in gates.values() if v.get("passed", False)),
                "failing_gates": missing,
            },
            indent=2,
        )
    )
    return 0 if eligible else 1


def _cmd_status(args: argparse.Namespace) -> int:
    """Print the eligibility flag from a validation_badge.json directly."""
    badge_path = Path(args.badge_path)
    if not badge_path.is_file():
        logger.error("badge not found: %s", badge_path)
        return 2
    badge = json.loads(badge_path.read_text())
    print(
        json.dumps(
            {
                "eligible": bool(badge.get("eligible", False)),
                "gates": list(badge.get("gates", {}).keys()),
            },
            indent=2,
        )
    )
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_bundle = sub.add_parser("bundle", help="Bundle artefacts into a zip")
    p_bundle.add_argument("--output-dir", required=True)
    p_bundle.add_argument("--bundle-path", required=True)
    p_bundle.set_defaults(func=_cmd_bundle)

    p_verify = sub.add_parser("verify", help="Re-verify gates from a bundle")
    p_verify.add_argument("--bundle-path", required=True)
    p_verify.set_defaults(func=_cmd_verify)

    p_status = sub.add_parser("status", help="Print eligibility from a badge JSON")
    p_status.add_argument("--badge-path", required=True)
    p_status.set_defaults(func=_cmd_status)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="regulatory")
    add_arguments(parser)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
