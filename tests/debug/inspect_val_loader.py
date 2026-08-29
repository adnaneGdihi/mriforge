"""Hand-run probe: does an arm's VALIDATION loader actually undersample?

Answers one question for one config -- is the validation input distinguishable
from the validation target, or is the arm silently validating on fully-sampled
data (which reports a flattering PSNR that no reconstruction earned)?

**Not a pytest module.** ``testpaths = ["tests"]`` puts this directory in
pytest's walk, but ``python_files`` is the default ``test_*.py``, so nothing
here is collected. Run it by hand::

    python tests/debug/inspect_val_loader.py                    # default arm
    python tests/debug/inspect_val_loader.py --config <arm.yaml>

Exit code is meaningful: ``0`` clean, ``1`` a load/build failure OR a critical
finding (input == target, or an all-ones mask). It used to ``print`` every
failure and fall off the end at ``0`` -- a debug tool that reports breakage
with a green exit code is the silent-failure shape this repo refuses
(CLAUDE.md non-negotiable 3), and it is why the two staleness defects below sat
unnoticed: the script "ran fine" while doing nothing.

Fixes #1280, which found this file stale on two axes -- a config path from
before the ``experiments/inprogress/<paradigm>/`` reorg, and a read of
``data.validation_split``, retired to ``data.split.validation_fraction``.
Both are now mechanically pinned by ``tests/unit/debug/test_inspect_val_loader.py``
rather than left to rot again.

Requires the dataset on disk. It builds real loaders and fetches a real batch,
so it is a cluster/workstation tool, not a CI one.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import torch

# The package lives at <repo>/src/mriforge, so putting the REPO ROOT on the path
# -- which is what the original `sys.path.append(os.getcwd())` did -- never made
# `mriforge` importable. It worked only when the venv already had the editable
# install, and it silently depended on being launched from the repo root.
# Anchor on __file__ instead and add `src`, matching how every committed sbatch
# script sets PYTHONPATH (submit_exp11_fpk_ablation.sbatch:111).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mriforge.config.settings import TrainingSettings  # noqa: E402
from mriforge.infrastructure.builders.context import BuilderContext  # noqa: E402
from mriforge.infrastructure.builders.directors.data_pipeline_director import (  # noqa: E402
    DataPipelineDirector,
)

#: Default arm. The pre-reorg `experiments/inprogress/experiment_11_*.yaml` this
#: used to name has not existed since the paradigm-subdirectory reorg, so every
#: run of this script died at `from_yaml` and printed a green exit code (#1280).
DEFAULT_CONFIG = (
    _REPO_ROOT
    / "experiments/inprogress/kspace_filling/experiment_11_kspace_cold_diffusion.yaml"
)

#: Every dotted config path this script reads, declared so a test can assert
#: none of them is a retired spelling in `config.schemas.renames.RENAMES`.
#: Declaring them beats grepping the source: the check then fails on the NEXT
#: rename too, instead of only on the one that has already been fixed.
INSPECTED_CONFIG_PATHS: tuple[str, ...] = ("data.split.validation_fraction",)


def _read_config_path(config: TrainingSettings, dotted: str) -> object:
    """Read one dotted path off the frozen settings SSOT (never re-parse YAML)."""
    node: object = config
    for part in dotted.split("."):
        node = getattr(node, part)
    return node


def _resolve_keys(batch: dict) -> tuple[str | None, str | None]:
    """Pick the input/target keys present in this batch, preferring exact names.

    The original walked every key and let the LAST match win, so a batch with
    both `target` and `target_kspace` resolved non-deterministically by dict
    order. Prefer an explicit candidate list, first match wins.
    """
    input_candidates = ("input", "kspace", "input_kspace", "undersampled_kspace")
    target_candidates = ("target", "target_kspace", "full_kspace", "ground_truth")
    inp = next((k for k in input_candidates if k in batch), None)
    tgt = next((k for k in target_candidates if k in batch), None)
    return inp, tgt


def inspect_validation_data(config_path: Path) -> int:
    """Build the val loader for ``config_path`` and compare input vs target.

    Returns a process exit code: 0 clean, 1 on failure or a critical finding.
    """
    print(f"Loading config from {config_path}...")
    if not config_path.is_file():
        print(f"FAILED: config file does not exist: {config_path}", file=sys.stderr)
        return 1
    try:
        config = TrainingSettings.from_yaml(str(config_path))
    except Exception:
        print(f"FAILED to load config: {config_path}", file=sys.stderr)
        traceback.print_exc()
        return 1

    for dotted in INSPECTED_CONFIG_PATHS:
        print(f"  {dotted} = {_read_config_path(config, dotted)}")

    # Route through the maintained director, NOT `DataBuilder(...).build_train_val_loaders()`.
    # The old comment here read "call the chain methods manually as director
    # does" -- i.e. it was an admitted second copy of the director's build order
    # (CLAUDE.md non-negotiable 17). A debug tool that reconstructs the pipeline
    # by hand cannot answer questions about the pipeline production actually
    # builds, which is the only thing anyone runs this for.
    #
    # num_workers=0: keep decoding in-process so a dataset error surfaces as a
    # readable traceback here instead of a worker-crash message.
    print("Building data loaders via DataPipelineDirector...")
    try:
        director = DataPipelineDirector(BuilderContext(config=config))
        _train_loader, val_loader = director.build_dataloaders(num_workers=0)
    except Exception:
        print("FAILED to build loaders", file=sys.stderr)
        traceback.print_exc()
        return 1

    if val_loader is None:
        print("FAILED: director returned no validation loader.", file=sys.stderr)
        return 1

    print(f"Validation loader built. Batch size: {val_loader.batch_size}")
    print(f"Number of batches: {len(val_loader)}")

    print("Fetching first batch...")
    try:
        batch = next(iter(val_loader))
    except Exception:
        print("FAILED to fetch batch", file=sys.stderr)
        traceback.print_exc()
        return 1

    if not isinstance(batch, dict):
        print(f"FAILED: batch is {type(batch)}, expected a dict.", file=sys.stderr)
        return 1

    print("Batch keys:", sorted(batch.keys()))
    input_key, target_key = _resolve_keys(batch)
    if input_key is None or target_key is None:
        print(
            f"FAILED: could not resolve input/target keys "
            f"(input={input_key!r}, target={target_key!r}) from {sorted(batch.keys())}",
            file=sys.stderr,
        )
        return 1

    print(f"Comparing '{input_key}' vs '{target_key}'...")
    inp, tgt = batch[input_key], batch[target_key]
    if not (isinstance(inp, torch.Tensor) and isinstance(tgt, torch.Tensor)):
        print(
            f"FAILED: input is {type(inp)}, target is {type(tgt)}; expected tensors.",
            file=sys.stderr,
        )
        return 1

    print(f"Input shape:  {tuple(inp.shape)}")
    print(f"Target shape: {tuple(tgt.shape)}")

    critical: list[str] = []
    if inp.shape == tgt.shape and torch.equal(inp, tgt):
        critical.append(
            "input and target are IDENTICAL -- validation is not undersampling"
        )
        print(f"CRITICAL: {critical[-1]}")
    else:
        diff = (inp - tgt).abs()
        print(f"MSE(input, target): {(diff**2).mean().item():.6f}")
        print(f"Max Diff:           {diff.max().item():.6f}")

        if "mask" in batch:
            mask = batch["mask"]
            print(f"Mask shape: {tuple(mask.shape)}")
            if bool(torch.all(mask == 1)):
                critical.append("mask is all ONEs -- no undersampling applied")
                print(f"CRITICAL: {critical[-1]}")
            else:
                print("Mask contains zeros (undersampling active).")
                consistency = (inp - tgt * mask).abs().mean().item()
                print(f"Consistency mean|input - target*mask|: {consistency:.6f}")
        else:
            print("No 'mask' key in batch; cannot check data consistency.")

    if critical:
        print(f"\n{len(critical)} critical finding(s).", file=sys.stderr)
        return 1
    print("\nNo critical findings.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"experiment YAML to inspect (default: {DEFAULT_CONFIG})",
    )
    args = parser.parse_args(argv)
    return inspect_validation_data(args.config)


if __name__ == "__main__":
    sys.exit(main())
