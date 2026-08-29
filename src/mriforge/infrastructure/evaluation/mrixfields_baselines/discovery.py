from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)
MODALITIES = ["T1W", "T2W", "T2FLAIR"]
FIELDS = ["0.1T", "1.5T", "3T", "5T", "7T"]
FIELD_DIRS = {"0.1T": 0.1, "1.5T": 1.5, "3T": 3.0, "5T": 5.0, "7T": 7.0}
CONTRAST_DIR_TO_KEY = {"T1W": "T1w", "T2W": "T2w", "T2FLAIR": "T2FLAIR"}
_T12 = re.compile(
    r"^task(?P<task>[12])_(?P<src>[\d.]+)T_to_(?P<tgt>[\d.]+)T_(?P<c>T1W|T2W|T2FLAIR)$"
)
_T3 = re.compile(r"^task3_any_to_any_multimodal$")
_METHODS: dict[int, tuple[str, ...]] = {
    1: ("cut", "cyclegan"),
    2: ("cut", "cyclegan"),
    3: ("stargan_v2",),
}


def joint_domain(contrast_dir: str, field_dir: str) -> int:
    return MODALITIES.index(contrast_dir) * len(FIELDS) + FIELDS.index(field_dir)


def _field_dir(value: float) -> str:
    """Convert a float field strength (e.g. 0.1) to its dir string (e.g. '0.1T')."""
    return next(k for k, v in FIELD_DIRS.items() if abs(v - value) < 1e-9)


@dataclass(frozen=True)
class BaselineSpec:
    task: int
    method: str
    source_field: float | None
    target_field: float | None
    contrast: str | None
    checkpoint: Path
    name: str


@dataclass(frozen=True)
class EvalTask:
    spec: BaselineSpec
    source_field: float
    target_field: float
    contrast: str
    target_domain: int | None
    label: str


def _find_ckpt(method_dir: Path) -> Path | None:
    hits = sorted(method_dir.glob("pro_pretrained/weights/*.pth"))
    if not hits:
        return None
    if len(hits) > 1:
        # More than one checkpoint under weights/ — the choice is deterministic
        # (lexicographically first) but log it so an unexpected extra .pth (e.g. a
        # mid-training checkpoint) never silently changes which weights are scored.
        logger.warning(
            "%d checkpoints under %s/pro_pretrained/weights (%s); using %s",
            len(hits),
            method_dir.name,
            [h.name for h in hits],
            hits[0].name,
        )
    return hits[0]


def discover_baselines(root: Path) -> list[BaselineSpec]:
    root = Path(root)
    specs: list[BaselineSpec] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        m12 = _T12.match(d.name)
        m3 = _T3.match(d.name)
        if m12:
            task = int(m12["task"])
            src = float(m12["src"])
            tgt = float(m12["tgt"])
            contrast = CONTRAST_DIR_TO_KEY[m12["c"]]
            for method in _METHODS[task]:
                ck = _find_ckpt(d / method)
                if ck is None:
                    logger.warning("no checkpoint under %s/%s — skipping", d.name, method)
                    continue
                specs.append(
                    BaselineSpec(task, method, src, tgt, contrast, ck, f"{d.name}__{method}")
                )
        elif m3:
            ck = _find_ckpt(d / "stargan_v2")
            if ck is None:
                logger.warning("no checkpoint under %s/stargan_v2 — skipping", d.name)
                continue
            specs.append(BaselineSpec(3, "stargan_v2", None, None, None, ck, d.name))
        # non-matching dirs (LICENSE, README, json) ignored
    if not specs:
        raise ValueError(f"no baseline checkpoints discovered under {root}")
    return specs


def _ordered_field_pairs(
    fields: list[float] | tuple[float, ...], mode: str
) -> list[tuple[float, float]]:
    fields = list(fields)
    if mode == "to7t":
        return [(s, 7.0) for s in fields if s != 7.0]
    if mode == "task1_task2":
        # {*->7.0} U {0.1->*}: dedup the (0.1, 7.0) pair shared by both halves.
        pairs = [(s, 7.0) for s in fields if s != 7.0] + [(0.1, t) for t in fields if t != 0.1]
        return list(dict.fromkeys(pairs))
    if mode == "all":
        return [(s, t) for s in fields for t in fields if s != t]
    raise ValueError(f"unknown task3_pairs={mode!r} (all|task1_task2|to7t)")


def build_eval_tasks(
    specs: list[BaselineSpec],
    *,
    contrasts: tuple[str, ...] | list[str],
    fields: tuple[float, ...] | list[float],
    task3_pairs: str = "all",
) -> list[EvalTask]:
    tasks: list[EvalTask] = []
    for spec in specs:
        if spec.method == "stargan_v2":
            for contrast in contrasts:
                cdir = next(k for k, v in CONTRAST_DIR_TO_KEY.items() if v == contrast)
                for s, t in _ordered_field_pairs(fields, task3_pairs):
                    td = joint_domain(cdir, _field_dir(t))
                    tasks.append(
                        EvalTask(
                            spec,
                            s,
                            t,
                            contrast,
                            td,
                            f"{spec.name}__{contrast}__{s}to{t}",
                        )
                    )
        else:
            tasks.append(
                EvalTask(
                    spec,
                    spec.source_field,  # type: ignore[arg-type]
                    spec.target_field,  # type: ignore[arg-type]
                    spec.contrast,  # type: ignore[arg-type]
                    None,
                    spec.name,
                )
            )
    return tasks
