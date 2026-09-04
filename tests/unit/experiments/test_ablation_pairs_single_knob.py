"""An ablation arm differs from its baseline in the knob it names, and in nothing else that confounds it.

geomamba_ulf on 2026-09-03: every ablation shared v0's loss weights (the two
topology ablations were v0 re-run under another name) and differed from v0 in
the rotation augmentation and the early-stopping block instead (#17, one knob
per comparison). The rule is corpus-wide and pinned here with planted pairs.

An ablation is an arm whose ``metadata.role`` (or ``metadata.tags.role``) is
``ablation`` and whose ``metadata.baseline`` (or ``metadata.tags.baseline``)
resolves to a sibling by ``metadata.name`` or file stem -- the same resolution
``tests/unit/config/test_inprogress_baseline_resolves.py`` applies. A baseline
that resolves to nothing is that test's finding, not this one's.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from tests.utils.corpus import tracked_yamls  # noqa: E402

#: Blocks whose difference between an ablation and its baseline is a confound
#: unless the ablation says it is the knob (``metadata.tags.ablated_component``).
_CONFOUND_BLOCKS = (("data", "augmentation"), ("early_stopping",), ("optimization",))
#: Blocks in which a real ablation shows its knob.
_SUBSTANTIVE_BLOCKS = ("losses", "model", "training", "physics", "data")


def _get(d: dict, path: tuple[str, ...]):
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _normalised(doc: dict, names: set[str]) -> dict:
    """The document without metadata and without the strings that carry its own name."""

    def walk(v):
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items() if k != "metadata"}
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, str) and any(n and n in v for n in names):
            return "<NAME>"
        return v

    return walk(doc)


def ablation_defects(ablation: dict, baseline: dict, names: set[str]) -> list[str]:
    """The findings for one (ablation, baseline) pair; empty when the pair is one-knob."""
    a = _normalised(ablation, names)
    b = _normalised(baseline, names)
    out: list[str] = []
    if json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str):
        return ["identical to its baseline outside metadata: the same experiment twice"]
    knob = str(((ablation.get("metadata") or {}).get("tags") or {}).get("ablated_component") or "")
    # ``data`` is substantive AND holds the augmentation confound, so the confound
    # sub-blocks are stripped from the substantive view: an ablation that differs
    # from its baseline in augmentation alone is the same experiment, not a knob --
    # unless its ablated_component names that block.
    sa, sb = _substantive_view(a), _substantive_view(b)
    substantive = [blk for blk in _SUBSTANTIVE_BLOCKS if sa[blk] != sb[blk]]
    named = [p for p in _CONFOUND_BLOCKS if p[-1] in knob and _get(a, p) != _get(b, p)]
    if not substantive and not named:
        out.append(
            "differs from its baseline in no substantive block (losses/model/training/physics/data) "
            "and in no block its ablated_component names"
        )
    for path in _CONFOUND_BLOCKS:
        if _get(a, path) != _get(b, path) and path[-1] not in knob:
            out.append(
                f"{'.'.join(path)} differs from the baseline's while the ablated component "
                f"is {knob!r}"
            )
    return out


def _substantive_view(doc: dict) -> dict:
    """The substantive blocks with the confound sub-blocks removed from them."""
    view: dict = {}
    for blk in _SUBSTANTIVE_BLOCKS:
        v = doc.get(blk)
        if isinstance(v, dict):
            v = {k: x for k, x in v.items() if (blk, k) not in _CONFOUND_BLOCKS}
        view[blk] = v
    return view


def _arm_id(path: Path) -> str:
    parts = path.parts
    return "/".join(parts[parts.index("inprogress") + 1 :]) if "inprogress" in parts else path.name


def _corpus() -> tuple[dict[Path, dict], dict[str, Path]]:
    docs: dict[Path, dict] = {}
    by_name: dict[str, Path] = {}
    for path in tracked_yamls("experiments/inprogress"):
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            continue  # malformed YAML is a different test's concern
        if not isinstance(doc, dict):
            continue
        docs[path] = doc
        md = doc.get("metadata")
        if isinstance(md, dict) and md.get("name"):
            by_name.setdefault(str(md["name"]), path)
        by_name.setdefault(path.stem, path)
    return docs, by_name


def _pairs() -> list:
    docs, by_name = _corpus()
    out = []
    for path, doc in docs.items():
        md = doc.get("metadata")
        if not isinstance(md, dict):
            continue
        tags = md.get("tags") if isinstance(md.get("tags"), dict) else {}
        if (tags.get("role") or md.get("role")) != "ablation":
            continue
        base = md.get("baseline") or tags.get("baseline")
        if not isinstance(base, str):
            continue
        bpath = by_name.get(base) or by_name.get(base.removesuffix(".yaml"))
        if bpath is not None and bpath != path:
            out.append(pytest.param(path, bpath, id=_arm_id(path)))
    return out


_PAIRS = _pairs()


@pytest.mark.skipif(not _PAIRS, reason="no ablation arm names a resolvable baseline")
@pytest.mark.parametrize(("ablation_path", "baseline_path"), _PAIRS)
def test_every_ablation_is_one_knob_against_its_baseline(
    ablation_path: Path, baseline_path: Path
) -> None:
    ablation = yaml.safe_load(ablation_path.read_text(encoding="utf-8"))
    baseline = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
    names = {ablation_path.stem, baseline_path.stem}
    for doc in (ablation, baseline):
        md = doc.get("metadata") or {}
        if isinstance(md, dict) and md.get("name"):
            names.add(str(md["name"]))
    assert ablation_defects(ablation, baseline, names) == []


# ---------------------------------------------------------------------------
# Planted pairs: one per shape the rule can take (non-negotiable 15)
# ---------------------------------------------------------------------------

_PLANTED = "planted_abl_arm"


def _arm(losses, aug=None, es=None, knob="x"):
    return {
        "metadata": {"name": _PLANTED, "tags": {"role": "ablation", "ablated_component": knob}},
        "losses": {"image_losses": losses},
        "model": {"model_type": "m"},
        "training": {"output_dir": f"results/{_PLANTED}"},
        "data": {"augmentation": aug or {"enabled": False}},
        "early_stopping": es or {"enabled": False},
    }


def test_planted_identical_pair_is_the_same_experiment_twice() -> None:
    """Planted violation: geomamba's abl_ph_active against v0 before 2026-09-03."""
    a = _arm([{"name": "l1", "weight": 10.0}, {"name": "cubical_ph_w2", "weight": 1.0}])
    b = _arm([{"name": "l1", "weight": 10.0}, {"name": "cubical_ph_w2", "weight": 1.0}])
    assert ablation_defects(a, b, {_PLANTED}) == [
        "identical to its baseline outside metadata: the same experiment twice"
    ]


def test_planted_confound_is_named() -> None:
    """Planted violation: the knob is the PH term, but the augmentation differs too."""
    a = _arm(
        [{"name": "l1", "weight": 10.0}],
        aug={"enabled": True, "enable_rotation": True},
        knob="cubical_ph_w2",
    )
    b = _arm(
        [{"name": "l1", "weight": 10.0}, {"name": "cubical_ph_w2", "weight": 1.0}],
        aug={"enabled": False},
    )
    defects = ablation_defects(a, b, {_PLANTED})
    assert any(d.startswith("data.augmentation differs") for d in defects)
    assert not any("early_stopping" in d for d in defects)


def test_planted_early_stopping_confound_is_named() -> None:
    a = _arm(
        [{"name": "l1", "weight": 10.0}], es={"enabled": True, "patience": 5}, knob="cubical_ph_w2"
    )
    b = _arm([{"name": "l1", "weight": 10.0}, {"name": "cubical_ph_w2", "weight": 1.0}])
    assert any(d.startswith("early_stopping differs") for d in ablation_defects(a, b, {_PLANTED}))


def test_planted_no_substantive_difference_is_named() -> None:
    """Planted violation: only a confound block differs -- a knob nobody named."""
    a = _arm([{"name": "l1", "weight": 10.0}], es={"enabled": True, "patience": 5})
    b = _arm([{"name": "l1", "weight": 10.0}])
    defects = ablation_defects(a, b, {_PLANTED})
    assert any(d.startswith("differs from its baseline in no substantive block") for d in defects)


def test_planted_augmentation_only_difference_is_the_same_experiment() -> None:
    """Planted violation: only data.augmentation differs and nothing names it -- the
    shape the committed geomamba ablations had against v0 (weights aside)."""
    a = _arm([{"name": "l1", "weight": 10.0}], aug={"enabled": True}, knob="cubical_ph_w2")
    b = _arm([{"name": "l1", "weight": 10.0}], aug={"enabled": False})
    defects = ablation_defects(a, b, {_PLANTED})
    assert any(d.startswith("differs from its baseline in no substantive block") for d in defects)
    assert any(d.startswith("data.augmentation differs") for d in defects)


def test_a_one_knob_pair_passes() -> None:
    a = _arm([{"name": "l1", "weight": 10.0}], knob="cubical_ph_w2")
    b = _arm([{"name": "l1", "weight": 10.0}, {"name": "cubical_ph_w2", "weight": 1.0}])
    assert ablation_defects(a, b, {_PLANTED}) == []


def test_a_named_confound_block_is_the_knob() -> None:
    """An augmentation ablation may differ in data.augmentation: it says so."""
    a = _arm([{"name": "l1", "weight": 10.0}], aug={"enabled": True}, knob="augmentation")
    b = _arm([{"name": "l1", "weight": 10.0}], aug={"enabled": False})
    assert ablation_defects(a, b, {_PLANTED}) == []


def test_names_do_not_count_as_a_difference() -> None:
    a = _arm([{"name": "l1", "weight": 10.0}])
    a["training"]["output_dir"] = "results/abl_x"
    b = _arm([{"name": "l1", "weight": 10.0}])
    b["training"]["output_dir"] = "results/base_y"
    assert ablation_defects(a, b, {_PLANTED, "abl_x", "base_y"})[0].startswith("identical")
