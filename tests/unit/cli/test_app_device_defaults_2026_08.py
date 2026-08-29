"""D01#4 / D01#5: no verb may hardcode a device, and ``report`` must carry a seed.

Four subparsers shipped ``--device default="cuda"``. argparse defaults are
applied BEFORE any config is read, so those verbs could never observe an absent
``--device``: ``run.device`` was unreachable for them, and the requirement was
unconditional -- ``resolve_compute_device`` treats an explicit ``"cuda"`` as a
hard requirement that ``FORCE_CPU`` does not relax, while ``None`` resolves to
``"auto"``.

``predict``'s source even documented the intended behaviour ("``None`` -> auto")
beside an argparse default that made ``None`` unreachable -- a facade
(pitfall #16), which is why this asserts the *default*, the thing the comment
was wrong about, and not the comment.
"""

import argparse

import pytest

from mriforge.cli.app import build_parser

# Every verb that exposes ``--device``. The contract is uniform: the flag is a
# CLI OVERRIDE, so its absence must be representable.
_DEVICE_VERBS = [
    "train",
    "sanity_check",
    "ablation",
    "infer",
    "infer-dataset",
    "experiment",
    "predict",
    "benchmark",
    "audit",
    "hpo",
    # `profile` forwards --device verbatim to the verb it profiles, so its
    # absence must stay representable exactly as for the target verb.
    "profile",
]

# Verbs that deliberately pin a device. Listed, not skipped, so a new hardcoded
# default cannot arrive unnoticed: any verb absent from BOTH lists fails
# ``test_every_device_verb_is_accounted_for`` below.
_PINNED_DEVICE_VERBS = {
    "meta-evaluate": "auto",  # resolves "auto" itself
    "design-mrf-sequence": "cpu",  # CPU-pinned design routine
}


def _device_action(name: str) -> argparse.Action:
    parser = build_parser()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction) and name in action.choices:
            for sub_action in action.choices[name]._actions:
                if "--device" in sub_action.option_strings:
                    return sub_action
            raise AssertionError(f"subcommand {name!r} has no --device")
    raise AssertionError(f"subcommand {name!r} not found")


@pytest.mark.parametrize("verb", _DEVICE_VERBS)
def test_device_default_is_none(verb):
    action = _device_action(verb)
    assert action.default is None, (
        f"`{verb} --device` defaults to {action.default!r}; a hardcoded device "
        "makes the config's own device unreachable and the accelerator "
        "requirement unconditional (non-negotiable 9b)."
    )


@pytest.mark.parametrize("verb", _DEVICE_VERBS)
def test_device_flag_still_accepts_an_override(verb):
    """The override itself must keep working -- the fix is about its absence."""
    action = _device_action(verb)
    assert action.nargs is None and action.const is None
    assert action.type in (None, str)


@pytest.mark.parametrize(("verb", "pinned"), sorted(_PINNED_DEVICE_VERBS.items()))
def test_pinned_device_verbs_keep_their_default(verb, pinned):
    assert _device_action(verb).default == pinned


def test_every_device_verb_is_accounted_for():
    """A new subcommand with a hardcoded ``--device`` must land in one of the
    two lists above rather than slipping past this file."""
    parser = build_parser()
    subs = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    with_device = {
        name
        for name, sub in subs.choices.items()
        if any("--device" in a.option_strings for a in sub._actions)
    }
    assert with_device == set(_DEVICE_VERBS) | set(_PINNED_DEVICE_VERBS)


def test_ablation_does_not_reimpose_cuda_after_argparse():
    """The ``or "cuda"`` at the ablation call site defeated the parser default.

    Checked on the AST of the call, not on the source text: a text search also
    matches the comment that explains the removal, and comments move.
    """
    import ast
    import inspect
    import textwrap

    from mriforge.cli import app

    tree = ast.parse(textwrap.dedent(inspect.getsource(app.ablation)))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "run_ablation_study"
    ]
    assert len(calls) == 1, "ablation() no longer has exactly one study call"
    device_kw = next(k for k in calls[0].keywords if k.arg == "device")
    assert not isinstance(device_kw.value, ast.BoolOp), (
        "ablation forwards `<device> or <fallback>`; an unset --device must "
        "stay None so the config and the 9b resolver decide."
    )


def test_report_seed_reads_the_canonical_run_seed():
    """D01#5: the reader used the LEGACY top-level ``settings.seed``, which
    ``RENAMES`` retired to ``run.seed``. It could never resolve, so every
    report without ``--seed`` stamped ``seed: null`` instead of the seed the
    run actually used."""
    import inspect

    from mriforge.cli import app

    src = inspect.getsource(app)
    assert 'getattr(settings, "seed", None)' not in src
    assert 'kwargs["seed"] = settings.run.seed' in src


# ---------------------------------------------------------------------------
# The Tier-2 probe gate resolves a device too, and it had the same defect in a
# subtler form: it read ``run.device`` as a plain attribute. That field carries
# a schema default of ``"cuda"``, so the chain resolved to ``"cuda"`` for every
# arm that declared no device -- leaving the documented ``auto`` leg
# unreachable, ``FORCE_CPU`` inert (an explicit ``"cuda"`` is the one request it
# may not relax), and the fourth leg (the top-level ``device`` attribute, which
# ``RENAMES`` retired and which is not a schema field) permanently ``None``.
# The gate now shares ``main._declared_device`` -- one owner for the question
# "did the YAML declare a device?" (non-negotiable 17).
# ---------------------------------------------------------------------------


class _Report:
    def __init__(self):
        self.results = []


def _settings(**kw):
    from mriforge.config.settings import TrainingSettings

    return TrainingSettings(model={}, data={}, optimization={}, logging={}, **kw)


def _capture_request(monkeypatch):
    """Spy on what the gate hands the 9b resolver."""
    from mriforge.cli import app

    seen = {}

    def _fake(requested, *, pipeline, source="unspecified", **kw):
        seen["requested"] = requested
        seen["source"] = source
        raise ValueError("stop here — the request is what this test is about")

    monkeypatch.setattr(app, "resolve_torch_device", _fake)
    return seen


@pytest.mark.parametrize(
    ("kwargs", "cli_device", "expected_requested", "expected_source"),
    [
        ({}, None, None, "auto"),
        ({"run": {"device": "cpu"}}, None, "cpu", "run.device"),
        ({"run": {"device": "cuda"}}, None, "cuda", "run.device"),
        ({}, "cpu", "cpu", "cli"),
        ({"run": {"device": "cuda"}}, "cpu", "cpu", "cli"),
    ],
)
def test_probe_gate_request_and_source(
    monkeypatch, kwargs, cli_device, expected_requested, expected_source
):
    from mriforge.cli.app import _gate_probe_acceleration

    seen = _capture_request(monkeypatch)
    args = argparse.Namespace(probe=True, device=cli_device)
    _gate_probe_acceleration(args, _settings(**kwargs), _Report())
    assert seen["requested"] == expected_requested
    assert seen["source"] == expected_source, (
        "the source label is stamped into the DeviceDecision and into "
        "provenance; naming the wrong knob misdirects triage"
    )


def test_probe_gate_labels_training_device_as_its_own_source(monkeypatch):
    """``training.device`` was reported as ``run.device`` -- a different knob."""
    from mriforge.cli.app import _gate_probe_acceleration

    seen = _capture_request(monkeypatch)
    config = argparse.Namespace(training=argparse.Namespace(device="cpu"))
    _gate_probe_acceleration(argparse.Namespace(probe=True, device=None), config, _Report())
    assert seen == {"requested": "cpu", "source": "training.device"}


def test_force_cpu_reaches_the_documented_degraded_probe_row(monkeypatch):
    """The regression this fix exists for.

    ``accelerated_run_contract.rst`` documents a "CPU, user dictated -> pass
    (degraded)" row for the Tier-2 gate. With the attribute read, an arm that
    declared no device requested ``"cuda"``, which ``FORCE_CPU`` may not relax,
    so the row was unreachable and the gate appended a hard failure instead.
    """
    import torch

    from mriforge.cli.app import _gate_probe_acceleration

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("FORCE_CPU", "true")

    report = _Report()
    decision = _gate_probe_acceleration(
        argparse.Namespace(probe=True, device=None), _settings(), report
    )
    assert decision is not None, "the gate refused a user-dictated CPU probe"
    assert decision.device == "cpu"
    assert decision.cpu_opt_in is True
    assert not [r for r in report.results if not r.passed]


def test_probe_gate_has_no_dead_top_level_device_leg():
    """The retired top-level spelling is not a field, so the leg was dead."""
    import inspect

    from mriforge.cli.app import _gate_probe_acceleration

    src = inspect.getsource(_gate_probe_acceleration)
    assert 'getattr(config, "device", None)' not in src
    assert "_declared_device(config)" in src, (
        "the declared-vs-defaulted distinction has one owner, main._declared_device"
    )
