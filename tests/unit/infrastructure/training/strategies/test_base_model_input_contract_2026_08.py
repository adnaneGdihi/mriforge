"""The ``snapshot_prepared_is_model_input`` carve-out must point somewhere real.

Non-negotiable 14 lets a strategy that degrades its prepared input *inside* the
step opt out of the ``first_steps/input_prepared`` claim, by setting
``snapshot_prepared_is_model_input = False`` and naming the tag that carries the
real model input in ``snapshot_model_input_tag``. That pair is a **pointer**, and
until this change nothing checked its target existed.

It did not. ``DiffusionTrainingStrategy`` declares ``diffusion_step``, but the
only emitter lived in ``_prepare_diffusion_inputs`` — reachable solely through
``diffusion._compute_losses_impl``, the hook seven subclasses override. Each
inherited the promise and emitted nothing at its target, so their artifacts
stamped ``prepared_equals_model_input: False`` and named a snapshot that was
never written. The reader is then told the visible input is not the model input,
and sent to a tag that does not exist.

The fix puts the check in the ``_compute_losses`` **wrapper**, which an
overriding ``_compute_losses_impl`` cannot bypass — the wrapper-vs-hook boundary
that caused the bug, used the other way round.

Bare-instance pattern, as in ``test_base_model_output_snapshot_2026_06``.
"""

from __future__ import annotations

import contextlib
import json
from types import SimpleNamespace

import pytest
import torch

import mriforge.infrastructure.training.utils.domain_inference as di_mod
from mriforge.infrastructure.training.strategies.base import BaseTrainingStrategy


class _Strat(BaseTrainingStrategy):
    def _compute_losses_impl(self, *args, **kwargs):  # pragma: no cover - per test
        return {"g_total_loss": torch.zeros(())}


def _bare(tmp_path, *, enabled: bool = True) -> _Strat:
    s = object.__new__(_Strat)
    s.env = SimpleNamespace(run_output_dir=str(tmp_path), generator=None)
    s.config = SimpleNamespace(
        # NESTED, as the loader produces since the phase-10b block
        # decomposition. A flat `debug_snapshot_*` fixture raises AttributeError
        # inside `_resolve_config`, which the emitter swallows -- issue #1184.
        logging=SimpleNamespace(
            snapshots=SimpleNamespace(
                enabled=enabled,
                max_calls=8,
                save_images=False,  # no torchvision needed
                save_json=True,
                interval_steps=0,
            )
        ),
        training=SimpleNamespace(output_dir=str(tmp_path)),
        model=SimpleNamespace(model_type="configurable_unet"),
    )
    s._declared_model_input = None
    s._declared_channel_segments = None
    s._model_input_snapshot_done = False
    # Mirrors `__init__`. This fixture builds the instance with
    # `object.__new__`, so anything the constructor sets has to be set here too
    # -- and the width cache is read on the model-input path (#1298).
    s._model_input_width = None
    s._model_input_contract_warned = False
    return s


# ── The default: no carve-out, nothing to enforce ──────────────────────────


def test_a_strategy_that_does_not_degrade_in_step_is_untouched(tmp_path) -> None:
    """`input_prepared` really is the model input for most paradigms."""
    s = _bare(tmp_path)
    assert s.snapshot_prepared_is_model_input is True

    s._snapshot_declared_model_input(step=0)  # must not raise

    assert not (tmp_path / "debug_snapshots").exists()


# ── The carve-out, honoured ────────────────────────────────────────────────


def test_declared_input_is_written_under_the_declared_tag(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(di_mod, "needs_ifft_for_visualization", lambda _cfg: (False, False))
    s = _bare(tmp_path)
    s.snapshot_prepared_is_model_input = False
    s.snapshot_model_input_tag = "my_step"

    x_t = torch.randn(1, 1, 8, 8)
    s._declare_model_input(
        {"model_input": x_t},
        extra={"why": "noised in-step", "model_input_key": "model_input"},
    )
    s._snapshot_declared_model_input(step=0)

    snap = tmp_path / "debug_snapshots" / "my_step_step_000000"
    assert snap.is_dir(), "the declared tag must actually be written"
    assert (snap / "snapshot.json").exists()


def test_a_strategy_emitting_under_its_own_tag_satisfies_the_contract(
    tmp_path, monkeypatch
) -> None:
    """VF's shape: it already writes `vf_twin`, so it declares nothing.

    The flag is set inside `save_debug_snapshot` when the tag matches, so the
    existing emission is wired to the contract rather than duplicated.
    """
    monkeypatch.setattr(di_mod, "needs_ifft_for_visualization", lambda _cfg: (False, False))
    s = _bare(tmp_path)
    s.snapshot_prepared_is_model_input = False
    s.snapshot_model_input_tag = "vf_twin"

    s.save_debug_snapshot(
        {"twin_corrupted_input": torch.randn(1, 1, 8, 8)},
        step=0,
        tag="vf_twin",
        model_input_key="twin_corrupted_input",
    )
    assert s._model_input_snapshot_done is True

    s._snapshot_declared_model_input(step=0)  # must not raise


# ── The carve-out, violated ────────────────────────────────────────────────


def test_a_dangling_pointer_raises(tmp_path) -> None:
    """The bug this whole mechanism exists to close."""
    s = _bare(tmp_path)
    s.snapshot_prepared_is_model_input = False
    s.snapshot_model_input_tag = "diffusion_step"

    with pytest.raises(RuntimeError, match="diffusion_step"):
        s._snapshot_declared_model_input(step=0)


def test_declaring_the_carve_out_without_a_tag_raises(tmp_path) -> None:
    """"Not the model input" alone sends the reader back to the code."""
    s = _bare(tmp_path)
    s.snapshot_prepared_is_model_input = False
    s.snapshot_model_input_tag = None

    with pytest.raises(ValueError, match="snapshot_model_input_tag"):
        s._snapshot_declared_model_input(step=0)


def test_the_tag_check_is_static_and_fires_with_snapshots_disabled(tmp_path) -> None:
    """A misconfigured attribute pair is wrong before any run starts.

    Deliberately a different scope from the missing-declaration check below:
    this one reads only class attributes, so no config can make it correct.
    """
    s = _bare(tmp_path, enabled=False)
    s.snapshot_prepared_is_model_input = False
    s.snapshot_model_input_tag = None

    with pytest.raises(ValueError):
        s._snapshot_declared_model_input(step=0)


def test_disabled_snapshots_suppress_the_missing_declaration_raise(tmp_path) -> None:
    """The contract binds what an ARTIFACT claims.

    With snapshots off there is no `first_steps` record making the claim, so
    there is nothing to contradict — and a diagnostics contract must not become
    a training blocker for a run that turned diagnostics off. This is also the
    principled opt-out for a fixture that is not testing snapshots.
    """
    s = _bare(tmp_path, enabled=False)
    s.snapshot_prepared_is_model_input = False
    s.snapshot_model_input_tag = "diffusion_step"

    s._snapshot_declared_model_input(step=0)  # must not raise


# ── Per-step state ─────────────────────────────────────────────────────────


def test_the_declaration_does_not_survive_into_the_next_step(
    tmp_path, monkeypatch
) -> None:
    """A stale stash must never satisfy a later step, nor pin its tensors."""
    monkeypatch.setattr(di_mod, "needs_ifft_for_visualization", lambda _cfg: (False, False))
    s = _bare(tmp_path)
    s.snapshot_prepared_is_model_input = False
    s.snapshot_model_input_tag = "my_step"

    s._declare_model_input(
        {"model_input": torch.randn(1, 1, 8, 8)},
        extra={"model_input_key": "model_input"},
    )
    s._snapshot_declared_model_input(step=0)
    assert s._declared_model_input is None, "the stash must be cleared after emitting"

    # Reproduce the reset `_compute_losses` performs at the top of every step.
    # Without it the *satisfaction flag* left True by step 0's write would carry
    # over and answer for step 1 -- which is why the wrapper resets both, and
    # why this test resets only the flag: the stash clearing is its subject.
    s._model_input_snapshot_done = False

    with pytest.raises(RuntimeError):
        s._snapshot_declared_model_input(step=1)


def test_budget_exhaustion_does_not_manufacture_a_violation(
    tmp_path, monkeypatch
) -> None:
    """The satisfaction flag is set on ATTEMPT, not on a successful write.

    `save_debug_snapshot` caps writes per `(run_dir, tag)`. Had the flag been
    set only when a write happened, step `max_calls + 1` of a perfectly
    well-behaved run would raise — invisible to a one-step unit test, fatal to a
    real run. `max_calls=1` here makes every step past the first a no-write.
    """
    monkeypatch.setattr(di_mod, "needs_ifft_for_visualization", lambda _cfg: (False, False))
    s = _bare(tmp_path)
    s.config.logging.snapshots.max_calls = 1
    s.snapshot_prepared_is_model_input = False
    s.snapshot_model_input_tag = "vf_twin"

    for step in range(5):
        s._model_input_snapshot_done = False
        s.save_debug_snapshot(
            {"x": torch.randn(1, 1, 8, 8)},
            step=step,
            tag="vf_twin",
            model_input_key="x",
        )
        assert s._model_input_snapshot_done is True, f"flag not set on step {step}"
        s._snapshot_declared_model_input(step=step)  # must not raise on any step

    # Without this the test is VACUOUS: if the budget ever stopped suppressing,
    # every step would write, the flag would be set the "wrong" way round for
    # free, and the assertions above would still pass. Pin that the writer really
    # was refusing to write while the flag kept being set — 1 of 5 steps on disk.
    written = sorted(p.name for p in (tmp_path / "debug_snapshots").glob("vf_twin_step_*"))
    assert written == ["vf_twin_step_000000"], (
        f"budget did not suppress, so this test proves nothing: {written}"
    )


# ── The wrapper wiring (an overriding `_impl` cannot bypass it) ────────────


def test_the_wrapper_enforces_the_contract_end_to_end(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(di_mod, "needs_ifft_for_visualization", lambda _cfg: (False, False))
    gen = torch.nn.Conv2d(1, 1, 3, padding=1)
    s = _bare(tmp_path)
    s.env = SimpleNamespace(run_output_dir=str(tmp_path), generator=gen)
    s.amp_helper = SimpleNamespace(get_autocast_context=contextlib.nullcontext)
    s.snapshot_prepared_is_model_input = False
    s.snapshot_model_input_tag = "my_step"

    def _impl(*, input_batch, target_batch, epoch, **kwargs):
        noised = input_batch + torch.randn_like(input_batch)
        s._declare_model_input(
            {"model_input": noised},
            in_kspace_keys=set(),
            extra={"model_input_key": "model_input"},
        )
        return {"g_total_loss": torch.nn.functional.l1_loss(gen(noised), target_batch)}

    s._compute_losses_impl = _impl  # type: ignore[method-assign]

    x, y = torch.randn(2, 1, 8, 8), torch.randn(2, 1, 8, 8)
    s._compute_losses(x, y, epoch=0, iteration=0)

    assert (tmp_path / "debug_snapshots" / "my_step_step_000000").is_dir()


def test_the_wrapper_raises_when_an_overriding_impl_declares_nothing(
    tmp_path, monkeypatch
) -> None:
    """The seven diffusion subclasses, reproduced in miniature."""
    monkeypatch.setattr(di_mod, "needs_ifft_for_visualization", lambda _cfg: (False, False))
    gen = torch.nn.Conv2d(1, 1, 3, padding=1)
    s = _bare(tmp_path)
    s.env = SimpleNamespace(run_output_dir=str(tmp_path), generator=gen)
    s.amp_helper = SimpleNamespace(get_autocast_context=contextlib.nullcontext)
    s.snapshot_prepared_is_model_input = False
    s.snapshot_model_input_tag = "diffusion_step"

    def _impl(*, input_batch, target_batch, epoch, **kwargs):
        # Overrides the parent hook and never emits the promised tag -- exactly
        # what edm/flow_matching/ambient/... did.
        return {"g_total_loss": torch.nn.functional.l1_loss(gen(input_batch), target_batch)}

    s._compute_losses_impl = _impl  # type: ignore[method-assign]

    x, y = torch.randn(2, 1, 8, 8), torch.randn(2, 1, 8, 8)
    with pytest.raises(RuntimeError, match="_declare_model_input"):
        s._compute_losses(x, y, epoch=0, iteration=0)


# ── The CONTENT half: does the snapshot describe what the model was fed? ───
#
# #1298. Everything above verifies that the model-input snapshot ARRIVED. That
# is satisfied by one naming the wrong tensor, which is what `diffusion.py` did:
# it stamped `model_input_key = "noisy_kspace"` unconditionally while the
# backbone received the 16-channel `cat([noisy_images, smaps])`. The recorded
# snapshot did not contain the model input at all, and every test above passed.


def test_the_model_input_snapshot_must_say_which_tensor_it_is(tmp_path, monkeypatch) -> None:
    """A four-tensor snapshot that names none of them is not a record."""
    monkeypatch.setattr(di_mod, "needs_ifft_for_visualization", lambda _cfg: (False, False))
    s = _bare(tmp_path)
    s.snapshot_prepared_is_model_input = False
    s.snapshot_model_input_tag = "vf_twin"

    with pytest.raises(ValueError, match="without naming which of its tensors"):
        s.save_debug_snapshot(
            {"clean": torch.randn(1, 1, 8, 8), "corrupted": torch.randn(1, 1, 8, 8)},
            step=0,
            tag="vf_twin",
        )


def test_naming_a_tensor_the_snapshot_does_not_carry_raises(tmp_path, monkeypatch) -> None:
    """#1298 in miniature: the label points outside the artifact."""
    monkeypatch.setattr(di_mod, "needs_ifft_for_visualization", lambda _cfg: (False, False))
    s = _bare(tmp_path)
    s.snapshot_prepared_is_model_input = False
    s.snapshot_model_input_tag = "diffusion_step"

    with pytest.raises(ValueError, match="but that snapshot holds"):
        s.save_debug_snapshot(
            {"noisy_kspace": torch.randn(1, 8, 8, 8)},
            step=0,
            tag="diffusion_step",
            model_input_key="model_input",
        )


def test_other_snapshots_are_not_asked_to_name_a_model_input(tmp_path, monkeypatch) -> None:
    """The demand is scoped to the declared tag, not to every emission."""
    monkeypatch.setattr(di_mod, "needs_ifft_for_visualization", lambda _cfg: (False, False))
    s = _bare(tmp_path)
    s.snapshot_prepared_is_model_input = False
    s.snapshot_model_input_tag = "diffusion_step"

    s.save_debug_snapshot({"model_output": torch.randn(1, 1, 8, 8)}, step=0, tag="model_output")

    payload = json.loads(
        (tmp_path / "debug_snapshots" / "model_output_step_000000" / "snapshot.json").read_text()
    )
    assert "model_input_contract" not in payload, (
        "a snapshot that makes no model-input claim must not grow a hollow verdict"
    )


def test_a_channel_mismatch_is_recorded_in_the_artifact(tmp_path, monkeypatch) -> None:
    """The #1298 defect itself: 8 recorded channels, a 16-plane first conv.

    Warn-and-stamp, not raise. The verdict has to survive into the artifact --
    a log line is one clamped level away from invisible, and the attention arms
    that produced this defect run at `level: warning`.
    """
    monkeypatch.setattr(di_mod, "needs_ifft_for_visualization", lambda _cfg: (False, False))
    s = _bare(tmp_path)
    s.env = SimpleNamespace(
        run_output_dir=str(tmp_path), generator=torch.nn.Conv2d(16, 4, 3, padding=1)
    )
    s.snapshot_prepared_is_model_input = False
    s.snapshot_model_input_tag = "diffusion_step"

    s.save_debug_snapshot(
        {"noisy_kspace": torch.randn(1, 8, 8, 8)},
        step=0,
        tag="diffusion_step",
        model_input_key="noisy_kspace",
    )

    payload = json.loads(
        (tmp_path / "debug_snapshots" / "diffusion_step_step_000000" / "snapshot.json").read_text()
    )
    verdict = payload["model_input_contract"]
    assert verdict["status"] == "mismatch"
    assert verdict["declared"]["channels"] == 8
    assert verdict["applied"]["in_channels"] == 16
    # Both halves present: the divergence must be readable as a subtraction, not
    # re-derived by someone who still has the model (non-negotiable 14).
    assert verdict["model_input_key"] == "noisy_kspace"

    text = (tmp_path / "debug_snapshots" / "diffusion_step_step_000000" / "snapshot.txt").read_text()
    assert "MISMATCH" in text


def test_the_matching_case_records_a_match(tmp_path, monkeypatch) -> None:
    """The post-fix shape: the 16-channel concat against a 16-plane conv."""
    monkeypatch.setattr(di_mod, "needs_ifft_for_visualization", lambda _cfg: (False, False))
    s = _bare(tmp_path)
    s.env = SimpleNamespace(
        run_output_dir=str(tmp_path), generator=torch.nn.Conv2d(16, 4, 3, padding=1)
    )
    s.snapshot_prepared_is_model_input = False
    s.snapshot_model_input_tag = "diffusion_step"

    s.save_debug_snapshot(
        {
            "model_input": torch.randn(1, 16, 8, 8),
            "noisy_kspace": torch.randn(1, 8, 8, 8),
        },
        step=0,
        tag="diffusion_step",
        model_input_key="model_input",
    )

    payload = json.loads(
        (tmp_path / "debug_snapshots" / "diffusion_step_step_000000" / "snapshot.json").read_text()
    )
    assert payload["model_input_contract"]["status"] == "match"


def test_an_unreadable_backbone_width_is_reported_not_dropped(tmp_path, monkeypatch) -> None:
    """`unresolved` is a verdict. A check that quietly declines to run reads as
    a passing one, which is the failure class this whole module closes."""
    monkeypatch.setattr(di_mod, "needs_ifft_for_visualization", lambda _cfg: (False, False))
    s = _bare(tmp_path)  # env.generator is None
    s.snapshot_prepared_is_model_input = False
    s.snapshot_model_input_tag = "diffusion_step"

    s.save_debug_snapshot(
        {"model_input": torch.randn(1, 16, 8, 8)},
        step=0,
        tag="diffusion_step",
        model_input_key="model_input",
    )

    payload = json.loads(
        (tmp_path / "debug_snapshots" / "diffusion_step_step_000000" / "snapshot.json").read_text()
    )
    verdict = payload["model_input_contract"]
    assert verdict["status"] == "unresolved"
    assert verdict["declared"]["channels"] == 16, "the half that IS knowable is still recorded"


def test_the_backbone_width_is_resolved_once_not_per_step(tmp_path, monkeypatch) -> None:
    """The wrapper runs every step (the write budget lives downstream), so an
    uncached `modules()` walk would be a per-step loop for a diagnostic."""
    monkeypatch.setattr(di_mod, "needs_ifft_for_visualization", lambda _cfg: (False, False))
    calls: list[int] = []

    class _CountingConv(torch.nn.Conv2d):
        def modules(self):  # counting shim
            calls.append(1)
            return super().modules()

    s = _bare(tmp_path)
    s.env = SimpleNamespace(
        run_output_dir=str(tmp_path), generator=_CountingConv(16, 4, 3, padding=1)
    )
    s.snapshot_prepared_is_model_input = False
    s.snapshot_model_input_tag = "diffusion_step"

    for step in range(4):
        s.save_debug_snapshot(
            {"model_input": torch.randn(1, 16, 8, 8)},
            step=step,
            tag="diffusion_step",
            model_input_key="model_input",
        )

    # `Conv2d` declares `in_channels`, so the walk should not happen at all --
    # and certainly not four times.
    assert len(calls) <= 1, f"modules() walked {len(calls)} times across 4 steps"
