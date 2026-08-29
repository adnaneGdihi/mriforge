"""The ``model_output`` snapshot must reach the sites that bypass the wrapper (#1190).

``BaseTrainingStrategy._compute_losses`` is the template method that arms the
generator-output hook, runs ``_compute_losses_impl``, and emits the generic
``model_output`` debug snapshot. 13 call sites invoke ``_compute_losses_impl``
DIRECTLY -- a custom ``train_step`` that needs the losses inside an optimizer
closure (6), or a ``validation_step`` reusing the training loss (7) -- so for
those paradigms the snapshot never fired at all.

Two things are being pinned here, and the second is the one that separates a
real fix from a facade:

1. ``_capture_model_output`` arms and emits outside the wrapper.
2. It hooks the module it is GIVEN. ``cut`` / ``cyclegan`` / ``stargan_v2``
   reuse ``env.generator`` only when it is a compatible type and build their own
   otherwise, so a fix that kept the old single global hook handle would hook
   whichever module armed first -- normally ``env.generator``, which those
   strategies never forward -- and the snapshot would be wired and still never
   fire (pitfall #16).
"""

from __future__ import annotations

import ast
import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import mriforge.infrastructure.training.debug_snapshot as ds_mod
import mriforge.infrastructure.training.utils.domain_inference as di_mod
from mriforge.infrastructure.training.strategies.base import BaseTrainingStrategy


class _Strat(BaseTrainingStrategy):
    def _compute_losses_impl(self, *args, **kwargs):  # pragma: no cover - per test
        return {"g_total_loss": torch.zeros(())}


def _bare(tmp_path) -> _Strat:
    """A strategy instance with only the attributes these seams read.

    Mirrors ``test_base_model_output_snapshot_2026_06._bare``: ``model_type`` is
    read by the dimension-contract guard, and ``get_model_capabilities`` returns
    ``None`` for an unannotated type, so no guard hook is installed and these
    stay hermetic.
    """
    s = object.__new__(_Strat)
    s.env = SimpleNamespace(run_output_dir=str(tmp_path), generator=None)
    s.config = SimpleNamespace(
        logging=SimpleNamespace(
            snapshots=SimpleNamespace(
                enabled=True,
                max_calls=8,
                save_images=False,
                save_json=True,
                interval_steps=0,
            )
        ),
        training=SimpleNamespace(output_dir=str(tmp_path)),
        model=SimpleNamespace(model_type="configurable_unet"),
    )
    s.amp_helper = SimpleNamespace(get_autocast_context=contextlib.nullcontext)
    return s


def _patch_save(monkeypatch) -> dict:
    """Capture what would be written, so the budget/disk path stays out of scope."""
    captured: dict = {}
    monkeypatch.setattr(ds_mod, "save_debug_snapshot", lambda **kw: captured.update(kw))
    monkeypatch.setattr(di_mod, "needs_ifft_for_visualization", lambda _cfg: (False, False))
    return captured


def _marked(value: float) -> torch.nn.Module:
    """A module whose output identifies WHICH module produced it."""

    class _Marked(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.full_like(x, value)

    return _Marked()


# ── the mechanism ──────────────────────────────────────────────────────────


class TestCaptureModelOutputHooksTheModuleItIsGiven:
    def test_a_strategy_owned_generator_is_captured(self, tmp_path, monkeypatch):
        captured = _patch_save(monkeypatch)
        s = _bare(tmp_path)
        owned = _marked(7.0)
        s.env = SimpleNamespace(run_output_dir=str(tmp_path), generator=_marked(1.0))

        x = torch.zeros(1, 1, 4, 4)
        with s._capture_model_output(module=owned, input_batch=x, target_batch=x, step=3):
            owned(x)

        assert captured["tag"] == "model_output"
        assert float(captured["tensors"]["model_output"].flatten()[0]) == 7.0

    def test_arming_env_generator_first_does_not_block_the_owned_one(self, tmp_path, monkeypatch):
        """The facade case, and the reason the handle guard is keyed per module.

        ``_compute_losses`` arms ``env.generator``. If a strategy then arms its
        OWN generator and a single global handle short-circuits the second
        registration, the owned module carries no hook: nothing is ever stashed
        and ``_snapshot_model_output`` returns early. The snapshot would be
        wired and permanently silent -- which is what #1190 is about.
        """
        captured = _patch_save(monkeypatch)
        s = _bare(tmp_path)
        env_gen = _marked(1.0)
        owned = _marked(7.0)
        s.env = SimpleNamespace(run_output_dir=str(tmp_path), generator=env_gen)

        # Whatever the base path did first must not consume the one registration.
        s._ensure_generator_output_capture()
        assert len(env_gen._forward_hooks) == 1

        x = torch.zeros(1, 1, 4, 4)
        with s._capture_model_output(module=owned, input_batch=x, target_batch=x, step=0):
            owned(x)

        assert len(owned._forward_hooks) == 1
        assert captured, "the owned generator's forward must have been captured"
        assert float(captured["tensors"]["model_output"].flatten()[0]) == 7.0

    def test_registration_is_idempotent_per_module(self, tmp_path):
        s = _bare(tmp_path)
        gen = _marked(1.0)
        s._ensure_generator_output_capture(module=gen)
        s._ensure_generator_output_capture(module=gen)
        s._ensure_generator_output_capture(module=gen)
        assert len(gen._forward_hooks) == 1

    def test_no_raise_when_there_is_nothing_to_hook(self, tmp_path, monkeypatch):
        """A mock-fed strategy has no hookable module; silence is the contract.

        This is deliberately NOT a raise: no declared carve-out is being
        violated, and converting it would redden the mock-fed paradigm census
        (``tests/smoke/test_paradigm_step.py``) for strategies that are not
        actually defective.
        """
        captured = _patch_save(monkeypatch)
        s = _bare(tmp_path)  # env.generator is None
        x = torch.zeros(1, 1, 4, 4)
        with s._capture_model_output(module=None, input_batch=x, target_batch=x):
            pass
        assert captured == {}


class TestTheTrainingDimensionContractStaysOnTheTrainingPath:
    """The shared manager serves 6 train sites AND 7 validation ones.

    ``_ensure_input_contract_guard`` is the TRAINING dimension contract and can
    raise under ``MRIFORGE_DIMENSION_CONTRACT=enforce``. Installing it from the
    manager would extend it to every validation path a snapshot fix happens to
    touch -- a runtime behaviour change nobody asked for, and the thing that
    reddened three val-path tests when this change first tried it.
    """

    def test_the_manager_does_not_install_it(self, tmp_path, monkeypatch):
        _patch_save(monkeypatch)
        s = _bare(tmp_path)
        called: list[int] = []
        monkeypatch.setattr(type(s), "_ensure_input_contract_guard", lambda self: called.append(1))
        gen = _marked(7.0)
        x = torch.zeros(1, 1, 4, 4)
        with s._capture_model_output(module=gen, input_batch=x, target_batch=x):
            gen(x)
        assert called == []

    def test_the_training_wrapper_still_does(self, tmp_path, monkeypatch):
        """The other half: it must not simply have been dropped."""
        _patch_save(monkeypatch)
        s = _bare(tmp_path)
        called: list[int] = []
        monkeypatch.setattr(type(s), "_ensure_input_contract_guard", lambda self: called.append(1))
        s._compute_losses_impl = lambda **kw: {  # type: ignore[method-assign]
            "g_total_loss": torch.zeros(())
        }
        s._compute_losses(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4), epoch=0)
        assert called == [1]


class TestCaptureModelOutputArmingWindow:
    def test_a_forward_after_the_block_cannot_overwrite_the_stash(self, tmp_path, monkeypatch):
        """Why the arming is tight rather than step-wide.

        ``cut`` / ``cyclegan`` / ``stargan_v2`` run the GENERATOR inside their
        discriminator closure to make fakes. A flag left armed across the whole
        step would let that no-grad pass overwrite the stash, and the emitted
        snapshot would show D's fake instead of G's output.
        """
        captured = _patch_save(monkeypatch)
        s = _bare(tmp_path)
        gen = _marked(7.0)
        x = torch.zeros(1, 1, 4, 4)

        with s._capture_model_output(module=gen, input_batch=x, target_batch=x):
            gen(x)
        emitted = float(captured["tensors"]["model_output"].flatten()[0])

        captured.clear()
        with torch.no_grad():  # the D-closure's fake, outside the window
            gen(x)
        s._snapshot_model_output(input_batch=x, target_batch=x, step=1)

        assert emitted == 7.0
        assert captured == {}, "a forward outside the window must not be stashed"

    def test_an_exception_in_the_body_skips_the_emit(self, tmp_path, monkeypatch):
        """Emission sits AFTER the try/finally, matching ``_compute_losses``:
        a forward that did not complete must not produce a snapshot."""
        captured = _patch_save(monkeypatch)
        s = _bare(tmp_path)
        gen = _marked(7.0)
        x = torch.zeros(1, 1, 4, 4)

        with (
            pytest.raises(RuntimeError, match="loss blew up"),
            s._capture_model_output(module=gen, input_batch=x, target_batch=x),
        ):
            gen(x)
            raise RuntimeError("loss blew up")

        assert captured == {}
        assert s._capture_gen_output is False  # still disarmed by the finally

    def test_the_declared_source_still_applies_at_emit_time(self, tmp_path, monkeypatch):
        """The 7 val-path sites nest ``snapshot_source('val')`` OUTSIDE this
        manager so the phase is still ``val`` when the emit runs on exit."""
        _patch_save(monkeypatch)
        s = _bare(tmp_path)
        gen = _marked(7.0)
        x = torch.zeros(1, 1, 4, 4)
        seen: list[str] = []

        original = s._snapshot_model_output

        def _spy(**kw):
            seen.append(s._snapshot_phase)
            return original(**kw)

        s._snapshot_model_output = _spy  # type: ignore[method-assign]

        # Same single-``with`` shape the 7 val-path sites use. Parenthesized
        # managers enter left-to-right and exit right-to-left, so ``val`` is
        # still the phase when the capture's exit runs the emit -- which is
        # exactly the ordering under test.
        with (
            s.snapshot_source("val"),
            s._capture_model_output(module=gen, input_batch=x, target_batch=x),
        ):
            gen(x)

        assert seen == ["val"]
        assert s._snapshot_phase == "train"  # restored, not leaked


# ── the wiring: no bypass site may be left unarmed ─────────────────────────


#: ``test_time_adaptation``'s inner loop runs ``num_adaptation_steps - 1``
#: intermediate adaptation calls before the final one. Only the FINAL call
#: represents the adapted model, so only it is wrapped -- capturing the loop
#: would spend the snapshot budget on states the arm is not evaluated on.
_DELIBERATELY_UNARMED = {("test_time_adaptation_strategy.py", "ttt_closure")}


def _direct_impl_calls() -> list[tuple[str, str, bool]]:
    """Every ``self._compute_losses_impl(...)`` in ``strategies/``.

    Returns ``(filename, enclosing function, armed)``. ``super()._compute_
    losses_impl(...)`` is excluded by construction: its callee is a ``Call`` to
    ``super``, not ``Name('self')``. Those are the intra-``_impl`` delegations
    (a subclass extending its parent's losses), which run INSIDE the wrapper and
    were never bypasses.
    """
    import mriforge.infrastructure.training.strategies as pkg

    found: list[tuple[str, str, bool]] = []
    for path in sorted(Path(pkg.__file__).parent.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # Walk with ancestry so an enclosing ``with`` can be identified.
        stack: list[tuple[ast.AST, list[ast.AST]]] = [(tree, [])]
        while stack:
            node, ancestors = stack.pop()
            for child in ast.iter_child_nodes(node):
                stack.append((child, [*ancestors, node]))
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (
                isinstance(fn, ast.Attribute)
                and fn.attr == "_compute_losses_impl"
                and isinstance(fn.value, ast.Name)
                and fn.value.id == "self"
            ):
                continue
            armed = any(
                "_capture_model_output" in ast.dump(item.context_expr)
                for a in ancestors
                if isinstance(a, (ast.With, ast.AsyncWith))
                for item in a.items
            )
            enclosing = next(
                (
                    a.name
                    for a in reversed(ancestors)
                    if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef))
                ),
                "<module>",
            )
            found.append((path.name, enclosing, armed))
    return found


def test_the_census_still_finds_the_bypass_sites():
    """Guards the scanner itself: a rename that silently matched nothing would
    make the coverage assertion below pass vacuously."""
    calls = _direct_impl_calls()
    assert len(calls) >= 14, f"expected the wrapper + 13 bypass sites, got {len(calls)}"
    assert any(f == "base.py" for f, _, _ in calls)


def test_every_direct_impl_call_arms_the_model_output_snapshot():
    unarmed = {
        (f, fn) for f, fn, armed in _direct_impl_calls() if not armed
    } - _DELIBERATELY_UNARMED
    assert not unarmed, (
        "these call _compute_losses_impl directly without entering "
        f"_capture_model_output, so no model_output snapshot is emitted: {sorted(unarmed)}"
    )
