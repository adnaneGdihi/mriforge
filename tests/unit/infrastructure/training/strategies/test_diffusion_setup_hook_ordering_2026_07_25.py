"""``_setup_strategy_specific_components`` must survive the base's early call.

``BaseTrainingStrategy.__init__`` invokes the subclass hook
(``strategies/base.py``) *before* ``DiffusionTrainingStrategy.__init__`` reaches
``initialize_diffusion_parameters``. The hook then runs
``_bind_generator_reverse_schedule``, which reads ``self.num_timesteps`` — so on
a latent-diffusion arm the constructor died with::

    Pipeline failed: 'DiffusionTrainingStrategy' object has no attribute 'num_timesteps'

which is how ``stage2_ldm_ulf_to_hf`` failed on 2026-07-25 (SLURM 7796517 task 5),
immediately after logging "Input Channels: 1".

The base's comment justifies the early call with "all setup methods are
idempotent". Idempotence is the wrong invariant: the hazard is not running
*twice*, it is running *too early* — before the subclass's own ``__init__`` body
has established the attributes the hook depends on.

Only latent-diffusion arms were affected because
``_bind_generator_reverse_schedule`` early-returns otherwise; that binder is
itself the fix for the stage-2 LDM schedule mismatch (train_psnr≈32 /
val_psnr≈6), so one fix walked into the next.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from spectramr.infrastructure.training.strategies.diffusion import (
    DiffusionTrainingStrategy,
)


def test_setup_hook_is_a_noop_before_diffusion_parameters_exist():
    """The base's premature hook call must not raise.

    Exercised on a bare instance — no ``__init__`` has run, so nothing the hook
    depends on exists. That is exactly the state the base constructor calls it in.
    """
    strategy = object.__new__(DiffusionTrainingStrategy)
    assert not hasattr(strategy, "num_timesteps"), "precondition: uninitialised"

    # Must return quietly rather than AttributeError on self.num_timesteps.
    strategy._setup_strategy_specific_components()


def test_setup_hook_guard_keys_off_diffusion_initialised_flag():
    """The guard must key off the mixin's own initialisation flag.

    ``initialize_diffusion_parameters`` sets ``_diffusion_initialized`` as the
    last thing it does, so the flag is the honest "my invariants hold" signal —
    a ``hasattr(self, 'num_timesteps')`` check would also pass on a
    half-initialised object.
    """
    src = inspect.getsource(DiffusionTrainingStrategy._setup_strategy_specific_components)
    assert "_diffusion_initialized" in src, (
        "the early-return guard must test the mixin's initialisation flag"
    )


def test_init_initialises_diffusion_before_calling_the_setup_hook():
    """Source-order guard: the explicit hook call comes AFTER the schedule init.

    If someone reorders ``__init__`` so ``_setup_strategy_specific_components``
    precedes ``initialize_diffusion_parameters``, the guard above turns the real
    call into a silent no-op and the reverse schedule is never bound — which is
    the val_psnr≈6 collapse the binder exists to prevent. Pin the order.
    """
    tree = ast.parse(Path(inspect.getfile(DiffusionTrainingStrategy)).read_text())
    cls = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "DiffusionTrainingStrategy"
    )
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    calls = [
        n.func.attr
        for n in ast.walk(init)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    ]
    assert "initialize_diffusion_parameters" in calls
    assert "_setup_strategy_specific_components" in calls
    assert calls.index("initialize_diffusion_parameters") < calls.index(
        "_setup_strategy_specific_components"
    ), "the schedule must be initialised before the setup hook runs for real"


def test_no_silent_1000_timestep_fallback():
    """``num_timesteps`` must not degrade to a hardcoded 1000 (pitfall #9).

    The old ``self.num_timesteps if hasattr(...) else 1000`` was a band-aid for
    the same premature call. With the guard in place the attribute is always
    present, and a silent fallback would quietly train k-space components on a
    1000-step schedule while the YAML declared something else.
    """
    src = inspect.getsource(DiffusionTrainingStrategy._setup_strategy_specific_components)
    assert 'hasattr(self, "num_timesteps")' not in src, (
        "silent fallback resurrected — the guard makes it dead code (pitfall #9)"
    )


def test_no_subclass_bypasses_the_explicit_setup_call():
    """Every subclass must reach ``DiffusionTrainingStrategy.__init__``.

    The guard above makes the base's early hook call a no-op, so the *explicit*
    call at the end of ``DiffusionTrainingStrategy.__init__`` is now the only one
    that does real work. A subclass defining ``__init__`` without
    ``super().__init__()`` would therefore silently get no setup at all — the
    exact failure mode this change exists to remove. Pin the invariant.
    """
    root = Path(inspect.getfile(DiffusionTrainingStrategy)).parent
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        for cls in (
            n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.ClassDef)
        ):
            bases = [getattr(b, "id", getattr(b, "attr", "")) for b in cls.bases]
            if "DiffusionTrainingStrategy" not in bases:
                continue
            init = next(
                (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
                None,
            )
            if init is None:
                continue  # inherits __init__ verbatim — fine
            if not any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "__init__"
                and isinstance(n.func.value, ast.Call)
                and getattr(n.func.value.func, "id", "") == "super"
                for n in ast.walk(init)
            ):
                offenders.append(f"{path.name}::{cls.name}")
    assert not offenders, (
        "these subclasses never reach the explicit setup call, so the guard "
        f"silently disables their setup: {offenders}"
    )
