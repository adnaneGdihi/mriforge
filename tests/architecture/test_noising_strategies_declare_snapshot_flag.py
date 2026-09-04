"""A strategy that noises the prepared input must say the snapshot is not the model input.

CLAUDE.md non-negotiable 14: ``first_steps/input_prepared`` is captured before
the forward pass. A strategy whose loss hook builds the model input from
``randn_like`` / ``q_sample`` / ``exp_map`` (every diffusion family) feeds the
model something else, and must set ``snapshot_prepared_is_model_input = False``
and name ``snapshot_model_input_tag``. Eight strategies escaped by subclassing
``BaseTrainingStrategy`` directly (cohort review 2026-09-02, T0.8); this scan is
the fitness function, with the escaping shape planted below.

The candidate set is found by AST (a noising call inside a loss hook); the
declaration is checked on the IMPORTED class, so a flag inherited from
``DiffusionTrainingStrategy`` two levels up counts and a name-based allowlist
is not needed.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

_STRATEGIES = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "spectramr"
    / "infrastructure"
    / "training"
    / "strategies"
)
_NOISING_CALLS = {"randn_like", "q_sample", "add_noise", "exp_map"}
_LOSS_HOOKS = {"_compute_losses_impl", "compute_generator_loss", "compute_loss", "train_step"}


def _calls_in(node: ast.AST) -> set[str]:
    names = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            fn = sub.func
            if isinstance(fn, ast.Attribute):
                names.add(fn.attr)
            elif isinstance(fn, ast.Name):
                names.add(fn.id)
    return names


def noising_classes(source: str) -> list[str]:
    """Strategy classes whose loss hook builds the model input from a noising call."""
    tree = ast.parse(source)
    found = []
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        bases = {b.id if isinstance(b, ast.Name) else getattr(b, "attr", "") for b in cls.bases}
        if not any(b.endswith("Strategy") for b in bases):
            continue
        hooks = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in _LOSS_HOOKS]
        if any(_calls_in(h) & _NOISING_CALLS for h in hooks):
            found.append(cls.name)
    return found


def _declares_flag_false(cls: ast.ClassDef) -> bool:
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign):
            targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
            value = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            targets, value = [stmt.target.id], stmt.value
        else:
            continue
        if "snapshot_prepared_is_model_input" in targets:
            return isinstance(value, ast.Constant) and value.value is False
    return False


def violations(source: str) -> list[str]:
    """Source-level check used by the planted cases: noising classes without the flag."""
    tree = ast.parse(source)
    by_name = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    return [n for n in noising_classes(source) if not _declares_flag_false(by_name[n])]


_PLANTED = """
class Escapee(BaseTrainingStrategy):
    def _compute_losses_impl(self, batch, step):
        x = self._prepare(batch)
        x_t = x + torch.randn_like(x)
        return self.generator_model(x_t)
"""

_FIXED = _PLANTED.replace(
    "class Escapee(BaseTrainingStrategy):\n",
    "class Escapee(BaseTrainingStrategy):\n    snapshot_prepared_is_model_input = False\n",
)


def test_planted_escapee_is_caught() -> None:
    assert violations(_PLANTED) == ["Escapee"]


def test_declaring_the_flag_clears_it() -> None:
    assert violations(_FIXED) == []


def test_a_strategy_that_does_not_noise_is_not_a_candidate() -> None:
    src = _PLANTED.replace("x + torch.randn_like(x)", "x")
    assert noising_classes(src) == []


@pytest.mark.parametrize("path", sorted(_STRATEGIES.glob("*.py")), ids=lambda p: p.name)
def test_every_noising_strategy_declares_the_flag(path: Path) -> None:
    """Resolved at runtime: the class attribute, inherited or not, must be False."""
    names = noising_classes(path.read_text())
    if not names:
        pytest.skip("no noising loss hook in this module")
    module = importlib.import_module(f"spectramr.infrastructure.training.strategies.{path.stem}")
    escaped = [
        n
        for n in names
        if getattr(getattr(module, n), "snapshot_prepared_is_model_input", True) is not False
        or not getattr(getattr(module, n), "snapshot_model_input_tag", None)
    ]
    assert not escaped, (
        f"{path.name}: {escaped} build the model input from a noising call in a loss hook "
        "but inherit snapshot_prepared_is_model_input = True or name no "
        "snapshot_model_input_tag; set the flag False and name the tag (non-negotiable 14)."
    )
