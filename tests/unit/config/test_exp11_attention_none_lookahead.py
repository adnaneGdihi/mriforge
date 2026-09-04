"""The Lookahead wrapper's wiring, and the shootout control's freedom from it.

History, because this file's original premise is now deliberately false. exp_11
``attention_none`` was the corpus's first Lookahead consumer: adoption of the
in-repo optimizer surface was zero (of 1174 ``optimizer_type`` declarations, 1173
were adamw/adam/sgd and **no** arm enabled the wrapper), because it was reachable
from YAML but not safely usable, in two silent ways:

* a scheduler attached to the wrapper decayed a *copy* of ``param_groups`` while
  the inner optimizer kept the initial LR (fixed by
  ``Lookahead._sync_hyperparams_to_inner``);
* ``load_state_dict`` aliased group 1 onto group 0 and left raw ``int`` indices
  behind wherever a parameter had no gradient.

The block was then **removed from that arm on purpose** (``c6f84aa84``): the arm
is the shootout's CONTROL (``metadata.role: baseline``) in a cohort whose whole
contract is that arms differ only in ``attention_type``. A baseline carrying a
second optimization axis makes all ten results unattributable -- pitfall #17 in
its canonical shape. The arm's YAML now carries a "do not re-add it here" note.

Which leaves two separate subjects, and this file keeps both:

**The wrapper's wiring** is exercised against a SYNTHETIC optimization config,
not an arm. Coupling it to one YAML is what broke it: a legitimate,
well-reasoned config edit silently invalidated six tests, and nothing paired the
two (the source<->test gate pairs a changed *module* with its sibling test, and
that change was a YAML). The wrapper is real capability -- it must keep working
whether or not any arm currently opts in, which is exactly the "unused is
capability to wire, not dead code" reading. The synthetic config mirrors the
hyperparameters the arm used, so the coverage stays representative of a real
consumer.

**The control's freedom from it** is asserted against the corpus, and is the
live invariant: no shootout arm -- including ``attention_none`` -- may declare
the block. That pins the removal so a later change cannot quietly re-add it.

Scope note: ``check_optimizer_specs_resolve.py`` already walks the whole corpus
for resolve-and-build. What it cannot say is whether the *shootout* has stayed
single-axis, which is the question below.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from spectramr.config.schemas.optimization import OptimizationConfigSchema  # noqa: E402
from spectramr.config.settings import TrainingSettings  # noqa: E402
from spectramr.infrastructure.training.optimizer_resolution import (  # noqa: E402
    build_optimizer_from_spec,
    resolve_optimizer_spec,
)
from spectramr.infrastructure.training.optimizers.lookahead import (  # noqa: E402
    Lookahead,
)
from tests.utils.corpus import tracked_yamls  # noqa: E402

ARM = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "inprogress"
    / "kspace_filling"
    / "attention_shootout"
    / "experiment_11_attention_none.yaml"
)

#: The hyperparameters the arm declared while it was the wrapper's consumer.
#: Kept verbatim rather than simplified: ``eps`` is 1e-4 here on purpose (see the
#: arm's own note) and is exactly the kind of value a wrapper is prone to lose.
_ARM_OPTIMIZER = {
    "type": "adamw",
    "learning_rate": 5e-5,
    "weight_decay": 1e-4,
    "betas": [0.9, 0.999],
    "kwargs": {"eps": 1e-4},
    "lookahead": {"enabled": True, "sync_period": 5, "alpha": 0.5},
}


@pytest.fixture()
def built():
    """A Lookahead-wrapped AdamW built the way a run builds one.

    Through ``resolve_optimizer_spec`` / ``build_optimizer_from_spec`` -- the
    production path -- rather than by constructing ``Lookahead`` directly, so the
    resolution step stays covered. Only the *source* of the config is synthetic.
    """
    optimization = OptimizationConfigSchema.model_validate({"optimizer": _ARM_OPTIMIZER})
    spec = resolve_optimizer_spec(optimization)
    model = nn.Linear(4, 2)
    return spec, build_optimizer_from_spec(spec, model.named_parameters()), model


def test_a_declared_block_builds_a_lookahead_wrapped_adamw(built) -> None:
    """The mechanism-fires assertion: declaring the block is not evidence the
    wrapper is what the run steps (pitfall #16)."""
    _, optimizer, _ = built
    assert isinstance(optimizer, Lookahead)
    assert isinstance(optimizer.optimizer, torch.optim.AdamW)
    assert optimizer.la_steps == 5
    assert optimizer.la_alpha == pytest.approx(0.5)


def test_wrapping_does_not_disturb_the_base_hyperparameters(built) -> None:
    """Everything but the wrapper must survive being wrapped."""
    spec, optimizer, _ = built
    assert spec.name == "adamw"
    assert spec.lr == pytest.approx(5e-5)
    assert spec.params["betas"] == (0.9, 0.999)
    assert spec.params["weight_decay"] == pytest.approx(1e-4)
    assert spec.params["eps"] == pytest.approx(1e-4)
    assert optimizer.optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)


def test_lookahead_is_stamped_into_provenance(built) -> None:
    """An unrecorded wrapper makes the run unreproducible from its own
    artifacts (non-negotiable #8)."""
    spec, _, _ = built
    assert spec.as_provenance()["lookahead"] == {"sync_period": 5, "alpha": 0.5}


def test_a_declared_schedule_reaches_the_inner_optimizer(built) -> None:
    """Before the sync fix the wrapper absorbed every scheduler write, and a run
    trained its whole budget at the initial LR while the logs drew the decay."""
    _, optimizer, model = built
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    for _ in range(3):
        optimizer.zero_grad()
        (model(torch.randn(8, 4)) ** 2).mean().backward()
        optimizer.step()
        scheduler.step()
    optimizer.zero_grad()
    (model(torch.randn(8, 4)) ** 2).mean().backward()
    optimizer.step()  # the sync happens inside step()

    inner_lr = optimizer.optimizer.param_groups[0]["lr"]
    assert inner_lr < 5e-5, "the inner optimizer never saw the decay"
    assert inner_lr == pytest.approx(optimizer.param_groups[0]["lr"])


def test_a_resume_keeps_every_parameter_in_its_own_group(built) -> None:
    """Resume used to duplicate a group's tensors, aliasing group 1 onto 0."""
    _, optimizer, model = built
    optimizer.zero_grad()
    (model(torch.randn(8, 4)) ** 2).mean().backward()
    optimizer.step()

    before = [[id(p) for p in g["params"]] for g in optimizer.param_groups]
    optimizer.load_state_dict(optimizer.state_dict())
    after = [[id(p) for p in g["params"]] for g in optimizer.param_groups]

    assert after == before
    flat = [i for g in after for i in g]
    assert len(set(flat)) == len(flat)


def _shootout_lookahead_flags() -> dict[str, bool]:
    """``arm stem -> optimizer.lookahead.enabled`` for every shootout arm.

    Reads the RESOLVED settings, not the YAML text. It used to walk
    ``optimization.lookahead`` with ``yaml.safe_load``, and the 2026-08-02
    canonical-key drain moved the block to ``optimization.optimizer.lookahead``
    -- a pure text move leaving the resolved document byte-identical. Half the
    old test failed loudly on it; the other half went *vacuously green*, because
    a missing path reads as ``False`` for every arm. A raw-YAML predicate tests
    the document, not the run, and fails silently in the direction that matters.
    """
    return {
        sibling.stem: TrainingSettings.from_yaml(
            str(sibling)
        ).optimization.optimizer.lookahead.enabled
        for sibling in tracked_yamls(ARM.parent, "experiment_11_attention_*.yaml", recursive=False)
    }


def test_the_shootout_is_single_axis() -> None:
    """The cohort's contract is that arms differ only in ``attention_type``
    -- with the two documented ``kan_dual_domain`` sub-variants (``_smap``,
    ``_sparse``), which vary a knob inside that block and baseline against it.

    Lookahead is a second axis, and the control carried it until ``c6f84aa84``.
    This pins the removal in the only direction that matters: if a later change
    rolls the block back out -- to the control or to any sibling -- this fails
    and forces the comparability note in the arm's YAML to be revisited rather
    than quietly going stale.
    """
    flags = _shootout_lookahead_flags()

    assert flags, "no shootout arms found; the glob is wrong"
    assert len(flags) > 1, "cohort shrank to one arm; the comparison is vacuous"
    enabled = sorted(stem for stem, on in flags.items() if on)
    assert not enabled, (
        "shootout arm(s) enabled lookahead, adding a second axis to a cohort "
        f"whose contract is attention_type alone: {enabled}. Roll it out to all "
        "of them or none, and update the comparability note."
    )


def test_the_control_arm_does_not_declare_lookahead() -> None:
    """Stated separately from the cohort sweep, because it is the specific thing
    ``c6f84aa84`` decided and the arm's YAML asks in prose not to undo.

    A cohort-wide assertion would still pass if the control were the one arm
    that kept it and every sibling gained it too -- "all ten carry it" satisfies
    single-axis but is a different experiment from the one the results are read
    against.
    """
    flags = _shootout_lookahead_flags()

    assert "experiment_11_attention_none" in flags, "the control arm is missing"
    assert flags["experiment_11_attention_none"] is False, (
        "the shootout CONTROL re-declared lookahead. It is the baseline the "
        "other nine are compared against; a second optimization axis on it "
        "makes every one of those comparisons unattributable (pitfall #17). "
        "See the 'do not re-add it to this arm' note in the arm's YAML."
    )
