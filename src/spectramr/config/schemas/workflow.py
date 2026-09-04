"""Workflow configuration schema — the ``workflow:`` root block.

An arm declares *what its signal is* and *what it is doing to it*:

.. code-block:: yaml

    workflow:
      regime: mri_structural    # the physical REGIME (closed enum)
      task: reconstruction      # what you're DOING to it (closed enum)
      signal_domain: kspace     # which of the regime's domains this arm CONSUMES
      spatial_rank: 2           # 2-D slices or 3-D volumes

``signal_domain`` and ``spatial_rank`` narrow the regime. A
:class:`~spectramr.domain.workflows.profiles.WorkflowProfile` gives a *set* —
``mri_structural`` yields ``image``/``kspace``/``complex_image`` at rank 2 or 3
— so the profile alone cannot say which one an arm is actually in. Both are
optional: absent leaves the value inferred from the data (advisory), wrong is a
hard audit error. That is the polarity CLAUDE.md mandates, and it is why an
uncertain author should omit rather than guess.

**``signal_domain`` is what the arm CONSUMES**, matching
``ModelCapabilities.input_domain``, never what it emits. The distinction is not
pedantic: every parameter-mapping arm emits something other than its regime's
signal — ``MRSQuantificationStrategy`` consumes a ``spectrum`` and emits
resonance maps (domain ``image``), perfusion consumes a DCE series and emits
``(Ktrans, ve, vp)``. A field meaning "the domain this arm works in" is
ambiguous between the two, and reading it as *emits* would reject the very arms
those regimes exist for — the false positive
``check_workflow_signal_domain``'s docstring was written to prevent.

The regime field was called ``name`` until 2026-07-31. ``name`` is the most
overloaded key in the corpus — every loss-list entry, every ``metadata:`` block
has one — so it read as a label for the block rather than as the load-bearing
physical claim it is, and it made the block impossible to grep for. ``regime``
says what the value means.

``regime`` and ``task`` are closed enums, so a typo fails at Tier-0 (Pydantic
``ValidationError``) rather than silently mislabelling the arm. The frozen
:class:`~spectramr.domain.workflows.profiles.WorkflowProfile` facts backing each
regime — spatial ranks, required axes, forward operator, maturity — are checked
at audit time by ``config_health_checker.check_workflow_declared``.

The field is *optional* on :class:`TrainingSettings` today (nothing breaks if it
is absent), but the audit **errors** when it is missing — the "optional now,
required later" migration posture.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from spectramr.config.schemas.enums import Regime, SignalDomain, Task
from spectramr.config.schemas.renames import reject_renamed_keys


class WorkflowConfigSchema(BaseModel):
    """Declared imaging regime x task for an experiment arm."""

    model_config = {
        "protected_namespaces": (),
        "extra": "forbid",
        "frozen": True,
    }

    # `extra="forbid"` would already reject a leftover `name:`, but with the
    # stock "Extra inputs are not permitted" — which does not say that the value
    # is still wanted under a new key. The shim names the replacement.
    _reject_renamed = model_validator(mode="before")(classmethod(reject_renamed_keys("workflow")))

    regime: Regime = Field(
        description="Physical imaging regime — what the signal IS (closed enum).",
    )
    task: Task | None = Field(
        default=None,
        description="What the arm is doing to the signal (closed enum). "
        "Optional; when set the audit checks it against the regime's supported tasks.",
    )
    signal_domain: SignalDomain | None = Field(
        default=None,
        description=(
            "Which of the regime's signal domains this arm CONSUMES (closed "
            "enum), matching ModelCapabilities.input_domain — not what it "
            "emits. Optional; when set the audit checks it against the "
            "regime's signal_domains and against the model's input_domain."
        ),
    )
    spatial_rank: int | None = Field(
        default=None,
        ge=1,
        le=4,
        description=(
            "Spatial rank the arm operates at (2 = slices, 3 = volumes). "
            "Optional; when set the audit checks it against the regime's "
            "spatial_ranks and against the rank the dataset_type provides."
        ),
    )

    # The cross-checks against WORKFLOW_PROFILES live in the audit, not here.
    # `config/` is the innermost layer: importing `domain.workflows.profiles`
    # would be a leftward import (NN#5) and a cycle — `profiles.py` already
    # imports Regime/Task from this package.


__all__ = ["WorkflowConfigSchema"]
