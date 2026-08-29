"""The ``run:`` block — facts about *this execution*, not about the method.

Four scalars sat bare on :class:`TrainingSettings`, which is where a reader looks
first and learns least: ``seed``, ``device``, ``model_domain`` and
``deep_supervision_weight``. Only the first two describe the run; the other two
were a duplicate and a loss weight that had drifted up to the root. This block is
the home for the first two, and the other two moved to where they are read.

.. code-block:: yaml

    config_version: '1.0'     # stays at the ROOT — see below

    run:
      seed: 42
      device: cuda

``seed`` had **two** spellings before this block existed — root ``seed:`` and
``training.seed`` — and ``training.seed`` won at runtime
(``pipelines/train.py``). 529 arms used the nested one, 96 the root one, 95 set
both. Both now rename into ``run.seed``, because a canonical home that loses to
an older spelling is not canonical; it is a third spelling.

Why ``config_version`` stays at the root
----------------------------------------

It is declared here and populated from the root key, but the **file** keeps it at
the top level, and that is deliberate rather than a half-migration.

``config_version`` is a *pre-schema discriminator*: it decides whether a file is
loadable at all, and every consumer reads it from the raw document before any
schema exists — ``TrainingSettings.from_yaml`` reads it off the raw dict,
``check_experiment_configs_load`` does ``doc.get("config_version")``, and
``check_no_legacy_config_keys`` matches it with a **line regex before parsing**.
Nesting it under ``run:`` would mean you must know the block layout to learn the
version that tells you the block layout.

What *was* wrong is that it was validated and then ``del``'d, so it never existed
on the object and survived only into provenance. It is now a real field, which is
what makes the version an inspectable fact rather than a parse-time side effect.
The root key remains the only place an author writes it — declaring
``run.config_version`` explicitly raises, so this does not mint the
two-spellings problem the rename effort exists to remove.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from mriforge.config.schemas.renames import reject_renamed_keys

#: Raised by the loader when a document authors ``run.config_version`` itself.
#: Kept here so the message lives next to the field it is about, while the check
#: runs in the loader — where the *authoring* happens. Guarding inside this
#: schema is not possible without a sentinel, because the loader's own
#: population arrives as the same dict key a user would write.
CONFIG_VERSION_NOT_AUTHORED_HERE = (
    "`run.config_version` is populated from the root `config_version:` key, not "
    "authored. Declare the version at the top level of the file:\n"
    "    config_version: '1.0'\n"
    "It is a pre-schema discriminator — every consumer reads it from the raw "
    "document before any schema exists — so it cannot live inside a block whose "
    "layout the version itself determines. Declaring it in both places would "
    "recreate, inside the block meant to fix it, the two-spellings problem this "
    "effort exists to remove."
)


class RunConfigSchema(BaseModel):
    """Execution-level facts: seed, device, and the schema version that built it."""

    model_config = {
        "extra": "forbid",
        "frozen": True,
    }

    _reject_renamed = model_validator(mode="before")(classmethod(reject_renamed_keys("run")))

    seed: int = Field(
        default=42,
        description=(
            "Random seed for reproducibility. The ONE seed — both the old root "
            "`seed:` and `training.seed` rename into this."
        ),
    )
    device: str = Field(
        default="cuda",
        description=(
            "Device request (cuda / cpu / auto). Resolved ONLY through "
            "`mriforge.core.compute_device` (non-negotiable 9b) — never compared "
            "against `torch.cuda.is_available()` at the call site. The default is "
            "`cuda`, not `auto`, carried verbatim from the root field this "
            "replaces: every arm reaches the resolver as an explicit `cuda` "
            "request with `source='config.device'`, and changing that default "
            "would change device resolution and provenance corpus-wide inside a "
            "commit whose diff reads as a move."
        ),
    )
    config_version: str | None = Field(
        default=None,
        description=(
            "Schema version this run was built from. POPULATED from the root "
            "`config_version:` key, never authored here — declaring it raises. "
            "See the module docstring for why the file-level key stays at the "
            "root while the field lives in this block."
        ),
    )


__all__ = ["CONFIG_VERSION_NOT_AUTHORED_HERE", "RunConfigSchema"]
