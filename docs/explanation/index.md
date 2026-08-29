# Explanation

Design rationale and the invariants the framework refuses to break. These pages
answer *why*, not *how* — read them when a rule seems arbitrary, or before a
refactor that would cut across layers.

| Page | What it explains |
|---|---|
| [The audit ladder](audit_ladder.md) | Tier 0/1/2 pre-flight: what each tier can and cannot catch. |
| [Workflows](workflows.md) | The imaging-regime × task contract and how workflow profiles resolve. |

## The load-bearing ideas

Three ideas explain most of the framework's shape:

**Config is the single source of truth.** A frozen Pydantic v2 `TrainingSettings`
is loaded once in `main.py` and passed down. Nothing re-parses YAML, and nothing
mutates the object. Every layer sees the same values, so a run is reproducible
from its config alone.

**Physics is a library, not a habit.** Centering and `norm="ortho"` are
correctness-critical for MRI k-space, so the FFT lives in exactly one place. A
raw `torch.fft.fft2` on complex k-space is a bug even when it runs.

**Green means "did not crash", not "method worked".** A config can be entirely
schema-valid and audit-clean while its headline mechanism is never read — the
model swallows it in `**kwargs`, or the loss term is gated on a batch key the
dataloader never emits. The audit ladder narrows this gap but does not close it;
{doc}`audit_ladder` is explicit about which tier catches what.

For the deeper reference behind these, see {doc}`../config_schema_reference`
and {doc}`../audit_ladder_user_guide`.

```{toctree}
:maxdepth: 1
:hidden:

audit_ladder
workflows
```
