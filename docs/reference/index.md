# Reference

Precise descriptions of the machinery. Reference pages are for looking things
up, not for reading end-to-end — if you want the reasoning behind a rule, see
{doc}`../explanation/index`.

| Page | Covers |
|---|---|
| [YAML schema](yaml_schema.md) | The v6.1/v6.0 config schema: every block, every key, and the `extra="forbid"` contract. |
| [Registries](registries.md) | Every `@register_*` surface — models, losses, metrics, strategies, datasets — and what must import them. |
| [Workflow profiles](workflow_profiles.md) | The imaging-regime × task contract and how a profile resolves. |

## Related reference material

The schema and registry pages above are the entry points; these carry the full
detail:

- {doc}`../config_schema_reference` — the complete annotated schema with defaults.
- {doc}`../cli_reference` — every `mriforge` subcommand and flag.
- {doc}`../model_registry_reference` — every registered model.
- {doc}`../metrics_reference` — every registered metric and its direction.
- {doc}`../losses_reference` — every registered objective.
- {doc}`../scripting_api` — driving the framework from Python.

```{toctree}
:maxdepth: 1
:hidden:

yaml_schema
registries
workflow_profiles
```
