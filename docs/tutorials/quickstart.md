# Quickstart

Get from `pip install` to a working spectraMR process in under five minutes.

## Install

```bash
pip install spectramr[mri]
```

## Import

```python
import spectramr
print(spectramr.__version__)  # 0.1.0
```

You will see a one-shot `UserWarning` reminding you the framework is **NOT FOR
CLINICAL USE**. It is emitted **once, from the main process only** — spawned
`DataLoader` workers re-import the package but stay silent (the warning is gated
on `multiprocessing.parent_process() is None`), so a training run with `N` data
workers prints the banner once, not `N+1` times. Silence it everywhere in batch
jobs with `export SPECTRAMR_SUPPRESS_CLINICAL_WARNING=1`.

## A first model

```python
import torch
from spectramr.models.registry import MODEL_REGISTRY

entry = MODEL_REGISTRY.get("toeplitz_attention_unet")
model = entry["class"](in_channels=2, out_channels=2)
model.eval()
with torch.no_grad():
    y = model(torch.randn(1, 2, 64, 64))
print(y.shape)  # torch.Size([1, 2, 64, 64])
```

The registry holds 300+ entries; see [Reference: Registries](../reference/registries.md).

## Next steps

- [First reconstruction](first_reconstruction.md) — load a phantom, run a
  reconstruction strategy end-to-end.
- [Add a paradigm](../how_to/add_paradigm.md) — register a new training
  paradigm.
- [Write an experiment YAML](../how_to/write_experiment_yaml.md) — the v6.0
  schema explained.
