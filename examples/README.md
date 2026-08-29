# Examples

Self-contained, CPU-friendly demos. Each finishes in under 30 seconds.

| File | Demonstrates |
|---|---|
| [quickstart_reconstruction.py](quickstart_reconstruction.py) | Pull a registered U-Net from `MODEL_REGISTRY`, run one forward pass on a synthetic phantom. |
| [quickstart_diffusion.py](quickstart_diffusion.py) | Generate one α-stable Lévy noise step from `mriforge.models.diffusion` and confirm the heavy-tailed kurtosis signature. |
| [quickstart_physics.py](quickstart_physics.py) | Verify `ifft2c(fft2c(x)) == x` and a Cartesian-mask adjoint identity via the physics SSOT. |

Run any of them with:

```bash
python examples/quickstart_reconstruction.py
python examples/quickstart_diffusion.py
python examples/quickstart_physics.py
```

The first import triggers the clinical-use warning. Set
`MRIFORGE_SUPPRESS_CLINICAL_WARNING=1` in batch jobs.
