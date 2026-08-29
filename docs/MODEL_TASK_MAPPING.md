# Model-Task Fitness Matrix

Fitness scores (1-10) indicating how well each model architecture is suited for each task.

- **10** = Purpose-built for this task
- **7-9** = Excellent fit, commonly used
- **4-6** = Can work but not optimal
- **1-3** = Poor fit, not recommended

---

## Legend

| Score | Meaning |
|-------|---------|
| 🟢 10 | Purpose-built, state-of-the-art |
| 🟢 9 | Excellent fit |
| 🟢 8 | Very good fit |
| 🟡 7 | Good fit |
| 🟡 6 | Reasonable fit |
| 🟡 5 | Moderate fit |
| 🟠 4 | Marginal fit |
| 🟠 3 | Poor fit |
| 🔴 2 | Very poor fit |
| 🔴 1 | Not designed for this |

---

## Reconstruction Models

| Model | Recon | SR | Field Trans | K-Space | 3D Vol | Denoise | Motion | Gen |
|-------|:-----:|:--:|:-----------:|:-------:|:------:|:-------:|:------:|:---:|
| `standard_unet` | 7 | 6 | 5 | 3 | 3 | 6 | 3 | 3 |
| `enhanced_unet` | 8 | 7 | 6 | 3 | 3 | 7 | 3 | 3 |
| `unrolled_reconstruction` | **10** | 4 | 3 | 8 | 4 | 5 | 4 | 2 |
| `varnet` | **10** | 4 | 3 | 9 | 4 | 5 | 4 | 2 |
| `physics_driven_network` | **10** | 4 | 3 | 9 | 4 | 4 | 4 | 2 |
| `reconformer` | 9 | 5 | 4 | 7 | 4 | 5 | 4 | 2 |

---

## Diffusion Models

| Model | Recon | SR | Field Trans | K-Space | 3D Vol | Denoise | Motion | Gen |
|-------|:-----:|:--:|:-----------:|:-------:|:------:|:-------:|:------:|:---:|
| `kspace_cold_diffusion` | 7 | 3 | 3 | **10** | 3 | 4 | 3 | 4 |
| `latent_diffusion` | 6 | 9 | 8 | 4 | 7 | 7 | 5 | **10** |
| `score_based_diffusion` | 7 | 8 | 7 | 6 | 6 | 8 | 5 | **10** |
| `conditional_diffusion` | 8 | 8 | 9 | 5 | 6 | 7 | 5 | 9 |
| `diffusion_sr` | 5 | **10** | 6 | 3 | 4 | 6 | 3 | 7 |
| `rician_diffusion` | 6 | 5 | 4 | 4 | 4 | **10** | 4 | 5 |
| `chi_square_diffusion` | 6 | 5 | 4 | 4 | 4 | 9 | 4 | 5 |
| `consistency_model` | 6 | 7 | 6 | 4 | 5 | 6 | 4 | 9 |
| `rectified_flow` | 6 | 7 | 6 | 4 | 5 | 6 | 4 | 9 |

---

## Super-Resolution Models

| Model | Recon | SR | Field Trans | K-Space | 3D Vol | Denoise | Motion | Gen |
|-------|:-----:|:--:|:-----------:|:-------:|:------:|:-------:|:------:|:---:|
| `edsr` | 4 | **10** | 4 | 2 | 3 | 5 | 2 | 2 |
| `rcan` | 4 | **10** | 4 | 2 | 3 | 5 | 2 | 2 |
| `swin_ir` | 5 | **10** | 5 | 2 | 4 | 6 | 3 | 3 |
| `han` | 4 | 9 | 4 | 2 | 3 | 5 | 2 | 2 |
| `nafnet` | 5 | 9 | 5 | 2 | 4 | 9 | 3 | 3 |
| `restormer` | 5 | 9 | 5 | 2 | 4 | **10** | 3 | 3 |
| `imdn` | 4 | 8 | 4 | 2 | 3 | 5 | 2 | 2 |
| `elan` | 4 | 9 | 4 | 2 | 3 | 5 | 2 | 2 |
| `dcsrn` | 4 | 9 | 4 | 2 | 3 | 5 | 2 | 2 |

---

## GAN Models

| Model | Recon | SR | Field Trans | K-Space | 3D Vol | Denoise | Motion | Gen |
|-------|:-----:|:--:|:-----------:|:-------:|:------:|:-------:|:------:|:---:|
| `cycle_gan` | 4 | 6 | **10** | 3 | 3 | 4 | 3 | 7 |
| `vqgan` | 5 | 7 | 6 | 3 | 5 | 5 | 3 | 9 |
| `hyperspectral_gan` | 4 | 5 | 7 | 3 | 3 | 4 | 3 | 8 |

---

## VAE Models

| Model | Recon | SR | Field Trans | K-Space | 3D Vol | Denoise | Motion | Gen |
|-------|:-----:|:--:|:-----------:|:-------:|:------:|:-------:|:------:|:---:|
| `vae` | 5 | 5 | 6 | 3 | 5 | 5 | 3 | 9 |
| `vae_3d` | 4 | 4 | 5 | 3 | 8 | 4 | 4 | 9 |
| `vqvae` | 5 | 5 | 6 | 3 | 5 | 5 | 3 | **10** |
| `sparse_vae` | 6 | 5 | 5 | 4 | 5 | 6 | 3 | 8 |
| `robust_vae` | 6 | 5 | 5 | 3 | 5 | 8 | 4 | 7 |
| `kan_vae` | 6 | 6 | 6 | 3 | 5 | 5 | 3 | 8 |

---

## Transformer Models

| Model | Recon | SR | Field Trans | K-Space | 3D Vol | Denoise | Motion | Gen |
|-------|:-----:|:--:|:-----------:|:-------:|:------:|:-------:|:------:|:---:|
| `swin_transformer` | 6 | 8 | 6 | 3 | 4 | 6 | 3 | 4 |
| `vision_transformer` | 5 | 7 | 5 | 3 | 4 | 5 | 3 | 4 |
| `efficient_transformer` | 5 | 8 | 5 | 3 | 4 | 6 | 3 | 4 |
| `kspace_gpt` | 7 | 3 | 3 | **10** | 3 | 3 | 3 | 5 |
| `unetr` | 8 | 6 | 5 | 5 | 7 | 5 | 4 | 3 |

---

## 3D / Volume Models

| Model | Recon | SR | Field Trans | K-Space | 3D Vol | Denoise | Motion | Gen |
|-------|:-----:|:--:|:-----------:|:-------:|:------:|:-------:|:------:|:---:|
| `nesvor` | 6 | 4 | 4 | 3 | **10** | 4 | **10** | 3 |
| `slice_to_volume` | 5 | 4 | 4 | 3 | **10** | 3 | 8 | 3 |
| `unet_3d` | 7 | 5 | 5 | 4 | 9 | 6 | 5 | 4 |
| `unet_2_5d` | 7 | 5 | 5 | 4 | 8 | 5 | 6 | 3 |
| `vnet` | 7 | 5 | 5 | 4 | 9 | 5 | 5 | 3 |

---

## State-Space Models

| Model | Recon | SR | Field Trans | K-Space | 3D Vol | Denoise | Motion | Gen |
|-------|:-----:|:--:|:-----------:|:-------:|:------:|:-------:|:------:|:---:|
| `mamba` | 7 | 6 | 5 | 6 | 5 | 6 | 5 | 5 |
| `mamba_unet` | 8 | 7 | 6 | 6 | 6 | 7 | 5 | 5 |

---

## Graph Models

| Model | Recon | SR | Field Trans | K-Space | 3D Vol | Denoise | Motion | Gen |
|-------|:-----:|:--:|:-----------:|:-------:|:------:|:-------:|:------:|:---:|
| `graph_unet` | 8 | 5 | 5 | 8 | 5 | 5 | 4 | 4 |
| `gnn_coil_fusion` | 7 | 3 | 3 | 7 | 4 | 3 | 3 | 2 |

---

## KAN Models

| Model | Recon | SR | Field Trans | K-Space | 3D Vol | Denoise | Motion | Gen |
|-------|:-----:|:--:|:-----------:|:-------:|:------:|:-------:|:------:|:---:|
| `kan_unet` | 8 | 7 | 6 | 4 | 4 | 6 | 4 | 4 |
| `fast_kan_unet` | 8 | 7 | 6 | 4 | 4 | 6 | 4 | 4 |
| `moe_kan_generator` | 8 | 7 | 6 | 4 | 4 | 6 | 4 | 5 |
| `vit_kan` | 7 | 7 | 6 | 4 | 4 | 6 | 4 | 4 |

---

## Physics-Informed Models

| Model | Recon | SR | Field Trans | K-Space | 3D Vol | Denoise | Motion | Gen |
|-------|:-----:|:--:|:-----------:|:-------:|:------:|:-------:|:------:|:---:|
| `coil_sensitivity_network` | 8 | 2 | 2 | 7 | 4 | 2 | 2 | 2 |
| `pimn` | 9 | 4 | 3 | 8 | 5 | 4 | 4 | 2 |
| `pin_inr` | 7 | 5 | 4 | 6 | 6 | 4 | 6 | 3 |
| `pinn_nerf` | 6 | 4 | 4 | 5 | 7 | 4 | 6 | 4 |
| `fno` | 8 | 5 | 4 | 8 | 5 | 5 | 4 | 3 |
| `deep_image_prior` | 7 | 6 | 4 | 5 | 5 | 7 | 5 | 4 |
| `bloch_cycle_network` | 8 | 3 | 6 | 6 | 4 | 3 | 4 | 3 |

---

## Multi-Contrast / Translation Models

| Model | Recon | SR | Field Trans | K-Space | 3D Vol | Denoise | Motion | Gen |
|-------|:-----:|:--:|:-----------:|:-------:|:------:|:-------:|:------:|:---:|
| `multi_contrast_fusion` | 6 | 6 | **10** | 4 | 5 | 5 | 4 | 7 |
| `multi_contrast_conditional` | 6 | 6 | **10** | 4 | 5 | 5 | 4 | 7 |
| `disentangled_mri` | 5 | 5 | **10** | 3 | 4 | 4 | 4 | 7 |
| `scanner_invariant_embedding` | 5 | 4 | 9 | 3 | 4 | 4 | 3 | 4 |

---

## Flow-Based Models

| Model | Recon | SR | Field Trans | K-Space | 3D Vol | Denoise | Motion | Gen |
|-------|:-----:|:--:|:-----------:|:-------:|:------:|:-------:|:------:|:---:|
| `normalizing_flow` | 5 | 5 | 6 | 4 | 4 | 5 | 4 | 9 |
| `latent_flow` | 5 | 5 | 6 | 4 | 4 | 5 | 4 | 8 |
| `gflownet` | 5 | 4 | 5 | 4 | 4 | 4 | 4 | 8 |

---

## Specialized Models

| Model | Recon | SR | Field Trans | K-Space | 3D Vol | Denoise | Motion | Gen |
|-------|:-----:|:--:|:-----------:|:-------:|:------:|:-------:|:------:|:---:|
| `digital_twin_simulator` | 4 | 3 | 5 | 4 | 4 | 3 | 3 | 8 |
| `neural_ode` | 7 | 5 | 5 | 5 | 5 | 5 | 6 | 6 |
| `nca` | 5 | 5 | 4 | 3 | 4 | 5 | 4 | 6 |
| `spiking_unet` | 6 | 5 | 4 | 3 | 4 | 5 | 4 | 4 |
| `foundation_model` | 6 | 6 | 6 | 5 | 5 | 5 | 5 | 7 |

---

## Summary: Best Models per Task

| Task | Best Models (Score 10) | Good Alternatives (8-9) |
|------|------------------------|-------------------------|
| **Reconstruction** | `unrolled_reconstruction`, `varnet`, `physics_driven_network` | `reconformer` (9), `pimn` (9), `kan_unet` (8) |
| **Super-Resolution** | `edsr`, `rcan`, `swin_ir`, `diffusion_sr` | `restormer` (9), `nafnet` (9), `han` (9) |
| **Field Translation** | `cycle_gan`, `multi_contrast_*`, `disentangled_mri` | `scanner_invariant_embedding` (9), `conditional_diffusion` (9) |
| **K-Space** | `kspace_cold_diffusion`, `kspace_gpt` | `varnet` (9), `spirit_diffusion` (9), `graph_unet` (8) |
| **3D Volume** | `nesvor`, `slice_to_volume` | `unet_3d` (9), `vnet` (9), `unet_2_5d` (8) |
| **Denoising** | `restormer`, `rician_diffusion` | `nafnet` (9), `chi_square_diffusion` (9), `robust_vae` (8) |
| **Motion Correction** | `nesvor` | `slice_to_volume` (8), `dual_inr_motion` (8) |
| **Generation** | `vqvae`, `latent_diffusion`, `score_based_diffusion` | `consistency_model` (9), `vae` (9), `conditional_diffusion` (9) |

---

## How to Use This Matrix

1. **Choose your task** (column)
2. **Find models with score ≥8** for that task
3. **Consider secondary requirements** (speed, memory, interpretability)
4. **Check training mode compatibility** in MODEL_CAPABILITIES.md
