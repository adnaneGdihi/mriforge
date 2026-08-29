from mriforge.models.diffusion import (
    ChiSquareDiffusion,
    ColdDiffusion,
    ColdDiffusionKAN,
    Diffusion,
    DiffusionKAN,
    LaplaceDiffusion,
    RicianDiffusion,
    ScoreBasedDiffusion,
    StableDiffusion,
)

# Diffusion model configurations
GENERATOR_DIFFUSION_CONFIGS = {
    # Standard U-Net generators
    "standard_unet": {
        "generator_type": "standard_unet",
        "diffusion_processes": ["diffusion", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "enhanced_unet": {
        "generator_type": "enhanced_unet",
        "diffusion_processes": ["diffusion", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "enhanced_deep_unet": {
        "generator_type": "enhanced_deep_unet",
        "diffusion_processes": ["diffusion", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    # Transformer-based generators
    "swin_ir": {
        "generator_type": "swin_ir",
        "diffusion_processes": ["diffusion", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "edsr": {
        "generator_type": "edsr",
        "diffusion_processes": ["diffusion", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "rcan": {
        "generator_type": "rcan",
        "diffusion_processes": ["diffusion", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    # Specialized generators
    "nafnet": {
        "generator_type": "nafnet",
        "diffusion_processes": ["diffusion", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "restormer": {
        "generator_type": "restormer",
        "diffusion_processes": ["diffusion", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "uformer": {
        "generator_type": "uformer",
        "diffusion_processes": ["diffusion", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "vit_kan": {
        "generator_type": "vit_kan",
        "diffusion_processes": ["diffusion", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "swin_transformer_kan": {
        "generator_type": "swin_transformer_kan",
        "diffusion_processes": ["diffusion", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "vit": {
        "generator_type": "vit",
        "diffusion_processes": ["diffusion", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "swin": {
        "generator_type": "swin",
        "diffusion_processes": ["diffusion", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "fourier_kan_unet": {
        "generator_type": "fourier_kan_unet",
        "diffusion_processes": ["diffusion", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "wavelet_kan_unet": {
        "generator_type": "wavelet_kan_unet",
        "diffusion_processes": ["diffusion", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "kan_unet": {
        "generator_type": "kan_unet",
        "diffusion_processes": ["diffusion", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "attention_unet": {
        "generator_type": "attention_unet",
        "diffusion_processes": ["diffusion", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "enhanced_attention_unet": {
        "generator_type": "enhanced_attention_unet",
        "diffusion_processes": ["diffusion", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "hybrid_attention_unet": {
        "generator_type": "hybrid_attention_unet",
        "diffusion_processes": ["diffusion", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "full_kan_attention_unet": {
        "generator_type": "full_kan_attention_unet",
        "diffusion_processes": ["diffusion", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "robust_unet": {
        "generator_type": "robust_unet",
        "diffusion_processes": ["diffusion", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "cycle_sr": {
        "generator_type": "cycle_sr",
        "diffusion_processes": ["diffusion", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "dcsrn": {
        "generator_type": "dcsrn",
        "diffusion_processes": ["diffusion", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "elixir_unet": {
        "generator_type": "elixir_unet",
        "diffusion_processes": ["diffusion", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "fsrcnn": {
        "generator_type": "fsrcnn",
        "diffusion_processes": ["diffusion", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "lapsrn": {
        "generator_type": "lapsrn",
        "diffusion_processes": ["diffusion", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "rdn": {
        "generator_type": "rdn",
        "diffusion_processes": ["diffusion", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "vdsr": {
        "generator_type": "vdsr",
        "diffusion_processes": ["diffusion", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "unet": {
        "generator_type": "unet",
        "diffusion_processes": ["diffusion", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "diffusion_sr": {
        "generator_type": "diffusion_sr",
        "diffusion_processes": ["diffusion", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "anisotropic_voxel_generator": {
        "generator_type": "anisotropic_voxel_generator",
        "diffusion_processes": ["diffusion", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "hybrid_2d_3d_generator": {
        "generator_type": "hybrid_2d_3d_generator",
        "diffusion_processes": ["diffusion", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "chi_square_diffusion_generator": {
        "generator_type": "chi_square_diffusion_generator",
        "diffusion_processes": ["chi_square", "cold", "stable", "score_based"],
        "default_noise_type": "chi_square",
        "params": {"time_dim": 256},
    },
    "cold_diffusion_generator": {
        "generator_type": "cold_diffusion_generator",
        "diffusion_processes": ["cold", "diffusion", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "conditional_diffusion_generator": {
        "generator_type": "conditional_diffusion_generator",
        "diffusion_processes": ["diffusion", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "laplace_diffusion_generator": {
        "generator_type": "laplace_diffusion_generator",
        "diffusion_processes": ["laplace", "cold", "stable", "score_based"],
        "default_noise_type": "laplace",
        "params": {"time_dim": 256},
    },
    "rician_diffusion_generator": {
        "generator_type": "rician_diffusion_generator",
        "diffusion_processes": ["rician", "cold", "stable", "score_based"],
        "default_noise_type": "rician",
        "params": {"time_dim": 256},
    },
    "score_based_diffusion_generator": {
        "generator_type": "score_based_diffusion_generator",
        "diffusion_processes": ["score_based", "cold", "stable", "diffusion"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "stable_diffusion_generator": {
        "generator_type": "stable_diffusion_generator",
        "diffusion_processes": ["stable", "cold", "diffusion", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "diffusion_kan_generator": {
        "generator_type": "diffusion_kan_generator",
        "diffusion_processes": ["diffusion_kan", "cold", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "cold_diffusion_kan_generator": {
        "generator_type": "cold_diffusion_kan_generator",
        "diffusion_processes": ["cold_kan", "diffusion", "stable", "score_based"],
        "default_noise_type": "gaussian",
        "params": {"time_dim": 256},
    },
    "kspace_cold_diffusion": {
        "generator_type": "kspace_cold_diffusion",
        "diffusion_processes": ["cold", "diffusion"],
        "default_noise_type": "gaussian",
        "params": {
            "time_embedding_dim": 256,
            "use_complex_conv": True,
        },
    },
    "padnet": {
        "generator_type": "padnet",
        "diffusion_processes": [],
        "default_noise_type": "gaussian",  # Not used
        "params": {},
    },
    "mae": {
        "generator_type": "mae",
        "diffusion_processes": [],
        "default_noise_type": "gaussian",  # Not used
        "params": {},
    },
}

# Diffusion process configurations
DIFFUSION_PROCESSES = {
    "diffusion": {"class": Diffusion, "params": {}},
    "cold": {"class": ColdDiffusion, "params": {}},
    "stable": {"class": StableDiffusion, "params": {}},
    "score_based": {"class": ScoreBasedDiffusion, "params": {}},
    "chi_square": {
        "class": ChiSquareDiffusion,
        "params": {
            "min_dof": ("chi_square_min_dof", 1),
            "max_dof": ("chi_square_max_dof", 10),
        },
    },
    "laplace": {"class": LaplaceDiffusion, "params": {}},
    "rician": {
        "class": RicianDiffusion,
        "params": {"v_min": ("rician_v_min", 0.1), "v_max": ("rician_v_max", 10.0)},
    },
    "diffusion_kan": {"class": DiffusionKAN, "params": {}},
    "cold_kan": {"class": ColdDiffusionKAN, "params": {}},
}


def _generate_diffusion_configs():
    """Generate combinations of generators and diffusion processes."""
    configs = {}
    for gen_name, gen_config in GENERATOR_DIFFUSION_CONFIGS.items():
        for diff_name in gen_config["diffusion_processes"]:
            if diff_name in DIFFUSION_PROCESSES:
                key = f"{gen_name}_{diff_name}"
                configs[key] = {
                    "generator_type": gen_config["generator_type"],
                    "generator_params": gen_config["params"],
                    "diffusion_class": diff_name,
                    "diffusion_params": DIFFUSION_PROCESSES[diff_name]["params"],
                    "noise_type": gen_config["default_noise_type"],
                }
    return configs


# Generate the diffusion configurations
DIFFUSION_CONFIGS = _generate_diffusion_configs()
