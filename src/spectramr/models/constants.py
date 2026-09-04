"""Model Architecture Constants
=============================

Centralized definition of magic numbers used across model implementations.
This ensures consistency and easy experimentation with different architectures.

All constants follow UPPER_SNAKE_CASE naming convention.
"""

# ============================================================================
# Channel Progression Constants
# ============================================================================

# Standard channel progression for encoder-decoder architectures
DEFAULT_CHANNEL_PROGRESSION = [64, 128, 256, 512]

# Alternative channel progressions for different model scales
SMALL_CHANNEL_PROGRESSION = [32, 64, 128, 256]
MEDIUM_CHANNEL_PROGRESSION = [64, 128, 256, 512]
LARGE_CHANNEL_PROGRESSION = [128, 256, 512, 1024]

# Channel progression for residual U-Net variants
RESIDUAL_CHANNEL_PROGRESSION = [64, 128, 256, 512, 512]

# Channel progression for multi-scale architectures
MULTISCALE_CHANNEL_PROGRESSION = [32, 64, 128, 256, 512]

# ============================================================================
# VAE & Latent Space Constants
# ============================================================================

# Default latent dimension for VAE models
DEFAULT_VAE_LATENT_DIM = 256

# Alternative VAE latent dimensions
VAE_LATENT_DIM_SMALL = 128
VAE_LATENT_DIM_MEDIUM = 256
VAE_LATENT_DIM_LARGE = 512

# KL divergence beta schedule constants
DEFAULT_VAE_BETA = 1.0
VAE_BETA_ANNEALING_START = 0.0
VAE_BETA_ANNEALING_END = 1.0

# ============================================================================
# KAN (Kolmogorov-Arnold Network) Constants
# ============================================================================

# Default number of KAN wavelet bases
DEFAULT_KAN_NUM_BASES = 8

# Alternative KAN configurations
KAN_NUM_BASES_MINIMAL = 4
KAN_NUM_BASES_STANDARD = 8
KAN_NUM_BASES_RICH = 16

# Wavelet families supported by KAN layers
DEFAULT_WAVELET_FAMILY = "haar"
SUPPORTED_WAVELET_FAMILIES = ["haar", "db2", "db4", "sym2", "sym4"]

# Fourier KAN maximum frequency
DEFAULT_FOURIER_MAX_FREQUENCY = 1.0

# ============================================================================
# Bias Field & Field Simulation Constants
# ============================================================================

# Bias field polynomial order
BIAS_FIELD_POLYNOMIAL_ORDER = 3
BIAS_FIELD_POLYNOMIAL_COEFFICIENTS = 10  # (order+1) * (order+2) / 2 for 2D

# Bias field value range (multiplicative factor)
BIAS_FIELD_MIN_VALUE = 0.3
BIAS_FIELD_MAX_VALUE = 3.0

# Gaussian smoothing kernel parameters
BIAS_FIELD_SMOOTHING_SIGMA = 2.0
BIAS_FIELD_SMOOTHING_KERNEL_SIZE = 7

# ============================================================================
# Spatial Dimensions & Pooling Constants
# ============================================================================

# Default input spatial dimensions for MRI images
DEFAULT_IMAGE_HEIGHT = 64
DEFAULT_IMAGE_WIDTH = 64
DEFAULT_IMAGE_DEPTH = 64

# Pooling stride for downsampling
DEFAULT_POOL_STRIDE = 2
DEFAULT_POOL_KERNEL_SIZE = 2

# ============================================================================
# Attention & Normalization Constants
# ============================================================================

# Attention mechanism constants
DEFAULT_NUM_HEADS = 8
DEFAULT_ATTENTION_HEAD_DIM = 64

# Normalization epsilon for numerical stability
LAYER_NORM_EPSILON = 1e-6
INSTANCE_NORM_EPSILON = 1e-5
BATCH_NORM_MOMENTUM = 0.99

# ============================================================================
# Activation & Regularization Constants
# ============================================================================

# Dropout rates for regularization
DEFAULT_DROPOUT_RATE = 0.1
ENCODER_DROPOUT_RATE = 0.1
DECODER_DROPOUT_RATE = 0.05

# Activation function ReLU inplace mode (for memory efficiency)
ACTIVATION_INPLACE = True

# ============================================================================
# Loss Function Constants
# ============================================================================

# Reconstruction loss weights
DEFAULT_L1_WEIGHT = 1.0
DEFAULT_L2_WEIGHT = 0.0
DEFAULT_PERCEPTUAL_WEIGHT = 0.0

# Adversarial loss weight
DEFAULT_ADVERSARIAL_WEIGHT = 0.1

# KL divergence weight for VAE
DEFAULT_KL_WEIGHT = 0.001

# Physics-informed loss weights
DEFAULT_DATA_CONSISTENCY_WEIGHT = 1.0
DEFAULT_KSPACE_WEIGHT = 0.5

# ============================================================================
# Diffusion Constants
# ============================================================================

# Default timesteps for diffusion models
DEFAULT_DIFFUSION_TIMESTEPS = 1000
DEFAULT_DIFFUSION_SAMPLING_STEPS = 50

# Noise schedule parameters
DIFFUSION_BETA_START = 0.0001
DIFFUSION_BETA_END = 0.02
DIFFUSION_SCHEDULE_TYPE = "linear"  # linear, quadratic, sqrt

# ============================================================================
# Complex-Valued Network Constants
# ============================================================================

# Complex number representation in real-valued networks
COMPLEX_CHANNELS = 2  # Real and imaginary parts

# Magnitude and phase representation
MAGNITUDE_CHANNEL_INDEX = 0
PHASE_CHANNEL_INDEX = 1

# ============================================================================
# Feature Extraction & Encoder Constants
# ============================================================================

# Max pooling after each encoder layer
ENCODER_POOL_STRIDE = 2

# Number of encoder blocks
DEFAULT_NUM_ENCODER_BLOCKS = 4

# Skip connection mode
SKIP_CONNECTION_MODE = "concat"  # options: concat, add, none

# ============================================================================
# Upsampling & Decoder Constants
# ============================================================================

# Upsampling mode for decoder
DECODER_UPSAMPLE_MODE = "bilinear"  # options: bilinear, nearest, bicubic
DECODER_ALIGN_CORNERS = False

# Scale factor for each upsampling step
DEFAULT_UPSAMPLE_SCALE_FACTOR = 2

# Number of decoder blocks
DEFAULT_NUM_DECODER_BLOCKS = 4

# ============================================================================
# Miscellaneous Constants
# ============================================================================

# Default input/output channels for medical imaging
DEFAULT_IN_CHANNELS = 1  # Grayscale MRI
DEFAULT_OUT_CHANNELS = 1  # Single modality output

# Multi-coil imaging
DEFAULT_NUM_COILS = 1  # Single-coil by default
MAX_NUM_COILS = 32

# Batch normalization track running stats
TRACK_RUNNING_STATS = True

# Initialization method for weights
WEIGHT_INIT_METHOD = "kaiming_normal"  # options: kaiming_normal, xavier_uniform

# ============================================================================
# Module Docstring Example Usage:
# ============================================================================
#
# from spectramr.models.constants import (
#     DEFAULT_CHANNEL_PROGRESSION,
#     DEFAULT_VAE_LATENT_DIM,
#     DEFAULT_KAN_NUM_BASES,
#     DEFAULT_POOL_STRIDE,
# )
#
# class MyEncoder(nn.Module):
#     def __init__(self, in_channels=1, latent_dim=DEFAULT_VAE_LATENT_DIM):
#         super().__init__()
#         channels = DEFAULT_CHANNEL_PROGRESSION
#         self.enc1 = nn.Conv2d(in_channels, channels[0], 3, padding=1)
#         self.enc2 = nn.Conv2d(channels[0], channels[1], 3, padding=1)
#         self.pool = nn.MaxPool2d(DEFAULT_POOL_STRIDE)
#         ...
#
# ============================================================================
