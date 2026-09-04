"""VAE implementations for MRI super-resolution."""

from .decoder import VAEDecoder
from .encoder import VAEEncoder
from .fourier_kan_decoder import FourierKANDecoder
from .fourier_kan_encoder import FourierKANEncoder
from .fourier_kan_vae import FourierKANVAE
from .sparse_vae import SparseVAE, create_sparse_vae
from .vae import VAE, create_vae
from .wavelet_kan_decoder import WaveletKANDecoder
from .wavelet_kan_encoder import WaveletKANEncoder
from .wavelet_kan_vae import WaveletKANVAE

__all__ = [
    "VAE",
    "FourierKANDecoder",
    "FourierKANEncoder",
    "FourierKANVAE",
    "SparseVAE",
    "VAEDecoder",
    "VAEEncoder",
    "WaveletKANDecoder",
    "WaveletKANEncoder",
    "WaveletKANVAE",
    "create_sparse_vae",
    "create_vae",
]
