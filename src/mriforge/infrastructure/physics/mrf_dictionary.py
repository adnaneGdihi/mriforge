"""
MRF Dictionary Generation and Matching Module.

This module implements the dictionary generation and matching logic for MRF-DiPh.
It simulates Bloch responses over a grid of tissue parameters (T1, T2) and
off-resonance (B0), and provides methods to match observed signals to the
dictionary to recover quantitative maps.

Off-resonance is encoded as a per-echo phase ``exp(i·2π·B0·TE)`` on the
(otherwise magnitude) steady-state signal, so the fingerprints are COMPLEX and
the B0 grid axis is recoverable from a complex signal (a real/magnitude signal
carries no phase, so ``match`` correctly returns B0≈0 for it). NOTE: the
underlying ``DifferentiableBlochSimulator`` is steady-state (Ernst), not a
transient MRF simulator — the dictionary is a steady-state approximation.

Theoretical Foundation:
    The MRF signal evolution S for a tissue with parameters M = (T1, T2) is given by:
    S = B(M; theta)
    where B is the Bloch operator and theta represents the sequence parameters.

    The projection step in MRF-DiPh solves:
    hat{M} = argmin_M || B(M) - x ||^2
    via dictionary matching.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.infrastructure.physics.bloch_simulation import DifferentiableBlochSimulator


class BlochDictionary(nn.Module):
    """
    Manages the generation and matching of Bloch response dictionaries.
    """

    def __init__(
        self,
        t1_range: tuple[float, float] = (100.0, 3000.0),
        t2_range: tuple[float, float] = (10.0, 500.0),
        b0_range: tuple[float, float] = (-200.0, 200.0),
        t1_step: float = 10.0,
        t2_step: float = 2.0,
        b0_step: float = 50.0,
        device: str = "cpu",
    ):
        """__init__.

        Args:
            t1_range (tuple[float, float]): Description.
            t2_range (tuple[float, float]): Description.
            b0_range (tuple[float, float]): Description.
            t1_step (float): Description.
            t2_step (float): Description.
            b0_step (float): Description.
            device (str): Description.
        """
        super().__init__()
        self.t1_range = t1_range
        self.t2_range = t2_range
        self.b0_range = b0_range
        self.t1_step = t1_step
        self.t2_step = t2_step
        self.b0_step = b0_step
        self.device = device

        self.simulator = DifferentiableBlochSimulator(sequence_type="GRE").to(device)

        # Buffers for the dictionary
        self.register_buffer("dictionary", None)
        self.register_buffer(
            "params", None
        )  # Stores (T1, T2, B0) pairs corresponding to dictionary entries

    def generate(self, sequence_params: list[dict[str, float]]):
        """
        Generates the dictionary for the given sequence parameters.

        Args:
            sequence_params: List of dictionaries, each containing 'TR', 'TE', 'alpha'
                             for each time point in the fingerprint.
        """
        # Create grid of T1, T2, B0 values. The B0 axis is now LOAD-BEARING:
        # off-resonance enters the fingerprint as a per-echo phase
        # ``exp(i·2π·B0·TE)`` (below), so different B0 grid points give distinct
        # COMPLEX fingerprints and ``match`` recovers B0 from a complex signal.
        # (Previously b0_range/b0_step were inert knobs — b0 was hardcoded to 0
        # Hz, so est_b0 was always 0: pitfall #15/#20.)
        t1_vals = torch.arange(
            self.t1_range[0],
            self.t1_range[1] + self.t1_step,
            self.t1_step,
            device=self.device,
        )
        t2_vals = torch.arange(
            self.t2_range[0],
            self.t2_range[1] + self.t2_step,
            self.t2_step,
            device=self.device,
        )
        b0_vals = torch.arange(
            self.b0_range[0],
            self.b0_range[1] + self.b0_step,
            self.b0_step,
            device=self.device,
        )

        # Create meshgrid over (T1, T2, B0)
        t1_grid, t2_grid, b0_grid = torch.meshgrid(t1_vals, t2_vals, b0_vals, indexing="ij")
        t1_flat = t1_grid.flatten()
        t2_flat = t2_grid.flatten()
        b0_flat = b0_grid.flatten()

        # PD is assumed to be 1.0 for dictionary generation (intensity scaling handled during matching or assumed normalized)
        pd_flat = torch.ones_like(t1_flat)

        # Stack into tissue maps: [N, 3, 1, 1] for the simulator (Simulator expects [Batch, 3, H, W])
        # We treat each dictionary entry as a "pixel"
        tissue_maps = torch.stack([pd_flat, t1_flat, t2_flat], dim=1).unsqueeze(-1).unsqueeze(-1)

        fingerprints = []

        # Simulate sequence
        # Note: DifferentiableBlochSimulator is stateless (steady-state approx).
        # MRF typically requires transient state simulation.
        # However, if the sequence is just varying parameters for steady state (e.g. variable flip angle GRE),
        # we can use the current simulator.
        # If true MRF (transient), we would need a stateful simulator.
        # Given the "Physics-Informed Diffusion" context, we assume the provided simulator is sufficient
        # for the "Bloch Manifold" projection, which might be defined by steady-state equations in this specific implementation
        # or we accept the limitation of the current simulator.
        # We proceed with the current simulator as the "Bloch operator".

        two_pi = 2.0 * torch.pi
        for params in sequence_params:
            tr = params.get("TR", 10.0)
            te = params.get("TE", 2.0)
            alpha = params.get("alpha", 10.0)

            magnitude = self.simulator(tissue_maps, TR=tr, TE=te, alpha=alpha).view(-1)
            # Off-resonance phase accrued at the echo time (TE in ms -> s):
            # φ = 2π · B0[Hz] · TE[s]. This is what makes the B0 axis recoverable
            # — the phase evolution across a TE-varying sequence differs per B0.
            phase = two_pi * b0_flat * (te / 1000.0)  # [N]
            signal = magnitude.to(torch.cfloat) * torch.exp(1j * phase)  # [N] complex
            fingerprints.append(signal)

        # Stack time points: [N, T] (complex)
        self.dictionary = torch.stack(fingerprints, dim=1)

        # Normalize dictionary (ℓ₂ magnitude norm) for correlation matching.
        norms = torch.linalg.vector_norm(self.dictionary, dim=1, keepdim=True)
        self.dictionary = self.dictionary / (norms + 1e-8)

        self.params = torch.stack([t1_flat, t2_flat, b0_flat], dim=1)

        return self.dictionary, self.params

    def match(self, signals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Finds the best matching T1, T2, B0 maps for the given signals.

        Args:
            signals: [Batch, T, H, W] or [Batch, T] tensor of observed signals (fingerprints).

        Returns:
            est_t1: Estimated T1 map
            est_t2: Estimated T2 map
            est_b0: Estimated B0 map
        """
        if self.dictionary is None:
            raise RuntimeError("Dictionary not generated. Call generate() first.")

        # Reshape signals to [N_pixels, T]
        if signals.dim() == 4:
            batch, t, h, w = signals.shape
            flat_signals = signals.permute(0, 2, 3, 1).contiguous().view(-1, t)
        else:
            flat_signals = signals

        # Cast to the dictionary's (complex) dtype so a real/magnitude input is
        # handled too — for a real signal the B0-phased atoms all correlate best
        # with the zero-phase (B0≈0) entry, so est_b0≈0, which is the honest
        # answer: you cannot measure off-resonance from magnitude data.
        flat_signals = flat_signals.to(self.dictionary.dtype)

        # Normalize signals (ℓ₂ magnitude norm)
        norms = torch.linalg.vector_norm(flat_signals, dim=1, keepdim=True)
        flat_signals_norm = flat_signals / (norms + 1e-8)

        # Complex correlation magnitude |<s, d>| = |s · conj(d)ᵀ|.
        # Dictionary: [D, T] complex ; Signals: [P, T] complex ; Result: [P, D]
        scores = torch.matmul(flat_signals_norm, self.dictionary.conj().T).abs()

        # Find max correlation
        best_indices = torch.argmax(scores, dim=1)

        # Retrieve params
        best_params = self.params[best_indices]  # [P, 3]

        t1 = best_params[:, 0]
        t2 = best_params[:, 1]
        b0 = best_params[:, 2]

        # Reshape back if input was image
        if signals.dim() == 4:
            est_t1 = t1.view(batch, h, w)
            est_t2 = t2.view(batch, h, w)
            est_b0 = b0.view(batch, h, w)
        else:
            est_t1 = t1
            est_t2 = t2
            est_b0 = b0

        return est_t1, est_t2, est_b0

    def project_signal(self, signals: torch.Tensor) -> torch.Tensor:
        """
        Projects the signals onto the Bloch manifold.
        Equivalent to: B(M_hat) where M_hat = argmin ||B(M) - signals||

        Args:
            signals: [Batch, T, H, W]

        Returns:
            projected_signals: [Batch, T, H, W] - The best matching dictionary entries.
        """
        if self.dictionary is None:
            raise RuntimeError("Dictionary not generated. Call generate() first.")

        if signals.dim() == 4:
            batch, t, h, w = signals.shape
            flat_signals = signals.permute(0, 2, 3, 1).contiguous().view(-1, t)
        else:
            flat_signals = signals

        # Cast to the dictionary's (complex) dtype so real inputs are handled.
        flat_signals = flat_signals.to(self.dictionary.dtype)

        # Best match by complex correlation magnitude |<s, d>|.
        input_norms = torch.linalg.vector_norm(flat_signals, dim=1, keepdim=True)
        flat_signals_norm = flat_signals / (input_norms + 1e-8)
        scores = torch.matmul(flat_signals_norm, self.dictionary.conj().T).abs()
        best_indices = torch.argmax(scores, dim=1)

        # Retrieve best dictionary entries
        best_fingerprints = self.dictionary[best_indices]  # [P, T] complex

        # Project onto the unit-norm atom: proj = <s, d> · d, where the complex
        # coefficient <s, d> = Σ s_t · conj(d_t) recovers both the PD magnitude
        # and the phase of the matched fingerprint.
        projection_coefficients = torch.sum(
            flat_signals * best_fingerprints.conj(), dim=1, keepdim=True
        )  # [P, 1] complex

        projected_flat = best_fingerprints * projection_coefficients

        if signals.dim() == 4:
            projected_signals = projected_flat.view(batch, h, w, t).permute(0, 3, 1, 2)
        else:
            projected_signals = projected_flat

        return projected_signals
