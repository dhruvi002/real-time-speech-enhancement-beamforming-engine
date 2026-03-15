"""
beamforming/mvdr.py
-------------------
Minimum Variance Distortionless Response (MVDR) beamformer.

For an M-microphone array, MVDR solves:
    w* = argmin  w^H Phi_nn w
         s.t.    w^H d(theta) = 1

Closed-form solution:
    w_mvdr = Phi_nn^{-1} d / (d^H Phi_nn^{-1} d)

where:
    Phi_nn  : noise PSD matrix  (M x M)
    d       : relative transfer function (RTF) steering vector (M,)

The RTF is estimated from the noisy covariance using the first
microphone as reference (oracle-free, data-driven approach).

Reference:
    Souden et al., "On Optimal Frequency-Domain Multichannel Linear
    Filtering for Noise Reduction", IEEE TASLP 2010.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple


class MVDRBeamformer(nn.Module):
    """
    MVDR beamformer operating in the STFT domain.

    Input:  multi-channel STFT  (B, M, F, T)   complex
    Output: beamformed signal   (B, F, T)        complex
    """

    def __init__(
        self,
        n_mics: int = 4,
        n_fft: int = 512,
        ref_mic: int = 0,
        diag_load: float = 1e-5,
    ):
        super().__init__()
        self.n_mics = n_mics
        self.n_fft = n_fft
        self.n_bins = n_fft // 2 + 1
        self.ref_mic = ref_mic
        self.diag_load = diag_load  # diagonal loading for numerical stability

    def forward(
        self,
        mc_spec: torch.Tensor,
        speech_mask: Optional[torch.Tensor] = None,
        noise_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            mc_spec:     (B, M, F, T) complex STFT of multi-channel input
            speech_mask: (B, F, T) soft mask for speech regions  [0,1]
            noise_mask:  (B, F, T) soft mask for noise regions   [0,1]

        Returns:
            enhanced:    (B, F, T) complex beamformed signal
        """
        B, M, F, T = mc_spec.shape

        # ── Default masks: first 20% frames = noise, rest = speech ─────
        if speech_mask is None or noise_mask is None:
            speech_mask, noise_mask = self._default_masks(B, F, T, mc_spec.device)

        enhanced_list = []
        for b in range(B):
            spec_b = mc_spec[b]          # (M, F, T)
            sm = speech_mask[b]          # (F, T)
            nm = noise_mask[b]           # (F, T)

            enhanced_b = self._beamform_single(spec_b, sm, nm)  # (F, T)
            enhanced_list.append(enhanced_b)

        return torch.stack(enhanced_list, dim=0)  # (B, F, T)

    def _beamform_single(
        self,
        spec: torch.Tensor,     # (M, F, T) complex
        speech_mask: torch.Tensor,  # (F, T)
        noise_mask: torch.Tensor,   # (F, T)
    ) -> torch.Tensor:
        """Compute MVDR beamformed output for a single utterance."""
        M, F, T = spec.shape
        enhanced = torch.zeros(F, T, dtype=torch.complex64, device=spec.device)

        for f in range(F):
            # Frames for this frequency bin
            Y = spec[:, f, :]          # (M, T)
            sm = speech_mask[f, :]     # (T,)
            nm = noise_mask[f, :]      # (T,)

            # Weighted covariance matrices
            # Phi_ss = sum_t mask_s(t) * y(t) y(t)^H
            sm_sum = sm.sum() + 1e-8
            nm_sum = nm.sum() + 1e-8

            # Y weighted by mask: (M, T) -> sum over T
            Phi_ss = (Y * sm.unsqueeze(0)) @ Y.conj().T / sm_sum   # (M, M)
            Phi_nn = (Y * nm.unsqueeze(0)) @ Y.conj().T / nm_sum   # (M, M)

            # Diagonal loading
            Phi_nn = Phi_nn + self.diag_load * torch.eye(M, dtype=Phi_nn.dtype, device=Phi_nn.device)

            # RTF estimation: principal eigenvector of Phi_nn^{-1} Phi_ss
            # Equivalent to generalized eigenvector problem
            try:
                Phi_nn_inv = torch.linalg.inv(Phi_nn)
                # Steering vector via GEV
                G = Phi_nn_inv @ Phi_ss                   # (M, M)
                eigenvalues, eigenvectors = torch.linalg.eig(G)
                # Take eigenvector with largest real eigenvalue
                idx = eigenvalues.real.argmax()
                rtf = eigenvectors[:, idx]                # (M,) complex

                # Normalize: w^H d(ref) = 1
                rtf = rtf / (rtf[self.ref_mic] + 1e-8)

                # MVDR weights
                Phi_nn_inv_d = Phi_nn_inv @ rtf           # (M,)
                denom = (rtf.conj() @ Phi_nn_inv_d).real + 1e-8
                w = Phi_nn_inv_d / denom                  # (M,)

                # Apply weights
                enhanced[f, :] = w.conj() @ Y            # (T,)

            except Exception:
                # Fallback: delay-and-sum using reference mic
                enhanced[f, :] = Y[self.ref_mic, :]

        return enhanced

    @staticmethod
    def _default_masks(
        B: int, F: int, T: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Heuristic masks when oracle masks not available:
        - noise:  first 15% of frames
        - speech: remaining 85%
        """
        noise_frames = max(1, int(T * 0.15))
        noise_mask = torch.zeros(B, F, T, device=device)
        noise_mask[:, :, :noise_frames] = 1.0
        speech_mask = torch.zeros(B, F, T, device=device)
        speech_mask[:, :, noise_frames:] = 1.0
        return speech_mask, noise_mask


# ── Array geometry utilities ───────────────────────────────────────────────

def ula_steering_vector(
    n_mics: int,
    spacing: float,
    doa_deg: float,
    freq_hz: float,
    c: float = 343.0,
) -> np.ndarray:
    """
    Compute ULA steering vector for a given DOA and frequency.

    Args:
        n_mics:   number of microphones
        spacing:  inter-element spacing in meters
        doa_deg:  direction of arrival in degrees (broadside = 90)
        freq_hz:  frequency in Hz
        c:        speed of sound (m/s)

    Returns:
        d: complex steering vector (n_mics,)
    """
    theta = np.deg2rad(doa_deg)
    tau = spacing * np.cos(theta) / c          # time delay between adjacent mics
    k = 2 * np.pi * freq_hz / c
    d = np.exp(-1j * k * spacing * np.cos(theta) * np.arange(n_mics))
    return d


def time_delay_of_arrival(
    mic_positions: np.ndarray,   # (M, 3)
    source_position: np.ndarray, # (3,)
    c: float = 343.0,
) -> np.ndarray:
    """
    Compute TDOA for each microphone relative to reference mic 0.
    Returns delays in seconds (M,).
    """
    dists = np.linalg.norm(mic_positions - source_position, axis=1)  # (M,)
    tdoa = (dists - dists[0]) / c
    return tdoa


def build_ula_positions(
    n_mics: int = 4,
    spacing: float = 0.05,      # 5 cm
) -> np.ndarray:
    """
    Build linear microphone array positions (M, 3).
    Array is along x-axis, centered at origin.
    """
    x = np.arange(n_mics) * spacing
    x -= x.mean()
    positions = np.zeros((n_mics, 3))
    positions[:, 0] = x
    return positions
