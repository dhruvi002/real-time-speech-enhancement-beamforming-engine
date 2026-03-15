"""
utils/audio.py
--------------
STFT / iSTFT wrappers, audio I/O, and signal utilities.

Uses center=True (PyTorch default) which is COLA-compliant on all backends.
istft runs on CPU to work around an MPS center=False bug, then moves back.
"""

import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
from typing import Tuple, Optional


class STFT(nn.Module):
    """
    Differentiable STFT returning (real, imag).
    win_length == n_fft, center=True — guaranteed COLA on CPU/MPS/CUDA.
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: int = 512,   # ignored, always set to n_fft internally
        window: str = "hann",
    ):
        super().__init__()
        self.n_fft      = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft   # force win == n_fft
        self.n_bins     = n_fft // 2 + 1
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:  x (B, T) or (T,)
        Returns: real, imag each (B, n_bins, n_frames)
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        if squeeze:
            return spec.real.squeeze(0), spec.imag.squeeze(0)
        return spec.real, spec.imag

    def inverse(
        self,
        real:   torch.Tensor,
        imag:   torch.Tensor,
        length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:  real, imag (B, n_bins, n_frames) or (n_bins, n_frames)
        Returns: waveform (B, T) or (T,)
        """
        squeeze = real.dim() == 2
        if squeeze:
            real = real.unsqueeze(0)
            imag = imag.unsqueeze(0)

        spec   = torch.complex(real, imag)
        device = spec.device

        # Run istft on CPU — avoids MPS COLA bugs — then move result back
        wav = torch.istft(
            spec.cpu(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.cpu(),
            center=True,
            length=length,
            return_complex=False,
        ).to(device)

        if squeeze:
            wav = wav.squeeze(0)
        return wav


def stft_magnitude_phase(
    real: torch.Tensor, imag: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    mag   = torch.sqrt(real**2 + imag**2 + 1e-8)
    phase = torch.atan2(imag, real)
    return mag, phase


def apply_mask(mr, mi, nr, ni):
    return mr*nr - mi*ni, mr*ni + mi*nr


# ── Audio I/O ─────────────────────────────────────────────────────────────

def load_audio(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    peak = np.abs(wav).max()
    if peak > 0:
        wav = wav / peak * 0.9
    return wav, target_sr


def save_audio(path: str, wav: np.ndarray, sr: int = 16000) -> None:
    sf.write(path, np.clip(wav, -1.0, 1.0), sr, subtype="PCM_16")


def mix_signals(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean) / len(noise))))
    noise = noise[:len(clean)]
    scale = np.sqrt(
        (np.mean(clean**2) + 1e-8) / ((np.mean(noise**2) + 1e-8) * 10**(snr_db/10))
    )
    return clean + scale*noise, scale*noise


def si_snr(clean: np.ndarray, enhanced: np.ndarray) -> float:
    clean    = clean    - clean.mean()
    enhanced = enhanced - enhanced.mean()
    target   = np.dot(enhanced, clean) / (np.dot(clean, clean) + 1e-8) * clean
    noise    = enhanced - target
    return 10 * np.log10(np.dot(target, target) / (np.dot(noise, noise) + 1e-8))