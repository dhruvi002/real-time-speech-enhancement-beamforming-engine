"""
training/losses.py
------------------
Loss functions for speech enhancement training.

SI-SNR Loss:
    Scale-invariant SNR, computed in time domain on reconstructed waveform.
    Maximizing SI-SNR ≈ minimizing distortion regardless of signal scale.

    L_sisnr = -SI-SNR(s_hat, s)
            = -10 log10( ||<s_hat,s>s / ||s||^2 ||^2 / ||s_hat - proj||^2 )

Spectral MSE Loss:
    MSE on log-magnitude spectrum — encourages correct spectral shape.

Phase-Aware Loss:
    MSE on complex spectrogram (real + imag separately).

Combined Loss:
    L = α·L_sisnr + β·L_mag + γ·L_phase
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SISNRLoss(nn.Module):
    """
    Scale-invariant SNR loss (higher = better → returns negative SI-SNR).
    Both inputs should be time-domain waveforms (B, T).
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, estimated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            estimated: (B, T) predicted waveform
            target:    (B, T) clean waveform
        Returns:
            scalar loss (mean over batch)
        """
        # Zero-mean
        target = target - target.mean(dim=-1, keepdim=True)
        estimated = estimated - estimated.mean(dim=-1, keepdim=True)

        # Target projection
        dot = (estimated * target).sum(dim=-1, keepdim=True)          # (B, 1)
        target_energy = (target ** 2).sum(dim=-1, keepdim=True) + self.eps
        proj = dot / target_energy * target                            # (B, T)

        # Noise component
        noise = estimated - proj                                       # (B, T)

        # SI-SNR
        si_snr = 10 * torch.log10(
            (proj ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + self.eps) + self.eps
        )  # (B,)

        return -si_snr.mean()


class SpectralMSELoss(nn.Module):
    """
    Log-magnitude MSE: penalizes errors in spectral envelope.
    Operates on (B, F, T) magnitude spectrograms.
    """

    def __init__(self, log_scale: bool = True, eps: float = 1e-8):
        super().__init__()
        self.log_scale = log_scale
        self.eps = eps

    def forward(self, estimated_mag: torch.Tensor, target_mag: torch.Tensor) -> torch.Tensor:
        if self.log_scale:
            estimated_mag = torch.log(estimated_mag + self.eps)
            target_mag = torch.log(target_mag + self.eps)
        return F.mse_loss(estimated_mag, target_mag)


class ComplexMSELoss(nn.Module):
    """
    Phase-aware complex MSE on real + imag components.
    Both inputs are (B, F, T).
    """

    def forward(
        self,
        est_real: torch.Tensor,
        est_imag: torch.Tensor,
        tgt_real: torch.Tensor,
        tgt_imag: torch.Tensor,
    ) -> torch.Tensor:
        loss_real = F.mse_loss(est_real, tgt_real)
        loss_imag = F.mse_loss(est_imag, tgt_imag)
        return loss_real + loss_imag


class CombinedLoss(nn.Module):
    """
    Weighted combination of SI-SNR + spectral magnitude + complex MSE.

        L = α·L_sisnr + β·L_mag + γ·L_phase

    Default weights from: α=0.8, β=0.1, γ=0.1
    (SI-SNR dominates; spectral terms provide stability early in training)
    """

    def __init__(
        self,
        sisnr_weight: float = 0.8,
        mag_weight: float = 0.1,
        phase_weight: float = 0.1,
    ):
        super().__init__()
        self.sisnr_w = sisnr_weight
        self.mag_w = mag_weight
        self.phase_w = phase_weight

        self.sisnr_loss = SISNRLoss()
        self.mag_loss = SpectralMSELoss()
        self.phase_loss = ComplexMSELoss()

    def forward(
        self,
        # Time-domain
        est_wav: torch.Tensor,        # (B, T)
        tgt_wav: torch.Tensor,        # (B, T)
        # Spectral
        est_real: torch.Tensor,       # (B, F, T)
        est_imag: torch.Tensor,
        tgt_real: torch.Tensor,
        tgt_imag: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:

        # Magnitude
        est_mag = torch.sqrt(est_real ** 2 + est_imag ** 2 + 1e-8)
        tgt_mag = torch.sqrt(tgt_real ** 2 + tgt_imag ** 2 + 1e-8)

        l_sisnr = self.sisnr_loss(est_wav, tgt_wav)
        l_mag   = self.mag_loss(est_mag, tgt_mag)
        l_phase = self.phase_loss(est_real, est_imag, tgt_real, tgt_imag)

        total = (
            self.sisnr_w * l_sisnr
            + self.mag_w  * l_mag
            + self.phase_w * l_phase
        )

        metrics = {
            "loss/total":  total.item(),
            "loss/sisnr":  l_sisnr.item(),
            "loss/mag":    l_mag.item(),
            "loss/phase":  l_phase.item(),
        }
        return total, metrics
