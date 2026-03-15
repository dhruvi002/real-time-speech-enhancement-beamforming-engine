"""
models/crn.py
-------------
Frequency-strided CRN for CPU-efficient speech enhancement.
Encoder strides freq by 2 each layer: 129->65->33->17->9->5
Reduces compute ~26x vs no-stride. TemporalMLP replaces RNN.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=(2,3), freq_stride=2):
        super().__init__()
        self.conv   = nn.Conv2d(in_ch, out_ch, kernel_size=kernel,
                                stride=(freq_stride, 1),
                                padding=(kernel[0]-1, kernel[1]//2))
        self.t_trim = kernel[0] - 1
        self.bn     = nn.BatchNorm2d(out_ch)
        self.act    = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.t_trim > 0:
            x = x[:, :, :-self.t_trim, :]
        return self.act(self.bn(x))


class TransposeConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=(2,3), out_f=None, last=False):
        super().__init__()
        # Use interpolation + conv instead of ConvTranspose2d to avoid
        # output size ambiguity with strided decoder
        self.out_f  = out_f
        self.conv   = nn.Conv2d(in_ch, out_ch, kernel_size=(kernel[0], kernel[1]),
                                padding=(kernel[0]-1, kernel[1]//2))
        self.t_trim = kernel[0] - 1
        self.last   = last
        if not last:
            self.bn  = nn.BatchNorm2d(out_ch)
            self.act = nn.ELU(inplace=True)

    def forward(self, x):
        # Upsample freq axis to target size before conv
        if self.out_f is not None:
            x = nn.functional.interpolate(x, size=(self.out_f, x.shape[-1]),
                                           mode='nearest')
        x = self.conv(x)
        if self.t_trim > 0:
            x = x[:, :, :-self.t_trim, :]
        if not self.last:
            x = self.act(self.bn(x))
        return x


class TemporalMLP(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):   # x: (B, T, dim)
        return x + self.net(x)


class CRN(nn.Module):
    """
    Frequency-strided CRN.
    Input:  (B, 2, F, T)
    Output: mask_real, mask_imag each (B, F, T)
    """

    def __init__(
        self,
        in_channels:      int = 2,
        encoder_channels: List[int] = [16, 32, 64, 128, 256],
        encoder_kernel:   Tuple[int,int] = (2, 3),
        lstm_hidden:      int = 64,
        lstm_layers:      int = 1,
        n_freq_bins:      int = 129,
    ):
        super().__init__()
        self.n_freq_bins = n_freq_bins
        n_enc = len(encoder_channels)

        # ── Encoder ──────────────────────────────────────────────────
        enc_ch = [in_channels] + encoder_channels
        self.encoder = nn.ModuleList([
            ConvBlock(enc_ch[i], enc_ch[i+1], encoder_kernel, freq_stride=2)
            for i in range(n_enc)
        ])

        # Record freq size at each encoder stage via dummy pass
        with torch.no_grad():
            _x = torch.zeros(1, in_channels, n_freq_bins, 8)
            self.enc_f_sizes = [n_freq_bins]
            for layer in self.encoder:
                _x = layer(_x)
                self.enc_f_sizes.append(_x.shape[2])
        # enc_f_sizes[0]=F_in, enc_f_sizes[-1]=F_bottleneck
        f_bottleneck = self.enc_f_sizes[-1]

        # ── Temporal MLP ─────────────────────────────────────────────
        feat_dim = encoder_channels[-1] * f_bottleneck
        self.freq_compress = nn.Linear(feat_dim, lstm_hidden)
        self.temporal      = TemporalMLP(lstm_hidden, lstm_hidden * 2)
        self.freq_expand   = nn.Linear(lstm_hidden, feat_dim)
        self.f_bottleneck  = f_bottleneck
        self.enc_last_ch   = encoder_channels[-1]

        # ── Decoder: upsample freq back to each encoder stage size ────
        dec_in  = [encoder_channels[-1] * 2] + [
            encoder_channels[-(i+1)] * 2 for i in range(1, n_enc)
        ]
        dec_out = list(reversed(encoder_channels[:-1])) + [in_channels]
        # target freq sizes: reverse of encoder (skip bottleneck, include F_in)
        dec_f_targets = list(reversed(self.enc_f_sizes[:-1]))  # e.g. [9,17,33,65,129]

        self.decoder = nn.ModuleList([
            TransposeConvBlock(
                dec_in[i], dec_out[i], encoder_kernel,
                out_f=dec_f_targets[i],
                last=(i == n_enc - 1),
            )
            for i in range(n_enc)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, F, T = x.shape

        # Encode
        feat  = x
        skips = []
        for layer in self.encoder:
            feat = layer(feat)
            skips.append(feat)

        # Temporal MLP
        B2, C2, F2, T2 = feat.shape
        seq  = feat.permute(0, 3, 1, 2).reshape(B2, T2, C2 * F2)
        seq  = self.freq_compress(seq)
        seq  = self.temporal(seq)
        seq  = self.freq_expand(seq)
        feat = seq.reshape(B2, T2, C2, F2).permute(0, 2, 3, 1)

        # Decode with skip connections
        for i, layer in enumerate(self.decoder):
            skip = skips[-(i+1)]
            t    = min(feat.shape[-1], skip.shape[-1])
            feat = torch.cat([feat[:, :, :, :t], skip[:, :, :, :t]], dim=1)
            feat = layer(feat)

        # Crop to original F and T
        mask = torch.tanh(feat[:, :, :F, :T])
        return mask[:, 0], mask[:, 1]

    def enhance(
        self,
        noisy_real: torch.Tensor,
        noisy_imag: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x        = torch.stack([noisy_real, noisy_imag], dim=1)
        mr, mi   = self.forward(x)
        # Ensure mask matches input shape exactly
        mr = mr[:, :noisy_real.shape[1], :noisy_real.shape[2]]
        mi = mi[:, :noisy_imag.shape[1], :noisy_imag.shape[2]]
        return (mr * noisy_real - mi * noisy_imag,
                mr * noisy_imag + mi * noisy_real)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
