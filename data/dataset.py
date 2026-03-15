"""
data/dataset.py
---------------
Dataset for speech enhancement training.

Data pipeline:
    1. Load clean utterance from LibriSpeech
    2. Load noise segment from DEMAND corpus
    3. Optionally convolve with a room impulse response (RIR)
    4. Mix at random SNR ∈ [snr_min, snr_max] dB
    5. Simulate 4-channel array signal by convolving with per-mic RIRs
    6. Return: (noisy_multichannel, clean_mono, snr)

For training, clean = reference mic signal convolved with RIR.
For evaluation, clean = original dry clean utterance.
"""

import json
import random
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import soundfile as sf
import scipy.signal as signal

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.audio import load_audio, mix_signals


class SpeechEnhancementDataset(Dataset):
    """
    Reads a JSON manifest with fields:
        {
            "clean": "path/to/clean.wav",
            "noise": "path/to/noise.wav",
            "snr_db": 5.0,                   # optional, sampled if missing
            "rir":   "path/to/rir.npy"        # optional, augmentation
        }

    Returns:
        noisy_mc:  (n_mics, T) float32  — multi-channel noisy
        clean:     (T,)        float32  — reference clean
        snr_db:    float
    """

    def __init__(
        self,
        manifest_path: str,
        sample_rate: int = 16000,
        duration: float = 4.0,
        snr_range: Tuple[float, float] = (-5.0, 20.0),
        n_mics: int = 4,
        rir_prob: float = 0.8,
        augment: bool = True,
    ):
        self.sr = sample_rate
        self.n_samples = int(duration * sample_rate)
        self.snr_range = snr_range
        self.n_mics = n_mics
        self.rir_prob = rir_prob
        self.augment = augment

        with open(manifest_path) as f:
            self.samples: List[Dict] = json.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        item = self.samples[idx]

        # ── Load clean & noise ───────────────────────────────────────────
        clean, _ = load_audio(item["clean"], self.sr)
        noise, _ = load_audio(item["noise"], self.sr)

        # Trim / pad to fixed duration
        clean = self._fix_length(clean)
        noise = self._fix_length(noise)

        snr_db = item.get("snr_db", random.uniform(*self.snr_range))

        # ── RIR augmentation ─────────────────────────────────────────────
        if self.augment and "rir" in item and random.random() < self.rir_prob:
            rir_data = np.load(item["rir"])   # (n_mics, rir_len)
            # Convolve each mic
            noisy_mc = self._apply_rir_array(clean, noise, rir_data, snr_db)
            clean_ref = self._convolve(clean, rir_data[0])  # ref mic = mic 0
            clean_ref = self._fix_length(clean_ref)
        else:
            # No RIR: same signal on all mics (single-channel fallback)
            noisy_mono, _ = mix_signals(clean, noise, snr_db)
            noisy_mc = np.stack([noisy_mono] * self.n_mics, axis=0)
            clean_ref = clean

        # Normalize
        noisy_mc = noisy_mc / (np.abs(noisy_mc).max() + 1e-8) * 0.9
        clean_ref = clean_ref / (np.abs(clean_ref).max() + 1e-8) * 0.9

        return (
            torch.tensor(noisy_mc, dtype=torch.float32),   # (M, T)
            torch.tensor(clean_ref, dtype=torch.float32),  # (T,)
            snr_db,
        )

    def _fix_length(self, wav: np.ndarray) -> np.ndarray:
        if len(wav) >= self.n_samples:
            start = random.randint(0, len(wav) - self.n_samples)
            return wav[start: start + self.n_samples]
        else:
            return np.pad(wav, (0, self.n_samples - len(wav)))

    def _convolve(self, wav: np.ndarray, rir: np.ndarray) -> np.ndarray:
        out = signal.fftconvolve(wav, rir)
        return out[: len(wav)]

    def _apply_rir_array(
        self,
        clean: np.ndarray,
        noise: np.ndarray,
        rir: np.ndarray,       # (n_mics, rir_len)
        snr_db: float,
    ) -> np.ndarray:
        """Convolve clean + noise with per-mic RIRs and mix at snr_db."""
        n_mics = rir.shape[0]
        noisy_mc = np.zeros((n_mics, self.n_samples))

        for m in range(n_mics):
            clean_m = self._fix_length(self._convolve(clean, rir[m]))
            noise_m = self._fix_length(self._convolve(noise, rir[m]))
            noisy_m, _ = mix_signals(clean_m, noise_m, snr_db)
            noisy_mc[m] = self._fix_length(noisy_m)

        return noisy_mc


def build_dataloader(
    manifest_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    dataset = SpeechEnhancementDataset(manifest_path, **dataset_kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )