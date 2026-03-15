"""
data/fast_dataset.py
--------------------
Dataset that loads pre-mixed .npy pairs — pure numpy I/O, no audio processing.
10-20x faster than dataset.py for training.

Use after running precompute_dataset.py.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple


class PrecomputedDataset(Dataset):
    def __init__(self, manifest_path: str):
        with open(manifest_path) as f:
            self.items = json.load(f)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, float]:
        item  = self.items[idx]
        noisy = np.load(item["noisy"])   # (T,) float32
        clean = np.load(item["clean"])   # (T,) float32
        return (
            torch.from_numpy(noisy).unsqueeze(0),  # (1, T) — single channel
            torch.from_numpy(clean),                # (T,)
            0.0,
        )


def build_fast_dataloader(
    manifest_path: str,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    dataset = PrecomputedDataset(manifest_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )
