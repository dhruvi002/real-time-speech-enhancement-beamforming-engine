"""
data/precompute_dataset.py
--------------------------
Pre-mixes all noisy-clean pairs to disk so training is pure I/O.
Run once before training — typically takes 10-20 minutes for 20k pairs.

Usage:
    python data/precompute_dataset.py --split train --n_pairs 20000
    python data/precompute_dataset.py --split val   --n_pairs 2000
    python data/precompute_dataset.py --split test  --n_pairs 1000

Output:
    data/precomputed/train/
        000000_noisy.npy   # (T,) float32 — ref mic noisy
        000000_clean.npy   # (T,) float32 — clean reference
        ...
    data/precomputed/train_manifest.json
"""

import argparse
import json
import sys
import random
import numpy as np
import soundfile as sf
import scipy.signal as signal
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.audio import load_audio, mix_signals


def process_item(item: dict, n_samples: int, sr: int) -> tuple:
    """Load, optionally convolve with RIR, mix at SNR. Returns (noisy, clean)."""
    clean, _ = load_audio(item["clean"], sr)
    noise, _ = load_audio(item["noise"], sr)

    # Trim / pad
    def fix(w):
        if len(w) >= n_samples:
            start = random.randint(0, len(w) - n_samples)
            return w[start: start + n_samples]
        return np.pad(w, (0, n_samples - len(w)))

    clean = fix(clean)
    noise = fix(noise)
    snr   = item.get("snr_db", random.uniform(-5, 20))

    if "rir" in item and Path(item["rir"]).exists():
        rir = np.load(item["rir"])   # (n_mics, rir_len)
        rir_ref = rir[0]
        def conv(w):
            out = signal.fftconvolve(w, rir_ref)[:len(w)]
            return out.astype(np.float32)
        clean_r = fix(conv(clean))
        noise_r = fix(conv(noise))
        noisy, _ = mix_signals(clean_r, noise_r, snr)
        clean_out = clean_r
    else:
        noisy, _ = mix_signals(clean, noise, snr)
        clean_out = clean

    # Normalize
    noisy     = (noisy     / (np.abs(noisy).max()     + 1e-8) * 0.9).astype(np.float32)
    clean_out = (clean_out / (np.abs(clean_out).max() + 1e-8) * 0.9).astype(np.float32)
    return noisy, clean_out


def main(args):
    sr       = 16000
    n_samples = int(args.duration * sr)

    manifest_path = f"data/manifests/{args.split}.json"
    with open(manifest_path) as f:
        items = json.load(f)

    # Sample n_pairs from manifest
    random.seed(42)
    items = random.sample(items, min(args.n_pairs, len(items)))

    out_dir = Path(f"data/precomputed/{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    new_manifest = []
    errors = 0

    for i, item in enumerate(tqdm(items, desc=f"Precomputing {args.split}")):
        try:
            noisy, clean = process_item(item, n_samples, sr)
            noisy_path = out_dir / f"{i:06d}_noisy.npy"
            clean_path = out_dir / f"{i:06d}_clean.npy"
            np.save(noisy_path, noisy)
            np.save(clean_path, clean)
            new_manifest.append({
                "noisy": str(noisy_path),
                "clean": str(clean_path),
            })
        except Exception as e:
            errors += 1
            if errors < 5:
                tqdm.write(f"  [WARN] item {i}: {e}")

    out_manifest = f"data/manifests/{args.split}_precomputed.json"
    with open(out_manifest, "w") as f:
        json.dump(new_manifest, f, indent=2)

    print(f"\n✓ {args.split}: {len(new_manifest)} pairs → {out_dir}")
    print(f"  Manifest: {out_manifest}")
    if errors:
        print(f"  Warnings: {errors} items skipped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",    default="train", choices=["train","val","test"])
    parser.add_argument("--n_pairs",  type=int, default=20000)
    parser.add_argument("--duration", type=float, default=2.0)
    args = parser.parse_args()
    main(args)
