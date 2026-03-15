"""
data/prepare_dataset.py
-----------------------
Build train/val/test manifests from LibriSpeech (clean) + DEMAND (noise).

Downloads LibriSpeech train-clean-100 and test-clean if not present.
DEMAND corpus must be downloaded manually from:
    https://zenodo.org/record/1227121

Usage:
    python data/prepare_dataset.py \
        --librispeech_path /data/LibriSpeech \
        --demand_path /data/DEMAND \
        --rir_dir data/rirs \
        --output_dir data/manifests \
        --n_train 100000 --n_val 10000 --n_test 10000
"""

import argparse
import json
import random
import os
import sys
from pathlib import Path
from typing import List, Dict
import torchaudio

sys.path.insert(0, str(Path(__file__).parent.parent))


DEMAND_ENVIRONMENTS = [
    "DKITCHEN", "DLIVING", "DWASHING",   # domestic
    "NFIELD", "NPARK", "NRIVER",          # nature
    "OHALLWAY", "OMEETING", "OOFFICE",    # office
    "PCAFETER", "PRESTO", "PSTATION",     # public
    "SCAFE", "SPSQUARE", "STRAFFIC",      # street
    "TBUS", "TCAR", "TMETRO",             # transport
]


def find_audio_files(root: str, exts: tuple = (".flac", ".wav")) -> List[str]:
    """Recursively find all audio files under root."""
    files = []
    for path in Path(root).rglob("*"):
        if path.suffix.lower() in exts:
            files.append(str(path))
    return sorted(files)


def get_duration(path: str) -> float:
    """Return audio duration in seconds without loading full file."""
    info = torchaudio.info(path)
    return info.num_frames / info.sample_rate


def build_manifest(
    clean_files: List[str],
    noise_files: List[str],
    rir_files: List[str],
    n_pairs: int,
    snr_range: tuple = (-5, 20),
    rir_prob: float = 0.8,
    seed: int = 42,
) -> List[Dict]:
    """
    Build a list of (clean, noise, snr, rir) dicts.
    RIR assignment is random with probability rir_prob.
    """
    rng = random.Random(seed)
    manifest = []

    for i in range(n_pairs):
        clean = rng.choice(clean_files)
        noise = rng.choice(noise_files)
        snr   = rng.uniform(*snr_range)

        item = {"clean": clean, "noise": noise, "snr_db": round(snr, 2)}

        if rir_files and rng.random() < rir_prob:
            item["rir"] = rng.choice(rir_files)

        manifest.append(item)

    return manifest


def main(args):
    rng = random.Random(args.seed)

    print("Scanning LibriSpeech clean files...")
    clean_files = find_audio_files(args.librispeech_path)
    print(f"  Found {len(clean_files)} clean utterances")

    print("Scanning DEMAND noise files...")
    noise_files = find_audio_files(args.demand_path)
    print(f"  Found {len(noise_files)} noise files")

    rir_files = []
    if args.rir_dir and Path(args.rir_dir).exists():
        rir_files = sorted(str(p) for p in Path(args.rir_dir).glob("room_*.npy"))
        print(f"  Found {len(rir_files)} RIR files")

    if not clean_files:
        raise ValueError(f"No clean audio found at {args.librispeech_path}")
    if not noise_files:
        raise ValueError(f"No noise audio found at {args.demand_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split clean files: 90% train, 5% val, 5% test
    rng.shuffle(clean_files)
    n = len(clean_files)
    train_clean = clean_files[:int(n * 0.9)]
    val_clean   = clean_files[int(n * 0.9): int(n * 0.95)]
    test_clean  = clean_files[int(n * 0.95):]

    splits = [
        ("train", train_clean, args.n_train, 42),
        ("val",   val_clean,   args.n_val,   43),
        ("test",  test_clean,  args.n_test,  44),
    ]

    for split_name, clean_split, n_pairs, seed in splits:
        print(f"\nBuilding {split_name} manifest ({n_pairs} pairs)...")
        manifest = build_manifest(
            clean_files=clean_split,
            noise_files=noise_files,
            rir_files=rir_files,
            n_pairs=n_pairs,
            snr_range=(args.snr_min, args.snr_max),
            rir_prob=args.rir_prob,
            seed=seed,
        )

        out_path = output_dir / f"{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  ✓ Saved {out_path} ({len(manifest)} items)")

    print("\n✓ Dataset preparation complete.")
    print(f"  Total pairs: train={args.n_train} | val={args.n_val} | test={args.n_test}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech_path", required=True)
    parser.add_argument("--demand_path",      required=True)
    parser.add_argument("--rir_dir",          default="data/rirs")
    parser.add_argument("--output_dir",       default="data/manifests")
    parser.add_argument("--n_train", type=int, default=100000)
    parser.add_argument("--n_val",   type=int, default=10000)
    parser.add_argument("--n_test",  type=int, default=10000)
    parser.add_argument("--snr_min", type=float, default=-5.0)
    parser.add_argument("--snr_max", type=float, default=20.0)
    parser.add_argument("--rir_prob",type=float, default=0.8)
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()
    main(args)
