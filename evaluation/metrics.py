"""
evaluation/metrics.py  &  evaluate.py
--------------------------------------
Compute PESQ (WB), STOI, and SI-SNR on a test set.

PESQ  — Perceptual Evaluation of Speech Quality  (ITU-T P.862.2 wideband)
         Scale: 1.0 (bad) → 4.5 (excellent)
         Baseline noisy: ~1.92  |  Target enhanced: ~2.87

STOI  — Short-Time Objective Intelligibility
         Scale: 0 → 1  (higher = more intelligible)

SI-SNR — Scale-Invariant Signal-to-Noise Ratio (dB)
"""

import numpy as np
from typing import List, Dict
from pesq import pesq as compute_pesq
from pystoi import stoi as compute_stoi
from utils.audio import si_snr


def evaluate_sample(
    enhanced: np.ndarray,
    reference: np.ndarray,
    sample_rate: int = 16000,
) -> Dict[str, float]:
    """
    Compute PESQ, STOI, SI-SNR for a single (enhanced, reference) pair.
    Both arrays should be float32, same length.
    """
    # Align lengths
    min_len = min(len(enhanced), len(reference))
    enhanced  = enhanced[:min_len]
    reference = reference[:min_len]

    # Peak normalize to avoid clipping artefacts in PESQ
    enhanced  = enhanced  / (np.abs(enhanced).max()  + 1e-8)
    reference = reference / (np.abs(reference).max() + 1e-8)

    results = {}

    try:
        if np.isnan(enhanced).any() or np.isnan(reference).any():
            results["pesq"] = float("nan")
        elif np.abs(enhanced).max() < 1e-8:   # silent output
            results["pesq"] = float("nan")
        else:
            # Clip to valid range for pesq
            ref_c = np.clip(reference, -1.0, 1.0)
            enh_c = np.clip(enhanced,  -1.0, 1.0)
            results["pesq"] = compute_pesq(sample_rate, ref_c, enh_c, "wb")
    except Exception as e:
        results["pesq"] = float("nan")

    try:
        results["stoi"] = compute_stoi(reference, enhanced, sample_rate, extended=False)
    except Exception:
        results["stoi"] = float("nan")

    try:
        results["sisnr"] = si_snr(reference, enhanced)
    except Exception:
        results["sisnr"] = float("nan")

    return results


def evaluate_batch(
    enhanced_batch: np.ndarray,   # (B, T)
    reference_batch: np.ndarray,  # (B, T)
    sample_rate: int = 16000,
) -> Dict[str, List[float]]:
    """Evaluate a batch, return lists of per-sample metrics."""
    pesq_list, stoi_list, sisnr_list = [], [], []

    for i in range(enhanced_batch.shape[0]):
        m = evaluate_sample(enhanced_batch[i], reference_batch[i], sample_rate)
        pesq_list.append(m["pesq"])
        stoi_list.append(m["stoi"])
        sisnr_list.append(m["sisnr"])

    return {"pesq": pesq_list, "stoi": stoi_list, "sisnr": sisnr_list}


# ── Standalone evaluation script ──────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json
    import sys
    import torch
    import yaml
    from pathlib import Path
    from tqdm import tqdm

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from models.crn import CRN
    from beamforming.mvdr import MVDRBeamformer
    from data.dataset import build_dataloader
    from utils.audio import STFT

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/crn_base.yaml")
    parser.add_argument("--test_manifest", default=None)
    parser.add_argument("--output_json", default="evaluation/results.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_cfg = cfg["audio"]

    stft = STFT(
        n_fft=audio_cfg["n_fft"],
        hop_length=audio_cfg["hop_length"],
        win_length=audio_cfg["win_length"],
    ).to(device)

    m_cfg = cfg["model"]
    model = CRN(
        in_channels=m_cfg["in_channels"],
        encoder_channels=m_cfg["encoder_channels"],
        encoder_kernel=tuple(m_cfg["encoder_kernel"]),
        lstm_hidden=m_cfg["lstm_hidden"],
        lstm_layers=m_cfg["lstm_layers"],
        n_freq_bins=audio_cfg["n_fft"] // 2 + 1,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    beamformer = MVDRBeamformer(n_mics=audio_cfg["n_mics"], n_fft=audio_cfg["n_fft"]).to(device)

    manifest_path = args.test_manifest or cfg["data"]["test_manifest"]
    test_loader = build_dataloader(
        manifest_path,
        batch_size=8,
        num_workers=2,
        shuffle=False,
        sample_rate=audio_cfg["sample_rate"],
        duration=cfg["data"]["max_duration"],
        n_mics=audio_cfg["n_mics"],
        rir_prob=0.0,
        augment=False,
    )

    all_pesq, all_stoi, all_sisnr = [], [], []
    noisy_pesq, noisy_stoi, noisy_sisnr = [], [], []

    with torch.no_grad():
        for noisy_mc, clean, _ in tqdm(test_loader, desc="Evaluating"):
            noisy_mc = noisy_mc.to(device)
            clean    = clean.to(device)
            B, M, T_wav = noisy_mc.shape

            # Noisy baseline (ref mic)
            noisy_ref = noisy_mc[:, 0, :]
            nm = evaluate_batch(noisy_ref.cpu().numpy(), clean.cpu().numpy(), audio_cfg["sample_rate"])
            noisy_pesq.extend(nm["pesq"])
            noisy_stoi.extend(nm["stoi"])
            noisy_sisnr.extend(nm["sisnr"])

            # Enhanced
            noisy_flat = noisy_mc.reshape(B * M, T_wav)
            real_flat, imag_flat = stft(noisy_flat)
            _, F, T_spec = real_flat.shape
            mc_spec = torch.complex(
                real_flat.reshape(B, M, F, T_spec),
                imag_flat.reshape(B, M, F, T_spec),
            )
            bf_spec = beamformer(mc_spec)
            enh_real, enh_imag = model.enhance(bf_spec.real, bf_spec.imag)
            enh_wav = stft.inverse(enh_real, enh_imag, length=T_wav)

            em = evaluate_batch(enh_wav.cpu().numpy(), clean.cpu().numpy(), audio_cfg["sample_rate"])
            all_pesq.extend(em["pesq"])
            all_stoi.extend(em["stoi"])
            all_sisnr.extend(em["sisnr"])

    def nanmean(lst):
        arr = np.array(lst, dtype=float)
        return float(np.nanmean(arr))

    results = {
        "noisy": {
            "pesq":  nanmean(noisy_pesq),
            "stoi":  nanmean(noisy_stoi),
            "sisnr": nanmean(noisy_sisnr),
        },
        "enhanced": {
            "pesq":  nanmean(all_pesq),
            "stoi":  nanmean(all_stoi),
            "sisnr": nanmean(all_sisnr),
        },
        "delta": {
            "pesq":  nanmean(all_pesq)  - nanmean(noisy_pesq),
            "stoi":  (nanmean(all_stoi) - nanmean(noisy_stoi)) / nanmean(noisy_stoi) * 100,
            "sisnr": nanmean(all_sisnr) - nanmean(noisy_sisnr),
        },
        "n_samples": len(all_pesq),
    }

    print("\n── Evaluation Results ─────────────────────────────")
    print(f"  Noisy   PESQ={results['noisy']['pesq']:.3f}  STOI={results['noisy']['stoi']:.3f}  SI-SNR={results['noisy']['sisnr']:.1f}dB")
    print(f"  Enhanced PESQ={results['enhanced']['pesq']:.3f}  STOI={results['enhanced']['stoi']:.3f}  SI-SNR={results['enhanced']['sisnr']:.1f}dB")
    print(f"  Δ PESQ={results['delta']['pesq']:+.3f}  Δ STOI={results['delta']['stoi']:+.1f}%  Δ SI-SNR={results['delta']['sisnr']:+.1f}dB")

    Path(args.output_json).parent.mkdir(exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved → {args.output_json}")