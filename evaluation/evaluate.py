"""
evaluation/evaluate.py
----------------------
Evaluate trained CRN on precomputed test pairs.
Matches the training pipeline exactly (single-channel, no MVDR).

Usage:
    python -m evaluation.evaluate \
        --checkpoint checkpoints/best.pt \
        --config configs/crn_small.yaml
"""

import argparse
import json
import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.crn import CRN
from utils.audio import STFT, si_snr
from pesq import pesq as compute_pesq
from pystoi import stoi as compute_stoi


def evaluate(args):
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")
    audio_cfg = cfg["audio"]
    m_cfg     = cfg["model"]

    stft = STFT(
        n_fft=audio_cfg["n_fft"],
        hop_length=audio_cfg["hop_length"],
        win_length=audio_cfg["win_length"],
    )

    model = CRN(
        in_channels=m_cfg["in_channels"],
        encoder_channels=m_cfg["encoder_channels"],
        encoder_kernel=tuple(m_cfg["encoder_kernel"]),
        lstm_hidden=m_cfg["lstm_hidden"],
        lstm_layers=m_cfg["lstm_layers"],
        n_freq_bins=audio_cfg["n_fft"] // 2 + 1,
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch','?')})")

    # Use precomputed test manifest if available
    test_manifest = cfg["data"]["test_manifest"].replace(".json", "_precomputed.json")
    if not Path(test_manifest).exists():
        # Fall back to building test pairs on the fly from val set
        test_manifest = cfg["data"]["val_manifest"].replace(".json", "_precomputed.json")
    print(f"Test manifest: {test_manifest}")

    with open(test_manifest) as f:
        items = json.load(f)

    sr = audio_cfg["sample_rate"]
    results_noisy    = {"pesq": [], "stoi": [], "sisnr": []}
    results_enhanced = {"pesq": [], "stoi": [], "sisnr": []}

    with torch.no_grad():
        for item in tqdm(items[:args.n_samples], desc="Evaluating"):
            noisy = np.load(item["noisy"])   # (T,)
            clean = np.load(item["clean"])   # (T,)

            # Normalize
            noisy = noisy / (np.abs(noisy).max() + 1e-8)
            clean = clean / (np.abs(clean).max() + 1e-8)

            # ── Noisy baseline metrics ────────────────────────────────
            try:
                results_noisy["pesq"].append(compute_pesq(sr, clean, noisy, "wb"))
            except Exception:
                results_noisy["pesq"].append(float("nan"))
            try:
                results_noisy["stoi"].append(compute_stoi(clean, noisy, sr, extended=False))
            except Exception:
                results_noisy["stoi"].append(float("nan"))
            try:
                results_noisy["sisnr"].append(si_snr(clean, noisy))
            except Exception:
                results_noisy["sisnr"].append(float("nan"))

            # ── Enhanced ──────────────────────────────────────────────
            noisy_t = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0)  # (1, T)
            r, i    = stft(noisy_t)
            er, ei  = model.enhance(r, i)
            er = torch.nan_to_num(er, nan=0.0, posinf=0.0, neginf=0.0)
            ei = torch.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
            enh_wav = stft.inverse(er, ei, length=noisy_t.shape[-1])
            enh = enh_wav.squeeze(0).numpy()
            enh = enh / (np.abs(enh).max() + 1e-8)

            # Align lengths
            L = min(len(clean), len(enh))
            c, e = clean[:L], enh[:L]

            try:
                results_enhanced["pesq"].append(compute_pesq(sr, c, e, "wb"))
            except Exception:
                results_enhanced["pesq"].append(float("nan"))
            try:
                results_enhanced["stoi"].append(compute_stoi(c, e, sr, extended=False))
            except Exception:
                results_enhanced["stoi"].append(float("nan"))
            try:
                results_enhanced["sisnr"].append(si_snr(c, e))
            except Exception:
                results_enhanced["sisnr"].append(float("nan"))

    def m(lst): return float(np.nanmean(lst))

    print("\n── Evaluation Results ─────────────────────────────────")
    print(f"  Samples evaluated: {len(items[:args.n_samples])}")
    print(f"  Noisy    PESQ={m(results_noisy['pesq']):.3f}  "
          f"STOI={m(results_noisy['stoi']):.3f}  "
          f"SI-SNR={m(results_noisy['sisnr']):.1f}dB")
    print(f"  Enhanced PESQ={m(results_enhanced['pesq']):.3f}  "
          f"STOI={m(results_enhanced['stoi']):.3f}  "
          f"SI-SNR={m(results_enhanced['sisnr']):.1f}dB")
    print(f"  Δ PESQ={m(results_enhanced['pesq'])-m(results_noisy['pesq']):+.3f}  "
          f"Δ STOI={( m(results_enhanced['stoi'])-m(results_noisy['stoi']) ) / m(results_noisy['stoi'])*100:+.1f}%  "
          f"Δ SI-SNR={m(results_enhanced['sisnr'])-m(results_noisy['sisnr']):+.1f}dB")

    out = {
        "noisy":    {k: m(v) for k,v in results_noisy.items()},
        "enhanced": {k: m(v) for k,v in results_enhanced.items()},
        "n_samples": len(items[:args.n_samples]),
    }
    Path(args.output).parent.mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✓ Results saved → {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     default="configs/crn_small.yaml")
    parser.add_argument("--n_samples",  type=int, default=500)
    parser.add_argument("--output",     default="evaluation/results.json")
    args = parser.parse_args()
    evaluate(args)