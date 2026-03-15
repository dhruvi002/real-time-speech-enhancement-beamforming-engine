"""
inference.py
------------
CLI for running the full enhancement pipeline on a single WAV file.

Runs: MVDR beamforming (if multi-channel) → CRN masking → save output.

Usage:
    # Single-channel (CRN only)
    python inference.py --model model.onnx --input noisy.wav --output enhanced.wav

    # Multi-channel (MVDR + CRN)
    python inference.py --model model.onnx --input noisy_4ch.wav --output enhanced.wav --n_mics 4

    # Benchmark mode (print latency)
    python inference.py --model model.onnx --input noisy.wav --output enhanced.wav --benchmark
"""

import argparse
import time
import sys
import numpy as np
import soundfile as sf
import onnxruntime as ort
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.audio import STFT, save_audio


def enhance_onnx(
    noisy: np.ndarray,         # (T,) or (n_mics, T)
    sess: ort.InferenceSession,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 320,
    sample_rate: int = 16000,
    n_mics: int = 1,
    benchmark: bool = False,
) -> np.ndarray:
    """
    Full inference pipeline:
        1. (optional) MVDR beamforming on multi-channel input
        2. STFT
        3. CRN masking via ONNX Runtime
        4. iSTFT

    Returns: enhanced waveform (T,) float32
    """
    import torch
    from utils.audio import STFT as TorchSTFT

    # ── Beamforming ─────────────────────────────────────────────────
    if noisy.ndim == 2 and noisy.shape[0] == n_mics and n_mics > 1:
        from beamforming.mvdr import MVDRBeamformer
        stft_proc = TorchSTFT(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        bf = MVDRBeamformer(n_mics=n_mics, n_fft=n_fft)

        noisy_t = torch.tensor(noisy, dtype=torch.float32)  # (M, T)
        mc_real_list, mc_imag_list = [], []
        for m in range(n_mics):
            r, i = stft_proc(noisy_t[m])
            mc_real_list.append(r)
            mc_imag_list.append(i)

        mc_real = torch.stack(mc_real_list, dim=0).unsqueeze(0)   # (1, M, F, T)
        mc_imag = torch.stack(mc_imag_list, dim=0).unsqueeze(0)
        mc_spec = torch.complex(mc_real, mc_imag)
        bf_spec = bf(mc_spec)   # (1, F, T)

        noisy_real_np = bf_spec[0].real.numpy()   # (F, T)
        noisy_imag_np = bf_spec[0].imag.numpy()

    else:
        # Single channel: STFT directly
        if noisy.ndim == 2:
            noisy = noisy[0]   # use ref mic
        stft_proc = TorchSTFT(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        noisy_t = torch.tensor(noisy, dtype=torch.float32)
        r, i = stft_proc(noisy_t)
        noisy_real_np = r.numpy()   # (F, T)
        noisy_imag_np = i.numpy()

    # ── CRN inference ────────────────────────────────────────────────
    # Input shape: (1, F, T)
    noisy_real_in = noisy_real_np[np.newaxis, ...].astype(np.float32)
    noisy_imag_in = noisy_imag_np[np.newaxis, ...].astype(np.float32)

    timings = []
    for _ in range(1 if not benchmark else 20):
        t0 = time.perf_counter()
        ort_out = sess.run(
            ["enhanced_real", "enhanced_imag"],
            {"noisy_real": noisy_real_in, "noisy_imag": noisy_imag_in},
        )
        timings.append((time.perf_counter() - t0) * 1000)

    if benchmark:
        print(f"  Inference latency: {np.mean(timings[5:]):.2f}ms (mean of 15 runs)")

    enh_real = torch.tensor(ort_out[0][0])   # (F, T)
    enh_imag = torch.tensor(ort_out[1][0])

    # ── iSTFT ────────────────────────────────────────────────────────
    T_orig = noisy.shape[-1] if noisy.ndim == 1 else noisy.shape[-1]
    enhanced = stft_proc.inverse(enh_real, enh_imag, length=T_orig)
    return enhanced.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     required=True, help="Path to model.onnx")
    parser.add_argument("--input",     required=True, help="Input WAV file")
    parser.add_argument("--output",    required=True, help="Output WAV file")
    parser.add_argument("--n_mics",    type=int, default=1, help="Number of input channels")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    # Load audio
    wav, sr = sf.read(args.input, dtype="float32", always_2d=True)
    wav = wav.T   # (channels, T)
    if args.n_mics == 1:
        wav = wav[0]   # (T,)

    # Load ONNX session
    sess = ort.InferenceSession(
        args.model,
        providers=["CPUExecutionProvider"]
    )

    print(f"Input:  {args.input}  ({sr}Hz, {wav.shape})")

    t_start = time.perf_counter()
    enhanced = enhance_onnx(
        wav, sess,
        n_mics=args.n_mics,
        benchmark=args.benchmark,
    )
    total_ms = (time.perf_counter() - t_start) * 1000

    save_audio(args.output, enhanced, sr=sr)

    audio_dur_ms = len(enhanced) / sr * 1000
    rtf = total_ms / audio_dur_ms

    print(f"Output: {args.output}")
    print(f"Total processing: {total_ms:.1f}ms for {audio_dur_ms:.0f}ms audio (RTF={rtf:.3f})")


if __name__ == "__main__":
    main()
