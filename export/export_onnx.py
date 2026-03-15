"""
export/export_onnx.py
---------------------
Export the trained CRN model to ONNX format for C++ inference.

Export strategy:
    - CRN only (beamforming stays in Python/C++ preprocessing)
    - Dynamic axes: batch + time sequence
    - Opset 17 (LSTM support is stable)
    - fp32 (can quantize to fp16/int8 post-export)

Usage:
    python export/export_onnx.py \
        --checkpoint checkpoints/best.pt \
        --output model.onnx \
        --config configs/crn_base.yaml

Then verify:
    python export/export_onnx.py --verify --onnx model.onnx
"""

import argparse
import sys
import time
import yaml
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.crn import CRN


class CRNWrapper(nn.Module):
    """
    Thin wrapper around CRN that takes (real, imag) → (enhanced_real, enhanced_imag).
    Stacks inputs to (B, 2, F, T) internally. Easier for ONNX export.
    """

    def __init__(self, crn: CRN):
        super().__init__()
        self.crn = crn

    def forward(self, noisy_real: torch.Tensor, noisy_imag: torch.Tensor):
        """
        Args:
            noisy_real: (B, F, T)
            noisy_imag: (B, F, T)
        Returns:
            enhanced_real: (B, F, T)
            enhanced_imag: (B, F, T)
        """
        return self.crn.enhance(noisy_real, noisy_imag)


def export_onnx(
    model_path: str,
    output_path: str,
    config_path: str,
    opset: int = 17,
) -> None:
    # ── Load model ──────────────────────────────────────────────────────
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    m_cfg = cfg["model"]
    audio_cfg = cfg["audio"]
    n_bins = audio_cfg["n_fft"] // 2 + 1

    model = CRN(
        in_channels=m_cfg["in_channels"],
        encoder_channels=m_cfg["encoder_channels"],
        encoder_kernel=tuple(m_cfg["encoder_kernel"]),
        lstm_hidden=m_cfg["lstm_hidden"],
        lstm_layers=m_cfg["lstm_layers"],
        n_freq_bins=n_bins,
    )

    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    wrapper = CRNWrapper(model)
    wrapper.eval()

    # ── Dummy input ─────────────────────────────────────────────────────
    # Shape: (1, n_bins, T_frames) — single utterance, 50 frames
    T = 50
    dummy_real = torch.randn(1, n_bins, T)
    dummy_imag = torch.randn(1, n_bins, T)

    # ── Export ──────────────────────────────────────────────────────────
    print(f"Exporting to ONNX opset {opset}...")
    torch.onnx.export(
        wrapper,
        (dummy_real, dummy_imag),
        output_path,
        input_names=["noisy_real", "noisy_imag"],
        output_names=["enhanced_real", "enhanced_imag"],
        dynamic_axes={
            "noisy_real":     {0: "batch", 2: "n_frames"},
            "noisy_imag":     {0: "batch", 2: "n_frames"},
            "enhanced_real":  {0: "batch", 2: "n_frames"},
            "enhanced_imag":  {0: "batch", 2: "n_frames"},
        },
        opset_version=opset,
        do_constant_folding=True,
        verbose=False,
    )
    print(f"✓ Exported → {output_path}")

    # ── Validate ONNX model structure ───────────────────────────────────
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model structure check passed")

    # ── Numerical parity test ────────────────────────────────────────────
    print("\nRunning numerical parity test (PyTorch vs ONNX Runtime)...")
    with torch.no_grad():
        pt_real, pt_imag = wrapper(dummy_real, dummy_imag)

    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(
        ["enhanced_real", "enhanced_imag"],
        {"noisy_real": dummy_real.numpy(), "noisy_imag": dummy_imag.numpy()},
    )

    max_diff_real = np.abs(pt_real.numpy() - ort_out[0]).max()
    max_diff_imag = np.abs(pt_imag.numpy() - ort_out[1]).max()
    print(f"  Max abs diff (real): {max_diff_real:.2e}")
    print(f"  Max abs diff (imag): {max_diff_imag:.2e}")
    assert max_diff_real < 1e-4, f"Real output mismatch: {max_diff_real}"
    assert max_diff_imag < 1e-4, f"Imag output mismatch: {max_diff_imag}"
    print("✓ Numerical parity OK (max diff < 1e-4)")

    # ── Latency benchmark ───────────────────────────────────────────────
    print("\nONNX Runtime latency benchmark (T=32 frames, batch=1)...")
    T_rt = 32   # ~320ms audio at 10ms hop → typical streaming chunk
    r_in = np.random.randn(1, n_bins, T_rt).astype(np.float32)
    i_in = np.random.randn(1, n_bins, T_rt).astype(np.float32)

    # Warmup
    for _ in range(10):
        sess.run(None, {"noisy_real": r_in, "noisy_imag": i_in})

    # Timed
    n_runs = 200
    t0 = time.perf_counter()
    for _ in range(n_runs):
        sess.run(None, {"noisy_real": r_in, "noisy_imag": i_in})
    t_ms = (time.perf_counter() - t0) / n_runs * 1000

    print(f"  Mean inference time: {t_ms:.2f}ms  (target: <18ms)")
    if t_ms < 18:
        print(f"  ✓ Sub-18ms achieved ({t_ms:.2f}ms)")
    else:
        print(f"  ⚠ Exceeds 18ms target — consider quantization or pruning")

    # ── Model size ──────────────────────────────────────────────────────
    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"\nModel size: {size_mb:.1f}MB")
    print(f"\n✓ Export complete: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output",     default="model.onnx")
    parser.add_argument("--config",     default="configs/crn_base.yaml")
    parser.add_argument("--opset",      type=int, default=17)
    args = parser.parse_args()

    export_onnx(
        model_path=args.checkpoint,
        output_path=args.output,
        config_path=args.config,
        opset=args.opset,
    )
