# Real-Time Speech Enhancement & Beamforming Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)](https://pytorch.org)
[![ONNX](https://img.shields.io/badge/ONNX-opset%2018-green)](https://onnx.ai)
[![C++](https://img.shields.io/badge/C%2B%2B-17-lightgrey)](https://isocpp.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A dual-stage speech enhancement pipeline combining **MVDR beamforming** with a **CRN-based complex ratio mask network**, exported to ONNX and served via a C++ ONNXRuntime inference engine achieving **0.5ms end-to-end latency**.

---

## Results

| Metric | Noisy Input | Enhanced | Δ |
|---|---|---|---|
| **PESQ (WB)** | 1.72 | 1.96 | **+0.24** |
| **SI-SNR** | 8.0 dB | 11.2 dB | **+3.2 dB** |
| **STOI** | 0.795 | 0.823 | **+3.5%** |
| **C++ Latency** | — | — | **0.5ms** |
| **ONNX Latency** | — | — | **1.2ms** |

Evaluated on 500 held-out noisy-clean pairs (LibriSpeech clean + MS-SNSD noise, SNR range −5 to +20 dB).

---

## Architecture

```
Multi-channel Audio (4-mic ULA, 5cm spacing)
        │
        ▼
┌─────────────────────────┐
│    MVDR Beamformer      │  Spatial filtering via RTF/GEV steering
│    beamforming/mvdr.py  │  Noise covariance estimation + diagonal loading
└──────────┬──────────────┘
           │  Single-channel beamformed signal
           ▼
┌─────────────────────────┐
│  STFT  (n_fft=256)      │  256-pt Hann window, 128-sample hop
└──────────┬──────────────┘
           │  (B, 2, 129, T)  real + imag
           ▼
┌─────────────────────────┐
│  CRN Spectral Masking   │  Frequency-strided encoder (129→4 bins)
│  models/crn.py          │  TemporalMLP context module
│                         │  Interpolation decoder with skip connections
│                         │  Complex Ratio Mask (CRM) output
└──────────┬──────────────┘
           │  Enhanced complex spectrogram
           ▼
┌─────────────────────────┐
│  iSTFT + Overlap-Add    │  Waveform reconstruction
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  ONNX Runtime (C++)     │  0.5ms inference, ring buffer streaming
│  cpp_inference/         │  ONNXRuntime 1.17, single-threaded for
│                         │  deterministic latency
└─────────────────────────┘
```

### CRN Design

The network uses a **frequency-strided encoder** that progressively halves the frequency dimension (129 → 65 → 33 → 17 → 9 → 4 bins) via strided Conv2d, reducing computational cost ~26× compared to a no-stride baseline. A frame-wise **TemporalMLP** with residual connection provides temporal context without the sequential bottleneck of RNNs — critical for CPU/inference efficiency. The decoder uses bilinear interpolation upsampling with U-Net skip connections.

---

## Project Structure

```
speech_enhancement/
├── configs/
│   ├── crn_base.yaml          # Full model config
│   └── crn_small.yaml         # Lightweight config (trained)
├── data/
│   ├── dataset.py             # On-the-fly noisy-clean mixing
│   ├── fast_dataset.py        # Precomputed .npy loader (training)
│   ├── precompute_dataset.py  # Pre-mix pairs to disk
│   ├── prepare_dataset.py     # Build JSON manifests
│   └── simulate_rir.py        # Image-source RIR simulation
├── beamforming/
│   └── mvdr.py                # MVDR beamformer + ULA geometry
├── models/
│   └── crn.py                 # CRN architecture
├── training/
│   ├── train.py               # Training loop
│   └── losses.py              # SI-SNR + spectral losses
├── evaluation/
│   ├── evaluate.py            # Evaluation script (PESQ/STOI/SI-SNR)
│   └── metrics.py             # Per-sample metric functions
├── export/
│   └── export_onnx.py         # ONNX export + parity + latency test
├── cpp_inference/
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── speech_enhancer.h  # Streaming engine interface
│   │   └── wav_io.h           # Header-only WAV I/O
│   └── src/
│       └── speech_enhancer.cpp  # Ring buffer + overlap-add iSTFT
├── matlab/
│   ├── eval_rt60_robustness.m   # PESQ/STOI vs RT60 sweep
│   └── convert_rirs_to_mat.m   # Convert .npy RIRs to .mat
├── utils/
│   └── audio.py               # STFT, audio I/O, SI-SNR
└── tests/
    └── test_pipeline.py       # Smoke tests (9 tests, no data needed)
```

---

## Quickstart

### 1. Install dependencies

```bash
conda create -n speech-enhancement python=3.11
conda activate speech-enhancement
pip install -r requirements.txt
```

### 2. Prepare data

Download [LibriSpeech train-clean-100](https://www.openslr.org/12/) and clone [MS-SNSD](https://github.com/microsoft/MS-SNSD) for noise:

```bash
git clone https://github.com/microsoft/MS-SNSD.git

python data/simulate_rir.py --output_dir data/rirs --n_rooms 500

python data/prepare_dataset.py \
    --librispeech_path ./LibriSpeech \
    --demand_path ./MS-SNSD/noise_train \
    --rir_dir data/rirs

python data/precompute_dataset.py --split train --n_pairs 20000 --duration 2.0
python data/precompute_dataset.py --split val   --n_pairs 2000  --duration 2.0
python data/precompute_dataset.py --split test  --n_pairs 1000  --duration 2.0
```

### 3. Train

```bash
python training/train.py --config configs/crn_small.yaml
```

Resume from checkpoint:
```bash
python training/train.py --config configs/crn_small.yaml --resume checkpoints/last.pt
```

### 4. Evaluate

```bash
python -m evaluation.evaluate \
    --checkpoint checkpoints/best.pt \
    --config configs/crn_small.yaml \
    --n_samples 500
```

### 5. Export to ONNX

```bash
python export/export_onnx.py \
    --checkpoint checkpoints/best.pt \
    --config configs/crn_small.yaml \
    --output model.onnx
```

### 6. Build and run C++ inference engine

```bash
# macOS (Homebrew ONNXRuntime)
brew install cmake onnxruntime

cd cpp_inference
mkdir build && cd build
cmake .. -DONNXRUNTIME_DIR=/opt/homebrew \
         -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CXX_FLAGS="-I/opt/homebrew/include/onnxruntime"
make -j4
cd ../..

./cpp_inference/build/speech_enhancer model.onnx input.wav output.wav
```

Expected output:
```
[SpeechEnhancer] Loaded model: model.onnx
[SpeechEnhancer] STFT: n_fft=256 hop=128 bins=129
[SpeechEnhancer] Processed → output.wav
[SpeechEnhancer] Mean latency: 0.50ms
```

---

## Run Tests

No data required — uses synthetic tensors:

```bash
python tests/test_pipeline.py
```

```
============================================================
Speech Enhancement Pipeline — Smoke Tests
============================================================
[STFT round-trip]       ✓
[ULA positions]         ✓
[MVDR beamformer]       ✓
[CRN forward]           ✓
[CRN enhance]           ✓
[SI-SNR loss]           ✓
[Combined loss]         ✓
[mix_signals SNR]       ✓
[End-to-end pipeline]   ✓

Results: 9/9 passed ✓
```

---

## Training Details

| Setting | Value |
|---|---|
| Clean speech | LibriSpeech train-clean-100 |
| Noise | MS-SNSD (60+ environments) |
| SNR range | −5 to +20 dB |
| RIR augmentation | 500 rooms, RT60 0.2–0.8s, 80% probability |
| Training pairs | 20,000 |
| Epochs | 30 |
| Optimizer | Adam, lr=5e-4, cosine decay |
| Loss | 0.8×SI-SNR + 0.1×LogMag + 0.1×ComplexMSE |
| Model parameters | 280,050 |

---

## Room Simulation

RIRs are simulated using the **image-source method** via [pyroomacoustics](https://pyroomacoustics.readthedocs.io/), sweeping RT60 from 0.2s to 0.8s across 500 randomly sampled shoebox rooms. The microphone array is a 4-element linear ULA with 5cm inter-element spacing, centered in each room.

```bash
python data/simulate_rir.py --output_dir data/rirs --n_rooms 500 --n_mics 4
```

MATLAB scripts for RT60 robustness evaluation are in `matlab/`.

---

## Dependencies

| Package | Purpose |
|---|---|
| PyTorch ≥ 2.1 | Model training |
| onnx + onnxruntime | Export and inference |
| pyroomacoustics | RIR simulation |
| pesq + pystoi | Evaluation metrics |
| soundfile + librosa | Audio I/O |
| ONNXRuntime C++ | Low-latency inference engine |
