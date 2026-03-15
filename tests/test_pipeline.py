"""
tests/test_pipeline.py
-----------------------
Smoke tests for the full pipeline — no external data required.
Verifies shapes, numerical correctness, and that all components
are importable and run without error.

Run with:
    python -m pytest tests/ -v
    # or directly:
    python tests/test_pipeline.py
"""

import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Utility ───────────────────────────────────────────────────────────────

def rand_audio(duration=1.0, sr=16000, n_channels=1):
    T = int(duration * sr)
    if n_channels == 1:
        return torch.randn(T)
    return torch.randn(n_channels, T)


# ── Test: STFT round-trip ─────────────────────────────────────────────────

def test_stft_roundtrip():
    from utils.audio import STFT
    stft = STFT(n_fft=512, hop_length=160, win_length=320)

    wav = rand_audio(duration=2.0)
    real, imag = stft(wav)

    assert real.shape[0] == 257, f"Expected 257 bins, got {real.shape[0]}"
    assert real.shape == imag.shape

    # Round-trip: iSTFT should approximately recover input
    reconstructed = stft.inverse(real, imag, length=len(wav))
    assert reconstructed.shape == wav.shape

    # Allow for border effects — check energy correlation not exact match
    energy_orig  = (wav ** 2).mean().item()
    energy_recon = (reconstructed ** 2).mean().item()
    rel_error = abs(energy_orig - energy_recon) / (energy_orig + 1e-8)
    assert rel_error < 0.1, f"Round-trip energy error too large: {rel_error:.3f}"

    print(f"  ✓ STFT round-trip: input {wav.shape} → spec {real.shape} → recon {reconstructed.shape}")


# ── Test: CRN forward pass ────────────────────────────────────────────────

def test_crn_forward():
    from models.crn import CRN
    model = CRN(
        encoder_channels=[16, 32, 64, 128, 256],
        lstm_hidden=128,
        lstm_layers=1,
    )
    model.eval()

    B, F, T = 2, 257, 50
    x = torch.randn(B, 2, F, T)

    with torch.no_grad():
        mask_real, mask_imag = model(x)

    assert mask_real.shape == (B, F, T), f"Unexpected shape: {mask_real.shape}"
    assert mask_imag.shape == (B, F, T)

    # Tanh mask should be bounded
    assert mask_real.abs().max().item() <= 1.0 + 1e-4, "Mask real exceeds tanh range"
    assert mask_imag.abs().max().item() <= 1.0 + 1e-4, "Mask imag exceeds tanh range"

    params = model.count_parameters()
    print(f"  ✓ CRN forward: input {x.shape} → mask {mask_real.shape} | {params:,} params")


# ── Test: CRN enhance (full mask application) ─────────────────────────────

def test_crn_enhance():
    from models.crn import CRN
    model = CRN(encoder_channels=[16, 32, 64], lstm_hidden=64, lstm_layers=1)
    model.eval()

    B, F, T = 1, 257, 80
    noisy_real = torch.randn(B, F, T)
    noisy_imag = torch.randn(B, F, T)

    with torch.no_grad():
        enh_real, enh_imag = model.enhance(noisy_real, noisy_imag)

    assert enh_real.shape == noisy_real.shape
    assert not torch.isnan(enh_real).any(), "NaN in enhanced_real"
    assert not torch.isnan(enh_imag).any(), "NaN in enhanced_imag"

    print(f"  ✓ CRN enhance: ({B},{F},{T}) → ({B},{F},{T})")


# ── Test: MVDR beamformer ─────────────────────────────────────────────────

def test_mvdr_beamformer():
    from beamforming.mvdr import MVDRBeamformer

    n_mics, B, F, T = 4, 1, 257, 30
    beamformer = MVDRBeamformer(n_mics=n_mics, n_fft=512)

    # Simulate multi-channel complex STFT
    mc_real = torch.randn(B, n_mics, F, T)
    mc_imag = torch.randn(B, n_mics, F, T)
    mc_spec = torch.complex(mc_real, mc_imag)

    with torch.no_grad():
        bf_out = beamformer(mc_spec)

    assert bf_out.shape == (B, F, T), f"Unexpected shape: {bf_out.shape}"
    assert not torch.isnan(bf_out.real).any(), "NaN in beamformer output"

    print(f"  ✓ MVDR beamformer: ({B},{n_mics},{F},{T}) → ({B},{F},{T})")


# ── Test: ULA geometry ────────────────────────────────────────────────────

def test_ula_positions():
    from beamforming.mvdr import build_ula_positions
    positions = build_ula_positions(n_mics=4, spacing=0.05)

    assert positions.shape == (4, 3), f"Expected (4,3), got {positions.shape}"
    # Check spacing
    dists = np.diff(positions[:, 0])
    assert np.allclose(dists, 0.05, atol=1e-6), f"Incorrect spacing: {dists}"
    # Check centered
    assert abs(positions[:, 0].mean()) < 1e-6, "Array not centered"

    print(f"  ✓ ULA positions: {positions[:, 0].tolist()}")


# ── Test: SI-SNR loss ─────────────────────────────────────────────────────

def test_sisnr_loss():
    from training.losses import SISNRLoss
    loss_fn = SISNRLoss()

    B, T = 4, 16000
    clean = torch.randn(B, T)

    # Perfect reconstruction → SI-SNR should be very high → loss very negative
    loss_perfect = loss_fn(clean, clean)
    assert loss_perfect.item() < -30, f"Perfect SI-SNR loss should be << 0, got {loss_perfect.item():.1f}"

    # Random noise → loss should be positive (high distortion)
    noise = torch.randn(B, T)
    loss_noisy = loss_fn(noise, clean)
    assert loss_noisy.item() > -10, f"Random SI-SNR should be near 0 or positive"

    print(f"  ✓ SI-SNR loss: perfect={loss_perfect.item():.1f}, random={loss_noisy.item():.1f}")


# ── Test: Combined loss ───────────────────────────────────────────────────

def test_combined_loss():
    from training.losses import CombinedLoss
    loss_fn = CombinedLoss()

    B, F, T_wav, T_spec = 2, 257, 16000, 100
    est_wav = torch.randn(B, T_wav)
    tgt_wav = torch.randn(B, T_wav)
    est_real = torch.randn(B, F, T_spec)
    est_imag = torch.randn(B, F, T_spec)
    tgt_real = torch.randn(B, F, T_spec)
    tgt_imag = torch.randn(B, F, T_spec)

    total, metrics = loss_fn(est_wav, tgt_wav, est_real, est_imag, tgt_real, tgt_imag)

    assert not torch.isnan(total), "NaN in combined loss"
    assert "loss/sisnr" in metrics
    assert "loss/mag"   in metrics
    assert "loss/phase" in metrics

    print(f"  ✓ Combined loss: {total.item():.4f}  components={metrics}")


# ── Test: mix_signals SNR ─────────────────────────────────────────────────

def test_mix_signals():
    from utils.audio import mix_signals

    sr = 16000
    clean = np.random.randn(sr).astype(np.float32)
    noise = np.random.randn(sr).astype(np.float32)

    for target_snr in [-5, 0, 5, 10, 20]:
        noisy, scaled_noise = mix_signals(clean, noise, target_snr)
        clean_power = np.mean(clean ** 2)
        noise_power = np.mean(scaled_noise ** 2)
        achieved_snr = 10 * np.log10(clean_power / noise_power)
        assert abs(achieved_snr - target_snr) < 0.5, \
            f"SNR mismatch: target={target_snr}, got={achieved_snr:.2f}"

    print(f"  ✓ mix_signals: SNR accuracy verified for [-5, 0, 5, 10, 20] dB")


# ── Full mini-pipeline test ───────────────────────────────────────────────

def test_end_to_end():
    """
    Tiny end-to-end test without any file I/O:
    noisy multichannel audio → STFT → MVDR → CRN → iSTFT → waveform.
    """
    from utils.audio import STFT
    from beamforming.mvdr import MVDRBeamformer
    from models.crn import CRN

    sr, n_mics, duration = 16000, 4, 0.5
    T = int(duration * sr)
    B = 1

    stft = STFT(n_fft=512, hop_length=160, win_length=320)
    beamformer = MVDRBeamformer(n_mics=n_mics, n_fft=512)
    model = CRN(encoder_channels=[16, 32, 64], lstm_hidden=64, lstm_layers=1)
    model.eval()

    # (1) Simulate multichannel noisy input
    noisy_mc = torch.randn(B, n_mics, T)

    # (2) STFT per channel
    noisy_flat = noisy_mc.reshape(B * n_mics, T)
    real_flat, imag_flat = stft(noisy_flat)
    _, F, T_spec = real_flat.shape
    mc_spec = torch.complex(
        real_flat.reshape(B, n_mics, F, T_spec),
        imag_flat.reshape(B, n_mics, F, T_spec),
    )

    # (3) MVDR beamforming
    bf_spec = beamformer(mc_spec)

    # (4) CRN masking
    with torch.no_grad():
        enh_real, enh_imag = model.enhance(bf_spec.real, bf_spec.imag)

    # (5) iSTFT
    enhanced_wav = stft.inverse(enh_real, enh_imag, length=T)

    assert enhanced_wav.shape == (B, T), f"Output shape: {enhanced_wav.shape}"
    assert not torch.isnan(enhanced_wav).any()

    print(f"  ✓ End-to-end pipeline: noisy_mc {noisy_mc.shape} → enhanced {enhanced_wav.shape}")


# ── Runner ────────────────────────────────────────────────────────────────

TESTS = [
    ("STFT round-trip",     test_stft_roundtrip),
    ("ULA positions",        test_ula_positions),
    ("MVDR beamformer",      test_mvdr_beamformer),
    ("CRN forward",          test_crn_forward),
    ("CRN enhance",          test_crn_enhance),
    ("SI-SNR loss",          test_sisnr_loss),
    ("Combined loss",        test_combined_loss),
    ("mix_signals SNR",      test_mix_signals),
    ("End-to-end pipeline",  test_end_to_end),
]

if __name__ == "__main__":
    print("=" * 60)
    print("Speech Enhancement Pipeline — Smoke Tests")
    print("=" * 60)
    passed, failed = 0, 0
    for name, fn in TESTS:
        print(f"\n[{name}]")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{len(TESTS)} passed", "✓" if failed == 0 else "✗")
    if failed:
        sys.exit(1)
