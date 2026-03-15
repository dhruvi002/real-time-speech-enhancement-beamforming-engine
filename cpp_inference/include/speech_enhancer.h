/**
 * cpp_inference/include/speech_enhancer.h
 * ----------------------------------------
 * Real-time speech enhancement engine using ONNXRuntime.
 *
 * Implements a streaming inference pipeline:
 *   1. Accumulate audio samples in a ring buffer
 *   2. When a frame is ready, compute STFT
 *   3. Run CRN inference via ONNXRuntime
 *   4. Reconstruct waveform via overlap-add iSTFT
 *
 * Target: sub-18ms end-to-end latency at 16kHz, 20ms frames (320 samples),
 *         10ms hop (160 samples), 50% overlap-add.
 */

#pragma once

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>
#include <complex>
#include <stdexcept>


// ── STFT parameters ────────────────────────────────────────────────────────

constexpr int   SAMPLE_RATE  = 16000;
constexpr int   N_FFT        = 256;
constexpr int   HOP_LENGTH   = 128;    // 8ms
constexpr int   WIN_LENGTH   = 256;    // 16ms
constexpr int   N_BINS       = N_FFT / 2 + 1;   // 129
constexpr float TARGET_LATENCY_MS = 18.0f;


// ── Ring buffer ────────────────────────────────────────────────────────────

class RingBuffer {
public:
    explicit RingBuffer(int capacity);

    void push(const float* samples, int n);
    bool read(float* out, int n) const;  // peek last n samples
    int  size() const { return size_; }

private:
    std::vector<float> buf_;
    int capacity_;
    int head_ = 0;
    int size_  = 0;
};


// ── STFT (Hann window, real-to-complex) ────────────────────────────────────

class STFT {
public:
    STFT();

    // Compute STFT of a WIN_LENGTH-sample frame.
    // Returns real[] and imag[] each of length N_BINS.
    void forward(const float* frame, float* real_out, float* imag_out) const;

    // Reconstruct a WIN_LENGTH-sample frame from N_BINS real+imag.
    void inverse(const float* real_in, const float* imag_in, float* frame_out) const;

private:
    std::vector<float> window_;   // Hann window (WIN_LENGTH)
    mutable std::vector<float> fft_buf_;  // padded FFT buffer (N_FFT)

    // Simple DFT fallback (replace with FFTW / KissFFT in production)
    void dft(const float* in, float* real_out, float* imag_out, int n) const;
    void idft(const float* real_in, const float* imag_in, float* out, int n) const;
};


// ── Speech Enhancer ────────────────────────────────────────────────────────

class SpeechEnhancer {
public:
    /**
     * @param model_path  Path to CRN .onnx file
     */
    explicit SpeechEnhancer(const std::string& model_path);

    /**
     * Process a chunk of audio samples.
     *
     * @param input     Pointer to HOP_LENGTH float32 samples (single channel)
     * @param output    Pointer to HOP_LENGTH float32 samples (enhanced)
     *
     * Internally uses overlap-add with 50% overlap.
     * First call may output zeros (pipeline fill latency = one hop = 10ms).
     */
    void process(const float* input, float* output);

    /**
     * Process a full WAV file (offline mode).
     */
    void process_file(const std::string& in_path, const std::string& out_path);

    /** Returns mean latency of last 100 process() calls in milliseconds. */
    float mean_latency_ms() const;

    /** Reset internal state (for new utterance). */
    void reset();

private:
    // ── ONNX ──────────────────────────────────────────────────────────
    Ort::Env                          env_;
    Ort::SessionOptions               session_opts_;
    std::unique_ptr<Ort::Session>     session_;
    Ort::AllocatorWithDefaultOptions  allocator_;

    std::vector<const char*> input_names_  = {"noisy_real", "noisy_imag"};
    std::vector<const char*> output_names_ = {"enhanced_real", "enhanced_imag"};

    // ── Audio buffers ─────────────────────────────────────────────────
    RingBuffer         input_ring_;
    STFT               stft_;

    std::vector<float> frame_buf_;    // WIN_LENGTH
    std::vector<float> noisy_real_;   // N_BINS
    std::vector<float> noisy_imag_;   // N_BINS
    std::vector<float> enh_real_;     // N_BINS
    std::vector<float> enh_imag_;     // N_BINS
    std::vector<float> ola_buf_;      // WIN_LENGTH * 2 overlap-add accumulator

    // ── Latency tracking ─────────────────────────────────────────────
    std::vector<float> latency_history_;
    int                latency_idx_ = 0;

    // ── Helpers ───────────────────────────────────────────────────────
    void run_inference();
};