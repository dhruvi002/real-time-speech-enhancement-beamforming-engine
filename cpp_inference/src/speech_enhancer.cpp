/**
 * cpp_inference/src/speech_enhancer.cpp
 * ---------------------------------------
 * Implementation of the streaming speech enhancement engine.
 *
 * Production considerations noted inline:
 *   - FFTW3 or KissFFT should replace the naive DFT below
 *   - For multi-mic input, MVDR preprocessing runs before this stage
 *   - ONNXRuntime CUDA provider can be enabled for GPU acceleration
 */

#include "speech_enhancer.h"

#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <cstring>

// Simple WAV I/O (header-only, 16kHz mono 16-bit PCM only)
#include "wav_io.h"


// ── RingBuffer ─────────────────────────────────────────────────────────────

RingBuffer::RingBuffer(int capacity)
    : buf_(capacity, 0.0f), capacity_(capacity) {}

void RingBuffer::push(const float* samples, int n) {
    for (int i = 0; i < n; ++i) {
        buf_[head_] = samples[i];
        head_ = (head_ + 1) % capacity_;
        if (size_ < capacity_) ++size_;
    }
}

bool RingBuffer::read(float* out, int n) const {
    if (size_ < n) return false;
    int start = ((head_ - size_) % capacity_ + capacity_) % capacity_;
    for (int i = 0; i < n; ++i)
        out[i] = buf_[(start + i) % capacity_];
    return true;
}


// ── STFT ──────────────────────────────────────────────────────────────────

STFT::STFT() : window_(WIN_LENGTH), fft_buf_(N_FFT, 0.0f) {
    // Hann window
    for (int i = 0; i < WIN_LENGTH; ++i)
        window_[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (WIN_LENGTH - 1)));
}

void STFT::forward(const float* frame, float* real_out, float* imag_out) const {
    // Apply window + zero-pad to N_FFT
    std::fill(fft_buf_.begin(), fft_buf_.end(), 0.0f);
    int pad = (N_FFT - WIN_LENGTH) / 2;
    for (int i = 0; i < WIN_LENGTH; ++i)
        fft_buf_[pad + i] = frame[i] * window_[i];

    dft(fft_buf_.data(), real_out, imag_out, N_FFT);
}

void STFT::inverse(const float* real_in, const float* imag_in, float* frame_out) const {
    std::vector<float> time_buf(N_FFT);
    idft(real_in, imag_in, time_buf.data(), N_FFT);

    int pad = (N_FFT - WIN_LENGTH) / 2;
    float win_sum = 0.0f;
    for (float w : window_) win_sum += w * w;
    float scale = win_sum > 1e-8f ? 1.0f / win_sum : 1.0f;

    for (int i = 0; i < WIN_LENGTH; ++i)
        frame_out[i] = time_buf[pad + i] * window_[i] * scale;
}

/**
 * Naive O(N^2) DFT — replace with KissFFT in production:
 *   kiss_fft_cfg cfg = kiss_fft_alloc(N_FFT, 0, 0, 0);
 *   kiss_fft(cfg, cx_in, cx_out);
 */
void STFT::dft(const float* in, float* re, float* im, int n) const {
    for (int k = 0; k <= n / 2; ++k) {
        double r = 0.0, i = 0.0;
        for (int t = 0; t < n; ++t) {
            double angle = 2.0 * M_PI * k * t / n;
            r += in[t] * std::cos(angle);
            i -= in[t] * std::sin(angle);
        }
        re[k] = static_cast<float>(r);
        im[k] = static_cast<float>(i);
    }
}

void STFT::idft(const float* re, const float* im, float* out, int n) const {
    for (int t = 0; t < n; ++t) {
        double val = re[0];   // DC
        for (int k = 1; k < n / 2; ++k) {
            double angle = 2.0 * M_PI * k * t / n;
            val += 2.0 * (re[k] * std::cos(angle) - im[k] * std::sin(angle));
        }
        val += re[n / 2] * std::cos(M_PI * t);  // Nyquist
        out[t] = static_cast<float>(val / n);
    }
}


// ── SpeechEnhancer ────────────────────────────────────────────────────────

SpeechEnhancer::SpeechEnhancer(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "SpeechEnhancer"),
      input_ring_(WIN_LENGTH * 4),
      frame_buf_(WIN_LENGTH),
      noisy_real_(N_BINS), noisy_imag_(N_BINS),
      enh_real_(N_BINS),   enh_imag_(N_BINS),
      ola_buf_(WIN_LENGTH * 2, 0.0f),
      latency_history_(100, 0.0f)
{
    // Configure session: single-threaded for predictable latency
    session_opts_.SetIntraOpNumThreads(1);
    session_opts_.SetInterOpNumThreads(1);
    session_opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Optionally enable CUDA:
    // OrtCUDAProviderOptions cuda_opts{};
    // session_opts_.AppendExecutionProvider_CUDA(cuda_opts);

    session_ = std::make_unique<Ort::Session>(
        env_,
        model_path.c_str(),
        session_opts_
    );

    std::cout << "[SpeechEnhancer] Loaded model: " << model_path << "\n";
    std::cout << "[SpeechEnhancer] STFT: n_fft=" << N_FFT
              << " hop=" << HOP_LENGTH << " bins=" << N_BINS << "\n";
}

void SpeechEnhancer::process(const float* input, float* output) {
    auto t_start = std::chrono::high_resolution_clock::now();

    // Push new samples into ring buffer
    input_ring_.push(input, HOP_LENGTH);

    // Read a full frame (WIN_LENGTH samples) for STFT
    if (!input_ring_.read(frame_buf_.data(), WIN_LENGTH)) {
        // Buffer not full yet — output silence
        std::fill(output, output + HOP_LENGTH, 0.0f);
        return;
    }

    // Compute STFT of current frame
    stft_.forward(frame_buf_.data(), noisy_real_.data(), noisy_imag_.data());

    // Run CRN inference
    run_inference();

    // iSTFT reconstruction
    std::vector<float> recon_frame(WIN_LENGTH);
    stft_.inverse(enh_real_.data(), enh_imag_.data(), recon_frame.data());

    // Overlap-add: accumulate and output one hop
    for (int i = 0; i < WIN_LENGTH; ++i)
        ola_buf_[i] += recon_frame[i];

    // Output first HOP_LENGTH samples
    std::copy(ola_buf_.begin(), ola_buf_.begin() + HOP_LENGTH, output);

    // Shift OLA buffer
    std::copy(ola_buf_.begin() + HOP_LENGTH, ola_buf_.begin() + WIN_LENGTH, ola_buf_.begin());
    std::fill(ola_buf_.begin() + HOP_LENGTH, ola_buf_.end(), 0.0f);

    // Track latency
    auto t_end = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    latency_history_[latency_idx_++ % 100] = ms;
}

void SpeechEnhancer::run_inference() {
    // Shape: (1, N_BINS, 1) — single frame, one time step
    std::array<int64_t, 3> shape = {1, N_BINS, 1};

    // Create input tensors
    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value real_tensor = Ort::Value::CreateTensor<float>(
        mem_info, noisy_real_.data(), N_BINS, shape.data(), 3);
    Ort::Value imag_tensor = Ort::Value::CreateTensor<float>(
        mem_info, noisy_imag_.data(), N_BINS, shape.data(), 3);

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(real_tensor));
    inputs.push_back(std::move(imag_tensor));

    auto outputs = session_->Run(
        Ort::RunOptions{nullptr},
        input_names_.data(), inputs.data(), inputs.size(),
        output_names_.data(), output_names_.size()
    );

    // Copy outputs
    float* enh_re = outputs[0].GetTensorMutableData<float>();
    float* enh_im = outputs[1].GetTensorMutableData<float>();
    std::copy(enh_re, enh_re + N_BINS, enh_real_.begin());
    std::copy(enh_im, enh_im + N_BINS, enh_imag_.begin());
}

void SpeechEnhancer::process_file(
    const std::string& in_path,
    const std::string& out_path)
{
    // Load 16kHz mono WAV
    std::vector<float> wav_in;
    int sr = 0;
    wav_read(in_path, wav_in, sr);

    if (sr != SAMPLE_RATE)
        throw std::runtime_error("Input must be 16kHz mono WAV");

    reset();
    std::vector<float> wav_out(wav_in.size(), 0.0f);
    std::vector<float> chunk_out(HOP_LENGTH);

    for (int i = 0; i + HOP_LENGTH <= (int)wav_in.size(); i += HOP_LENGTH) {
        process(wav_in.data() + i, chunk_out.data());
        std::copy(chunk_out.begin(), chunk_out.end(), wav_out.begin() + i);
    }

    wav_write(out_path, wav_out, SAMPLE_RATE);
    std::cout << "[SpeechEnhancer] Processed → " << out_path << "\n";
    std::cout << "[SpeechEnhancer] Mean latency: " << mean_latency_ms() << "ms\n";
}

float SpeechEnhancer::mean_latency_ms() const {
    int n = std::min(latency_idx_, 100);
    if (n == 0) return 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += latency_history_[i];
    return sum / n;
}

void SpeechEnhancer::reset() {
    input_ring_ = RingBuffer(WIN_LENGTH * 4);
    std::fill(ola_buf_.begin(), ola_buf_.end(), 0.0f);
    latency_idx_ = 0;
}


// ── CLI entry point ───────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: speech_enhancer <model.onnx> <input.wav> <output.wav>\n";
        return 1;
    }

    try {
        SpeechEnhancer enhancer(argv[1]);
        enhancer.process_file(argv[2], argv[3]);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}