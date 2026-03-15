/**
 * cpp_inference/include/wav_io.h
 * --------------------------------
 * Minimal header-only WAV reader/writer.
 * Supports: 16kHz, mono, 16-bit PCM (standard output of speech datasets).
 *
 * For production use, replace with libsndfile or dr_wav.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <string>


// ── WAV header structures ─────────────────────────────────────────────────

#pragma pack(push, 1)
struct WavHeader {
    char     riff_id[4];      // "RIFF"
    uint32_t file_size;
    char     wave_id[4];      // "WAVE"
    char     fmt_id[4];       // "fmt "
    uint32_t fmt_size;        // 16 for PCM
    uint16_t audio_format;    // 1 = PCM
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;       // sample_rate * num_channels * bits/8
    uint16_t block_align;     // num_channels * bits/8
    uint16_t bits_per_sample;
    char     data_id[4];      // "data"
    uint32_t data_size;
};
#pragma pack(pop)


// ── Read ─────────────────────────────────────────────────────────────────

inline void wav_read(
    const std::string&   path,
    std::vector<float>&  samples,
    int&                 sample_rate)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open WAV: " + path);

    WavHeader hdr;
    file.read(reinterpret_cast<char*>(&hdr), sizeof(WavHeader));

    if (std::strncmp(hdr.riff_id, "RIFF", 4) != 0 ||
        std::strncmp(hdr.wave_id, "WAVE", 4) != 0)
        throw std::runtime_error("Not a valid WAV file: " + path);

    if (hdr.audio_format != 1)
        throw std::runtime_error("Only PCM WAV supported (format=1)");

    if (hdr.bits_per_sample != 16)
        throw std::runtime_error("Only 16-bit PCM supported");

    sample_rate = static_cast<int>(hdr.sample_rate);
    int n_frames = hdr.data_size / (hdr.bits_per_sample / 8) / hdr.num_channels;

    samples.resize(n_frames);
    std::vector<int16_t> raw(n_frames * hdr.num_channels);
    file.read(reinterpret_cast<char*>(raw.data()),
              n_frames * hdr.num_channels * sizeof(int16_t));

    // Convert to float32, downmix to mono if needed
    constexpr float kScale = 1.0f / 32768.0f;
    for (int i = 0; i < n_frames; ++i) {
        if (hdr.num_channels == 1) {
            samples[i] = raw[i] * kScale;
        } else {
            float sum = 0.0f;
            for (int c = 0; c < hdr.num_channels; ++c)
                sum += raw[i * hdr.num_channels + c] * kScale;
            samples[i] = sum / hdr.num_channels;
        }
    }
}


// ── Write ────────────────────────────────────────────────────────────────

inline void wav_write(
    const std::string&         path,
    const std::vector<float>&  samples,
    int                        sample_rate)
{
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot write WAV: " + path);

    uint32_t n_samples  = static_cast<uint32_t>(samples.size());
    uint32_t data_size  = n_samples * sizeof(int16_t);
    uint32_t file_size  = sizeof(WavHeader) - 8 + data_size;

    WavHeader hdr;
    std::memcpy(hdr.riff_id,  "RIFF", 4);
    hdr.file_size       = file_size;
    std::memcpy(hdr.wave_id,  "WAVE", 4);
    std::memcpy(hdr.fmt_id,   "fmt ", 4);
    hdr.fmt_size        = 16;
    hdr.audio_format    = 1;
    hdr.num_channels    = 1;
    hdr.sample_rate     = static_cast<uint32_t>(sample_rate);
    hdr.bits_per_sample = 16;
    hdr.byte_rate       = sample_rate * sizeof(int16_t);
    hdr.block_align     = sizeof(int16_t);
    std::memcpy(hdr.data_id,  "data", 4);
    hdr.data_size       = data_size;

    file.write(reinterpret_cast<const char*>(&hdr), sizeof(WavHeader));

    // Convert float32 → int16 with clipping
    constexpr float kScale = 32767.0f;
    for (float s : samples) {
        float clamped = std::max(-1.0f, std::min(1.0f, s));
        int16_t pcm   = static_cast<int16_t>(clamped * kScale);
        file.write(reinterpret_cast<const char*>(&pcm), sizeof(int16_t));
    }
}
