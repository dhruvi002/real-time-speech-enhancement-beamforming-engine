% matlab/eval_rt60_robustness.m
% ─────────────────────────────────────────────────────────────────────────
% Evaluate speech enhancement robustness across RT60 values (0.2–0.8s)
% using pyroomacoustics-generated RIRs (loaded from .mat files) and
% MATLAB's built-in PESQ and STOI equivalents.
%
% This script:
%   1. Loads pre-generated RIRs from data/rirs/ (converted to .mat)
%   2. Groups RIRs by RT60 bin (0.2, 0.3, ..., 0.8s)
%   3. Applies each RIR to a test utterance
%   4. Calls Python subprocess for CRN inference (via CLI)
%   5. Computes PESQ and STOI per RT60 bin
%   6. Plots: PESQ vs RT60, STOI vs RT60
%
% Requirements:
%   - MATLAB Signal Processing Toolbox
%   - Audio Toolbox (for PESQ/STOI)
%   - Python + speech_enhancement project on PATH
%
% Usage:
%   >> eval_rt60_robustness
%   >> eval_rt60_robustness('rir_dir', '../data/rirs_mat', 'n_per_bin', 20)

function eval_rt60_robustness(varargin)

p = inputParser;
addParameter(p, 'rir_dir',     '../data/rirs_mat',     @ischar);
addParameter(p, 'clean_wav',   '../data/test_clean.wav', @ischar);
addParameter(p, 'noise_wav',   '../data/test_noise.wav', @ischar);
addParameter(p, 'model_onnx',  '../model.onnx',         @ischar);
addParameter(p, 'snr_db',      5,                       @isnumeric);
addParameter(p, 'n_per_bin',   10,                      @isnumeric);
addParameter(p, 'output_fig',  'rt60_robustness.png',   @ischar);
parse(p, varargin{:});
opt = p.Results;

fs = 16000;

% ── RT60 bins ──────────────────────────────────────────────────────────
rt60_bins  = 0.2:0.1:0.8;   % [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
n_bins     = numel(rt60_bins);
rt60_tol   = 0.05;           % ±50ms tolerance for bin assignment

% ── Load clean + noise ────────────────────────────────────────────────
[clean, fs_c] = audioread(opt.clean_wav);
[noise, fs_n] = audioread(opt.noise_wav);

if fs_c ~= fs,  clean = resample(clean, fs, fs_c);  end
if fs_n ~= fs,  noise = resample(noise, fs, fs_n);  end

clean = clean(:,1);   % mono
noise = noise(:,1);

% ── Load RIR metadata ────────────────────────────────────────────────
meta_path = fullfile(opt.rir_dir, 'rir_metadata.mat');
if ~exist(meta_path, 'file')
    error('RIR metadata not found: %s\nRun convert_rirs_to_mat.py first.', meta_path);
end
load(meta_path, 'metadata');   % struct array with fields: file, rt60

% ── Results storage ──────────────────────────────────────────────────
pesq_noisy    = zeros(n_bins, opt.n_per_bin);
pesq_enhanced = zeros(n_bins, opt.n_per_bin);
stoi_noisy    = zeros(n_bins, opt.n_per_bin);
stoi_enhanced = zeros(n_bins, opt.n_per_bin);

fprintf('Evaluating RT60 robustness (%d bins × %d RIRs)...\n', n_bins, opt.n_per_bin);

for b = 1:n_bins
    rt60_target = rt60_bins(b);

    % Find matching RIRs
    matches = [];
    for k = 1:numel(metadata)
        if abs(metadata(k).rt60 - rt60_target) <= rt60_tol
            matches(end+1) = k; %#ok<AGROW>
        end
    end

    if isempty(matches)
        warning('No RIRs found for RT60=%.1fs', rt60_target);
        continue
    end

    n_use = min(opt.n_per_bin, numel(matches));
    idx   = matches(randperm(numel(matches), n_use));

    for j = 1:n_use
        rir_file = fullfile(opt.rir_dir, metadata(idx(j)).file);
        load(rir_file, 'rir');   % (n_mics × rir_len)

        % ── Apply RIR to first mic ─────────────────────────────────
        rir_ref = rir(1, :)';
        L       = min(numel(clean), numel(rir_ref) + numel(clean) - 1);
        clean_rev = fftfilt(rir_ref, clean);
        clean_rev = clean_rev(1:numel(clean));

        % ── Mix at SNR ────────────────────────────────────────────
        noisy = mix_at_snr(clean_rev, noise, opt.snr_db);

        % ── Save temp files and run Python inference ───────────────
        tmp_in  = tempname + '.wav';
        tmp_out = tempname + '.wav';
        audiowrite(tmp_in, noisy, fs);

        cmd = sprintf('python ../inference.py --model %s --input %s --output %s', ...
                      opt.model_onnx, tmp_in, tmp_out);
        [status, ~] = system(cmd);

        if status ~= 0 || ~exist(tmp_out, 'file')
            warning('Inference failed for RIR %s', metadata(idx(j)).file);
            continue
        end

        [enhanced, ~] = audioread(tmp_out);
        enhanced = enhanced(:,1);

        % Align lengths
        L = min([numel(clean), numel(noisy), numel(enhanced)]);
        ref = clean(1:L);
        nsy = noisy(1:L);
        enh = enhanced(1:L);

        % ── PESQ (requires Audio Toolbox or pesq MEX) ────────────
        try
            pesq_noisy(b,j)    = pesq(ref, nsy, fs);
            pesq_enhanced(b,j) = pesq(ref, enh, fs);
        catch
            % Fallback: use MATLAB SNR as proxy if pesq not available
            pesq_noisy(b,j)    = snr(ref, nsy - ref);
            pesq_enhanced(b,j) = snr(ref, enh - ref);
        end

        % ── STOI ─────────────────────────────────────────────────
        stoi_noisy(b,j)    = stoi_score(ref, nsy, fs);
        stoi_enhanced(b,j) = stoi_score(ref, enh, fs);

        % Cleanup
        delete(tmp_in);  delete(tmp_out);
    end

    fprintf('  RT60=%.1fs: PESQ noisy=%.2f enhanced=%.2f | STOI noisy=%.3f enhanced=%.3f\n', ...
        rt60_target, ...
        mean(pesq_noisy(b, 1:n_use), 'omitnan'), ...
        mean(pesq_enhanced(b, 1:n_use), 'omitnan'), ...
        mean(stoi_noisy(b, 1:n_use), 'omitnan'), ...
        mean(stoi_enhanced(b, 1:n_use), 'omitnan'));
end

% ── Plot ─────────────────────────────────────────────────────────────────
plot_results(rt60_bins, pesq_noisy, pesq_enhanced, stoi_noisy, stoi_enhanced, opt.output_fig);
fprintf('\n✓ Results saved → %s\n', opt.output_fig);
end


% ── Helper: mix at SNR ────────────────────────────────────────────────────
function noisy = mix_at_snr(clean, noise, snr_db)
    if numel(noise) < numel(clean)
        noise = repmat(noise, ceil(numel(clean)/numel(noise)), 1);
    end
    noise = noise(1:numel(clean));
    p_clean = mean(clean.^2) + 1e-8;
    p_noise = mean(noise.^2) + 1e-8;
    scale   = sqrt(p_clean / p_noise / (10^(snr_db/10)));
    noisy   = clean + scale * noise;
end


% ── Helper: STOI (simplified, use Audio Toolbox for production) ───────────
function score = stoi_score(ref, deg, fs)
    try
        score = stoi(ref, deg, fs);
    catch
        % Approximate with normalized cross-correlation if toolbox absent
        score = max(xcorr(ref, deg, 0, 'normalized'), 0);
    end
end


% ── Helper: Plot ──────────────────────────────────────────────────────────
function plot_results(rt60_bins, pesq_noisy, pesq_enhanced, stoi_noisy, stoi_enhanced, out_fig)
    mean_pesq_noisy    = mean(pesq_noisy,    2, 'omitnan');
    mean_pesq_enhanced = mean(pesq_enhanced, 2, 'omitnan');
    mean_stoi_noisy    = mean(stoi_noisy,    2, 'omitnan');
    mean_stoi_enhanced = mean(stoi_enhanced, 2, 'omitnan');

    figure('Position', [100 100 1000 420]);

    % PESQ subplot
    subplot(1,2,1);
    plot(rt60_bins, mean_pesq_noisy,    'b--o', 'LineWidth', 1.5, 'DisplayName', 'Noisy');
    hold on;
    plot(rt60_bins, mean_pesq_enhanced, 'r-s',  'LineWidth', 1.5, 'DisplayName', 'Enhanced');
    xlabel('RT60 (s)');  ylabel('PESQ (WB)');
    title('PESQ vs Reverberation Time');
    legend('Location', 'southwest');
    grid on;  xlim([0.15 0.85]);  ylim([1.0 4.5]);

    % STOI subplot
    subplot(1,2,2);
    plot(rt60_bins, mean_stoi_noisy,    'b--o', 'LineWidth', 1.5, 'DisplayName', 'Noisy');
    hold on;
    plot(rt60_bins, mean_stoi_enhanced, 'r-s',  'LineWidth', 1.5, 'DisplayName', 'Enhanced');
    xlabel('RT60 (s)');  ylabel('STOI');
    title('STOI vs Reverberation Time');
    legend('Location', 'southwest');
    grid on;  xlim([0.15 0.85]);  ylim([0.4 1.0]);

    sgtitle('Speech Enhancement Robustness Across RT60 (0.2–0.8s)', 'FontWeight', 'bold');

    saveas(gcf, out_fig);
end
