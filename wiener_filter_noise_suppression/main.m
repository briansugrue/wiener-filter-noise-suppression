%% Noise Suppression using a Wiener Filter
%
%  Instructions:
%    1. Set INPUT_FILE to your noisy .wav file path
%    2. Set NOISE_DURATION to the length (seconds) of
%       silence/noise-only audio at the START of the file
%       (used to estimate the noise profile)
%    3. Run the script — cleaned file is saved automatically
%
% -------------------------------------------------------------------------

clear; clc; close all;

%% Settings
INPUT_FILE     = "[path to input audio file]";                   % Path to input audio file
OUTPUT_FILE    = "[path to save audio output]";                  % Path to save output audio
NOISE_DURATION = 0.1;                                            % Seconds of noise-only audio at the start

% Wiener filter tuning
FRAME_LEN = 512;   % FFT frame size (samples). Increase for more freq resolution
HOP_LEN   = 256;   % Hop size (samples). Typically FRAME_LEN/2


%% Load Audio
fprintf('Loading: %s\n', INPUT_FILE);
[noisyAudio, fs] = audioread(INPUT_FILE);

% Convert stereo to mono if needed
if size(noisyAudio, 2) > 1
    fprintf('Stereo detected — converting to mono\n');
    noisyAudio = mean(noisyAudio, 2);
end

% Normalize input to [-1, 1]
noisyAudio = noisyAudio/max(abs(noisyAudio) + eps);

fprintf('Sample rate : %d Hz\n', fs);
fprintf('Duration    : %.2f seconds\n', length(noisyAudio)/fs);


%% Estimate Noise Profile
noiseFrames = floor((NOISE_DURATION*fs - FRAME_LEN)/HOP_LEN) + 1;
noiseFrames = max(noiseFrames, 1);

fprintf('Estimating noise from first %.2fs (%d frames)...\n', NOISE_DURATION, noiseFrames);

win = hann(FRAME_LEN);
noisePSD = zeros(FRAME_LEN, 1);

for i = 1:noiseFrames
    startIdx = (i-1) * HOP_LEN + 1;
    endIdx   = startIdx + FRAME_LEN - 1;
    if endIdx > length(noisyAudio), break; end

    frame    = noisyAudio(startIdx:endIdx).*win;
    noisePSD = noisePSD + abs(fft(frame)).^2;
end

noisePSD = noisePSD/noiseFrames;


%% Apply Wiener Filter
fprintf('Applying Wiener filter...\n');

numFrames = floor((length(noisyAudio) - FRAME_LEN) / HOP_LEN) + 1;
cleanAudio  = zeros(size(noisyAudio));
windowSum   = zeros(size(noisyAudio));

for i = 1:numFrames
    startIdx = (i-1) * HOP_LEN + 1;
    endIdx   = startIdx + FRAME_LEN - 1;
    if endIdx > length(noisyAudio), break; end

    frame    = noisyAudio(startIdx:endIdx) .* win;
    frameFFT = fft(frame);
    sigPSD   = abs(frameFFT).^2;

    % Wiener gain: SNR / (SNR + 1), floored at 0 to avoid boosting noise
    SNR        = max(sigPSD - noisePSD, 0)./(noisePSD + eps);
    wienerGain = SNR./(SNR + 1);

    % Apply gain and reconstruct frame
    cleanFrame = real(ifft(wienerGain .* frameFFT));

    % Overlap-add
    cleanAudio(startIdx:endIdx) = cleanAudio(startIdx:endIdx) + cleanFrame.*win;
    windowSum(startIdx:endIdx)  = windowSum(startIdx:endIdx)  + win.^2;
end

% Normalize by window overlap
cleanAudio = cleanAudio./(windowSum + eps);

% Match output RMS loudness to input
inputRMS  = sqrt(mean(noisyAudio.^2));
outputRMS = sqrt(mean(cleanAudio.^2));
cleanAudio = cleanAudio*(inputRMS/(outputRMS + eps));

% Clip to [-1, 1] to prevent any clipping artifacts
cleanAudio = max(min(cleanAudio, 1), -1);


%% --- SAVE OUTPUT ---
audiowrite(OUTPUT_FILE, cleanAudio, fs);
fprintf('Saved cleaned audio to: %s\n', OUTPUT_FILE);


%% --- VISUALISE RESULTS ---
t = (0:length(noisyAudio)-1)/fs;

figure('Name', 'Noise Suppression Results', 'Position', [100 100 1000 600]);

% Waveforms
subplot(2, 2, 1);
plot(t, noisyAudio, 'Color', [0.8 0.2 0.2]);
title('Noisy Signal (Waveform)'); xlabel('Time (s)'); ylabel('Amplitude');
xlim([0 t(end)]);

subplot(2, 2, 2);
plot(t, cleanAudio, 'Color', [0.2 0.6 0.2]);
title('Cleaned Signal (Waveform)'); xlabel('Time (s)'); ylabel('Amplitude');
xlim([0 t(end)]);

% Spectrograms
subplot(2, 2, 3);
spectrogram(noisyAudio, win, HOP_LEN, FRAME_LEN, fs, 'yaxis');
title('Noisy Signal (Spectrogram)');
colormap('jet'); clim([-80 0]);

subplot(2, 2, 4);
spectrogram(cleanAudio, win, HOP_LEN, FRAME_LEN, fs, 'yaxis');
title('Cleaned Signal (Spectrogram)');
colormap('jet'); clim([-80 0]);

% SNR estimate (if audio is long enough to split signal from noise region)
noisyNoise  = noisyAudio(1 : round(NOISE_DURATION*fs));
cleanSignal = cleanAudio(round(NOISE_DURATION*fs)+1 : end);
noisySignal = noisyAudio(round(NOISE_DURATION*fs)+1 : end);

if length(cleanSignal) > 0 && length(noisySignal) > 0
    snr_before = snr(noisySignal);
    snr_after  = snr(cleanSignal);
    fprintf('\nEstimated SNR before: %.1f dB\n', snr_before);
    fprintf('Estimated SNR after : %.1f dB\n', snr_after);
    fprintf('Improvement         : %.1f dB\n', snr_after - snr_before);
end

fprintf('\nDone.\n');