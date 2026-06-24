%% Noise Suppression using a Wiener Filter, with Adaptive Noise Estimation
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
 
%% SETTINGS
INPUT_FILE     = "C:\Users\brian\Desktop\street_10dB\sp05_street_sn10.wav";     % Path to your noisy input file
OUTPUT_FILE    = "C:\Users\brian\Desktop\clean_audio.wav";                      % Path to save cleaned output
NOISE_DURATION = 0.1;                                                           % Seconds of leading noise used to initialise estimate
 
% Wiener Filter Tuning
FRAME_LEN      = 512;       % FFT frame size. Increase for more frequency resolution
HOP_LEN        = 256;       % Hop size. Typically FRAME_LEN/2
NOISE_SCALE    = 0.5;       % Scale noise estimate down (0.1–1.0). Lower = less suppression
MIN_GAIN       = 0.1;       % Minimum Wiener gain (0–1). Prevents total muting of signal
 
% Adaptive Noise Estimation
NOISE_ALPHA    = 0.95;      % Noise PSD smoothing factor (0.9–0.98). Higher = slower adaptation
VAD_THRESHOLD  = 0.02;      % Energy threshold for VAD (0–1). Frames below this update the estimate
 
 
%% LOAD AUDIO
fprintf('Loading: %s\n', INPUT_FILE);
[noisyAudio, fs] = audioread(INPUT_FILE);
 
% Convert stereo to mono if needed
if size(noisyAudio, 2) > 1
    fprintf('Stereo detected — converting to mono\n');
    noisyAudio = mean(noisyAudio, 2);
end
 
% Normalize input to [-1, 1]
noisyAudio = noisyAudio / max(abs(noisyAudio) + eps);
 
fprintf('Sample rate : %d Hz\n', fs);
fprintf('Duration    : %.2f seconds\n', length(noisyAudio) / fs);
 
 
%% ESTIMATE NOISE POWER SPECTRAL DENSITY (PSD)
% Estimate noise PSD from the leading noise-only section
win = hann(FRAME_LEN);
 
noiseFrames = max(floor((NOISE_DURATION * fs - FRAME_LEN) / HOP_LEN) + 1, 1);
fprintf('Initialising noise estimate from first %.2fs (%d frames)...\n', NOISE_DURATION, noiseFrames);
 
noisePSD = zeros(FRAME_LEN, 1);
for i = 1:noiseFrames
    startIdx = (i-1) * HOP_LEN + 1;
    endIdx   = startIdx + FRAME_LEN - 1;
    if endIdx > length(noisyAudio), break; end
    frame    = noisyAudio(startIdx:endIdx) .* win;
    noisePSD = noisePSD + abs(fft(frame)).^2;
end
noisePSD = (noisePSD / noiseFrames) * NOISE_SCALE;
 
% Derive initial VAD threshold from the noise energy
% Set it to a multiple of the mean noise frame energy so it sits just above the noise floor

noiseEnergy    = mean(noisePSD) * FRAME_LEN;                    % rough energy from PSD
VAD_THRESHOLD  = 4 * noiseEnergy;                               % speech is typically well above this
fprintf('VAD energy threshold set to: %.6f\n', VAD_THRESHOLD);
 
 
%% APPLY WIENER FILTER WITH ADAPTIVE NOISE ESTIMATION
fprintf('Applying Wiener filter with adaptive noise estimation...\n');
 
numFrames  = floor((length(noisyAudio) - FRAME_LEN) / HOP_LEN) + 1;
cleanAudio = zeros(size(noisyAudio));
windowSum  = zeros(size(noisyAudio));
 
noiseFrameCount  = 0;   % diagnostic counter
speechFrameCount = 0;
 
for i = 1:numFrames
    startIdx = (i-1) * HOP_LEN + 1;
    endIdx   = startIdx + FRAME_LEN - 1;
    if endIdx > length(noisyAudio), break; end
 
    frame    = noisyAudio(startIdx:endIdx) .* win;
    frameFFT = fft(frame);
    sigPSD   = abs(frameFFT).^2;
 
    % VAD: classify frame as speech or noise
    frameEnergy = sum(frame.^2) / FRAME_LEN;
    isSpeech    = frameEnergy > VAD_THRESHOLD;
 
    if isSpeech
        speechFrameCount = speechFrameCount + 1;
    else
        % Silence frame. Update noise PSD estimate recursively
        noisePSD        = NOISE_ALPHA * noisePSD + (1 - NOISE_ALPHA) * sigPSD;
        noiseFrameCount = noiseFrameCount + 1;
    end
 
    % Wiener Gain
    SNR        = max(sigPSD - noisePSD, 0) ./ (noisePSD + eps);
    wienerGain = max(SNR ./ (SNR + 1), MIN_GAIN);
 
    % Apply gain and reconstruct frame
    cleanFrame = real(ifft(wienerGain .* frameFFT));
 
    % Overlap-add synthesis
    cleanAudio(startIdx:endIdx) = cleanAudio(startIdx:endIdx) + cleanFrame .* win;
    windowSum(startIdx:endIdx)  = windowSum(startIdx:endIdx)  + win.^2;
end
 
fprintf('Frames classified as speech : %d\n', speechFrameCount);
fprintf('Frames used to update noise : %d\n', noiseFrameCount);
 
% Normalize by window overlap
cleanAudio = cleanAudio ./ (windowSum + eps);
 
% Match output RMS loudness to input
inputRMS  = sqrt(mean(noisyAudio.^2));
outputRMS = sqrt(mean(cleanAudio.^2));
cleanAudio = cleanAudio * (inputRMS / (outputRMS + eps));
 
% Clip to [-1, 1] to prevent clipping artifacts
cleanAudio = max(min(cleanAudio, 1), -1);
 
 
%% SAVE AUDIO OUTPUT
audiowrite(OUTPUT_FILE, cleanAudio, fs);
fprintf('Saved cleaned audio to: %s\n', OUTPUT_FILE);
 
 
%% GRAPH/PLOT RESULTS
t = (0:length(noisyAudio)-1) / fs;
 
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
 
% SNR estimate
snr_before = snr(noisyAudio);
snr_after  = snr(cleanAudio);
fprintf('\nEstimated SNR before: %.1f dB\n', snr_before);
fprintf('Estimated SNR after : %.1f dB\n', snr_after);
fprintf('Improvement         : %.1f dB\n', snr_after - snr_before);
 
fprintf('\nDone.\n');