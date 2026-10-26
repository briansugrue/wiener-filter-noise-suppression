%% Adaptive Wiener Filter with Continuous Noise Tracking
clear; clc; close all;

%% 1. File paths
inputFile = "[drive name]\sp05\sp05_street_sn15.wav";
outputFile = '[drive name]\clean_audio.wav';
cleanFile = "[drive name]\sp05\sp05.wav";

%% 2. Load audio
[noisyAudio, fs] = audioread(inputFile);
[cleanReference, fsClean] = audioread(cleanFile);

% Convert stereo audio to mono
if size(noisyAudio, 2) > 1
    noisyAudio = mean(noisyAudio, 2);
end
if size(cleanReference, 2) > 1
    cleanReference = mean(cleanReference, 2);
end

noisyAudio = noisyAudio(:);
cleanReference = cleanReference(:);

assert(fs == fsClean, ...
    'The noisy audio and clean reference must have the same sample rate.');

N = length(noisyAudio);
duration = N / fs;

fprintf('Sample rate : %d Hz\n', fs);
fprintf('Duration    : %.2f seconds\n\n', duration);

%% 3. Parameters
FRAME_LEN = 512;
HOP_LEN = 256;
NFFT = FRAME_LEN;

% Noise estimation
INITIAL_NOISE_FRACTION = 0.10;  % lowest-energy frames used for the start value
SMOOTH_ALPHA = 0.85;            % smoothing of the noisy power before tracking
MIN_WIN = 25;                   % frames in the minimum-tracking window (~0.8 s)
BIAS = 1.3;                     % compensates for the minimum being biased low
                                % (tune between 1.5 and 2.5)

% Wiener filtering
MIN_GAIN = 0.25;
DD_ALPHA = 0.98;
GAIN_SMOOTHING = 0.10;
MIN_PRIOR_SNR = 10^(-25/10);    % floor for the a priori SNR (-25 dB)

% Diagnostic speech/noise label (does not affect the filter)
DIAG_SNR_THRESHOLD = 2.0;       % mean posterior SNR above this -> "speech"

win = hann(FRAME_LEN, 'periodic');

%% 4. Pad the signal for overlap-add reconstruction
% Padding prevents the beginning and end from being discarded.
padStart = FRAME_LEN;
paddedAudio = [zeros(padStart, 1); noisyAudio; zeros(FRAME_LEN, 1)];

numFrames = ceil((length(paddedAudio) - FRAME_LEN) / HOP_LEN) + 1;
requiredLength = (numFrames - 1) * HOP_LEN + FRAME_LEN;

if length(paddedAudio) < requiredLength
    paddedAudio(end + 1:requiredLength) = 0;
end

%% 5. Analyse frames and initialise the noise estimate
fprintf('Initialising noise estimate using low-energy frames...\n');

frameEnergy = zeros(numFrames, 1);
powerSpectra = zeros(NFFT/2 + 1, numFrames);
validFrame = false(numFrames, 1);

for i = 1:numFrames
    startIdx = (i - 1) * HOP_LEN + 1;
    rawFrame = paddedAudio(startIdx:startIdx + FRAME_LEN - 1);

    frameEnergy(i) = mean(rawFrame.^2);

    spectrum = fft(rawFrame .* win, NFFT);
    powerSpectra(:, i) = ...
        abs(spectrum(1:NFFT/2 + 1)).^2 / sum(win.^2);

    % Use only frames fully inside the original recording.
    validFrame(i) = ...
        startIdx >= padStart + 1 && ...
        startIdx + FRAME_LEN - 1 <= padStart + N;
end

validIndices = find(validFrame);
numInitial = max(3, ceil(INITIAL_NOISE_FRACTION * length(validIndices)));
numInitial = min(numInitial, length(validIndices));

% Select the lowest-energy frames as initial noise candidates.
[~, order] = sort(frameEnergy(validIndices), 'ascend');
initialIndices = validIndices(order(1:numInitial));

% Mean (not median): the median of a periodogram is only ~0.69 of the
% true mean, which would bias the noise estimate low.
noisePSD = max(mean(powerSpectra(:, initialIndices), 2), eps);

fprintf('Initial noise frames : %d\n\n', numInitial);

%% 6. Adaptive Wiener filtering
fprintf('Applying adaptive Wiener filter...\n');

outputBuffer = zeros(length(paddedAudio), 1);
windowSum = zeros(length(paddedAudio), 1);

speechFlags = false(numFrames, 1);
gainHistory = zeros(NFFT/2 + 1, numFrames);

previousGain = ones(NFFT/2 + 1, 1);
previousPower = zeros(NFFT/2 + 1, 1);

% Minimum-tracking state. The history starts at the initial estimate
% (divided by BIAS so the first noise estimate equals the initial one).
smoothedPower = noisePSD;
powerHistory = repmat(noisePSD / BIAS, 1, MIN_WIN);

for i = 1:numFrames
    startIdx = (i - 1) * HOP_LEN + 1;
    rawFrame = paddedAudio(startIdx:startIdx + FRAME_LEN - 1);

    % Analysis window and spectrum
    windowedFrame = rawFrame .* win;
    X = fft(windowedFrame, NFFT);
    noisyPower = abs(X(1:NFFT/2 + 1)).^2 / sum(win.^2);

    %% Continuous noise tracking (no speech/noise decision needed)
    % Only update on frames fully inside the recording, so the zero
    % padding at the edges cannot drag the estimate down.
    if validFrame(i)
        smoothedPower = SMOOTH_ALPHA * smoothedPower + ...
            (1 - SMOOTH_ALPHA) * noisyPower;

        powerHistory = [powerHistory(:, 2:end), smoothedPower];

        noisePSD = max(BIAS * min(powerHistory, [], 2), eps);
    end

    %% Decision-directed Wiener gain
    % Posterior SNR
    posteriorSNR = noisyPower ./ (noisePSD + eps);

    % Diagnostic label only (used for the printout and plot)
    speechFlags(i) = mean(posteriorSNR) > DIAG_SNR_THRESHOLD;

    % A priori SNR combines the previous clean-signal estimate
    % with the current observation.
    priorSNR = DD_ALPHA * ...
        (previousGain.^2 .* previousPower) ./ (noisePSD + eps) ...
        + (1 - DD_ALPHA) * max(posteriorSNR - 1, 0);

    % Keep the a priori SNR from collapsing to zero in noise-only bins
    priorSNR = max(priorSNR, MIN_PRIOR_SNR);

    % Wiener gain
    gain = priorSNR ./ (priorSNR + 1);

    % Smooth across neighbouring frequency bins
    gain = movmean(gain, 5);

    % Light smoothing of gain changes between frames
    gain = GAIN_SMOOTHING * previousGain + ...
        (1 - GAIN_SMOOTHING) * gain;

    % Prevent excessive attenuation
    gain = max(gain, MIN_GAIN);
    gain = min(gain, 1);

    gainHistory(:, i) = gain;

    %% Apply gain to the full spectrum
    % Mirror the positive-frequency gains to preserve a real signal.
    fullGain = [gain; gain(end-1:-1:2)];

    filteredFrame = real(ifft(X .* fullGain, NFFT));

    %% Overlap-add reconstruction
    outputBuffer(startIdx:startIdx + FRAME_LEN - 1) = ...
        outputBuffer(startIdx:startIdx + FRAME_LEN - 1) + ...
        filteredFrame .* win;

    windowSum(startIdx:startIdx + FRAME_LEN - 1) = ...
        windowSum(startIdx:startIdx + FRAME_LEN - 1) + win.^2;

    %% Save estimates for the next frame
    previousGain = gain;
    previousPower = noisyPower;
end

%% 7. Normalise overlap-add output and remove padding
covered = windowSum > 1e-10;
outputBuffer(covered) = outputBuffer(covered) ./ windowSum(covered);

filteredAudio = outputBuffer(padStart + 1:padStart + N);

%% 8. Match output level to input level
inputRMS = sqrt(mean(noisyAudio.^2));
filteredRMSBefore = sqrt(mean(filteredAudio.^2));

levelGain = inputRMS / (filteredRMSBefore + eps);
filteredAudio = filteredAudio * levelGain;

% Prevent clipping without unnecessarily normalising quiet audio.
outputPeakBeforeProtection = max(abs(filteredAudio));

if outputPeakBeforeProtection > 0.999
    filteredAudio = filteredAudio * ...
        (0.999 / outputPeakBeforeProtection);
    peakProtectionDB = 20 * log10( ...
        0.999 / outputPeakBeforeProtection);
else
    peakProtectionDB = 0;
end

%% 9. Save output
audiowrite(outputFile, filteredAudio, fs);

%% 10. Print diagnostics
speechCount = sum(speechFlags(validFrame));
noiseCount = sum(validFrame) - speechCount;

averageGain = mean(gainHistory(:, validFrame), 'all');
averageAttenuationDB = 20 * log10(averageGain + eps);

fprintf('\nFrame labels (diagnostic only):\n');
fprintf('Speech frames : %d (%.1f%%)\n', speechCount, ...
    100 * speechCount / sum(validFrame));
fprintf('Noise frames  : %d (%.1f%%)\n', noiseCount, ...
    100 * noiseCount / sum(validFrame));

fprintf('\nWiener gain:\n');
fprintf('Average gain  : %.3f\n', averageGain);
fprintf('Average attenuation: %.2f dB\n', averageAttenuationDB);
fprintf('Minimum gain  : %.3f\n', min(gainHistory(:)));
fprintf('Maximum gain  : %.3f\n', max(gainHistory(:)));

fprintf('\nLevel matching:\n');
fprintf('Input RMS                : %.6f\n', inputRMS);
fprintf('Filtered RMS before match: %.6f\n', filteredRMSBefore);
fprintf('Level matching gain      : %.2f dB\n', ...
    20 * log10(levelGain + eps));
fprintf('Peak protection applied  : %.2f dB\n', peakProtectionDB);
fprintf('Final output peak        : %.6f\n', max(abs(filteredAudio)));
fprintf('Final output RMS          : %.6f\n', ...
    sqrt(mean(filteredAudio.^2)));

fprintf('\nSaved cleaned audio to:\n%s\n', outputFile);

%% 11. Compare SNR against the clean reference
comparisonLength = min(length(cleanReference), N);
reference = cleanReference(1:comparisonLength);
noisyCompare = noisyAudio(1:comparisonLength);
filteredCompare = filteredAudio(1:comparisonLength);

snrBefore = snrVsRef(reference, noisyCompare);
snrAfter = snrVsRef(reference, filteredCompare);

fprintf('\nSNR before (noisy vs clean)    : %.2f dB\n', snrBefore);
fprintf('SNR after  (enhanced vs clean) : %.2f dB\n', snrAfter);
fprintf('Improvement                    : %.2f dB\n', ...
    snrAfter - snrBefore);

fprintf('\nDone.\n');

%% 12. Plots
time = (0:N-1) / fs;

figure('Name', 'Adaptive Wiener Filter Results');

subplot(3, 1, 1);
plot(time, noisyAudio);
title('Noisy Input');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(3, 1, 2);
plot(time, filteredAudio);
title('Enhanced Output');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(3, 1, 3);
plot(time, noisyAudio, 'DisplayName', 'Noisy input');
hold on;
plot(time, filteredAudio, 'DisplayName', 'Enhanced output');
title('Waveform Comparison');
xlabel('Time (s)');
ylabel('Amplitude');
legend;
grid on;

figure('Name', 'Frame Labels and Wiener Gain');

subplot(2, 1, 1);
frameTimes = (((0:numFrames-1) * HOP_LEN) + FRAME_LEN/2 ...
    - padStart) / fs;
plot(frameTimes(validFrame), double(speechFlags(validFrame)), '.-');
ylim([-0.1 1.1]);
yticks([0 1]);
yticklabels({'Noise', 'Speech'});
title('Frame Labels (diagnostic)');
xlabel('Time (s)');
grid on;

subplot(2, 1, 2);
imagesc(frameTimes, (0:NFFT/2) * fs/NFFT, gainHistory);
axis xy;
ylim([0 fs/2]);
colorbar;
title('Wiener Gain Over Time and Frequency');
xlabel('Time (s)');
ylabel('Frequency (Hz)');

%% 13. SNR evaluation function
function s = snrVsRef(ref, est)
    ref = ref(:);
    est = est(:);

    L = min(length(ref), length(est));
    ref = ref(1:L);
    est = est(1:L);

    denominator = est' * est;

    if denominator < eps
        s = -Inf;
        return;
    end

    % Align the estimate to the reference using least-squares scaling.
    scale = (est' * ref) / denominator;
    residual = ref - scale * est;

    s = 10 * log10( ...
        (sum(ref.^2) + eps) / (sum(residual.^2) + eps));
end