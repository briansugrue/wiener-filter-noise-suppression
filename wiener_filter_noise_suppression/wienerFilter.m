function cleanAudio = wienerFilter(noisyAudio, noiseFrames)
    frameLen = 512;
    hopLen = 256;
    win = hann(frameLen);

    % Estimate Noise Power Spectral Density (PSD)
    noisePSD = zeros(frameLen, 1);
    for i = 1:noiseFrames
        frame = noisyAudio((i-1)*hopLen + 1 : (i-1)*hopLen + frameLen);
        noisePSD = noisePSD + abs(fft(frame .* win)).^2;
    end
    noisePSD = noisePSD/noiseFrames;

    numFrames = floor((length(noisyAudio) - frameLen)/hopLen) + 1;
    cleanAudio = zeros(size(noisyAudio));
    windowSum = zeros(size(noisyAudio));

    for i = 1:numFrames
        startIdx = (i-1)*hopLen + 1;
        endIdx = startIdx + frameLen - 1;
        if endIdx > length(noisyAudio), break; end

        frame = noisyAudio(startIdx:endIdx) .* win;
        frameFFT = fft(frame);
        signalPSD = abs(frameFFT).^2;

        % Wiener Gain: H(w) = SNR/(SNR + 1)
        SNR = max(signalPSD - noisePSD, 0)./(noisePSD + eps);
        wienerGain = SNR./(SNR + 1);

        cleanFrame = real(ifft(wienerGain .* frameFFT));
        cleanAudio(startIdx:endIdx) = cleanAudio(startIdx:endIdx) + cleanFrame.*win;
        windowSum(startIdx:endIdx) = windowSum(startIdx:endIdx) + win.^2;
    end

    cleanAudio = cleanAudio./(windowSum + eps);
end