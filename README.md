# wiener-filter-noise-suppression

A MATLAB implementation of an adaptive Wiener filter for speech enhancement. It tracks background noise continuously, estimates a per-frequency a priori SNR with the decision-directed method, and applies a smoothed, floored Wiener gain in the STFT domain.

Noisy audio is sourced from the NOIZEUS speech corpus, available to researchers courtesy of Dr. Philip Loizou et al. at UT Dallas.
Link: https://ecs.utdallas.edu/loizou/speech/noizeus/

## Method

1. **Framing:** 512-sample Hann windows with 50% overlap (64 ms frames, 32 ms hop at 8 kHz).
2. **Noise estimate:** initialised from the mean power spectrum of the lowest-energy frames, then tracked continuously with a sliding-window minimum of the smoothed power spectrum (about 0.8 s window), scaled by a bias-compensation factor. No hard voice-activity gating is used.
3. **A priori SNR:** decision-directed estimation (`DD_ALPHA = 0.98`), which blends the previous frame's enhanced output with the current frame's measured SNR, with a -25 dB floor.
4. **Gain:** Wiener gain, smoothed across neighbouring frequency bins and lightly across time, with a minimum gain floor to limit speech distortion and musical noise.
5. **Reconstruction:** overlap-add with window-sum normalisation. The output is level-matched to the input RMS and peak-protected.

## Results (speaker sp05, NOIZEUS street noise)

Settings: `BIAS = 1.3`, `MIN_GAIN = 0.25`, `DD_ALPHA = 0.98`, `GAIN_SMOOTHING = 0.10`.

| Input SNR | SNR before (dB) | SNR after (dB) | Improvement (dB) |
|-----------|-----------------|----------------|------------------|
| 0 dB      | 2.62            | 3.44           | +0.83            |
| 5 dB      | 5.73            | 7.64           | +1.91            |
| 10 dB     | 9.80            | 10.63          | +0.83            |
| 15 dB     | 14.45           | 11.07          | -3.38            |

SNR is measured against the clean reference after least-squares scaling of the signal, so even the noisy input gets credit for the best possible global gain. This is a stricter baseline than plain SNR, and the "before" values therefore differ from the nominal input levels.

**Takeaways**

- The filter gives modest gains (about 1-2 dB) when the noise is strong (0-10 dB input SNR).
- At 15 dB input the filter degrades the waveform SNR by 3.38 dB. There is little noise to remove, and the heavy attenuation (average gain about 0.39) distorts the speech more than it helps. The output can still sound cleaner, because waveform SNR does not capture perceived noise in the gaps between words.
- The average gain changes little between conditions, so the filter's aggressiveness is not yet adapted to the input SNR.

## Limitations and next steps

- Results are from one speaker (sp05) and one noise type, so the parameters may be fitted to this file.
- A parameter sweep over `MIN_GAIN` and `BIAS`, evaluation on more speakers, and an input-SNR-dependent gain floor are planned.
- Perceptual metrics (STOI, PESQ) and segmental SNR would track listening quality better than a single global SNR.

## Usage

1. Download the NOIZEUS clean and noisy files.
2. Edit the three file paths at the top of `main.m` (`inputFile`, `outputFile`, `cleanFile`).
3. Run `main.m`. It writes the enhanced audio, prints the gain and SNR diagnostics, and plots the waveforms and the Wiener gain over time and frequency.
