# Benchmark metrics

Every metric is in `helix_lm.bench.metrics`. `INFO` there gives each one's unit and whether lower or higher is
better; reports and the regression check read it.

## Pitch and voicing

Pitch is tracked with YIN (`helix_lm.audio.yin`) at 100 frames a second, frames centred on their time.

- `f0_rmse_cents`, `f0_rmse_hz`: error on frames voiced in both tracks. One octave slip in a hundred frames adds
  about 120 cents, so read it with `gpe` and `ffe`.
- `log_f0_corr`: correlation of the log pitch contours; blind to a constant offset.
- `vuv_error`: share of frames whose voicing differs.
- `gpe`, `fpe_cents`, `ffe`: gross pitch errors (over 20%), the spread of the rest, and frames with either error.
- `note_accuracy`, `note_error_cents`: voiced frames within 50 cents of the written note, and the mean error.
- `shift_error_cents`: the pitch shift asked for against the one sung.
- vibrato rate and depth from the detrended pitch in the 3 to 9 Hz band.

## Spectrum

- `mcd_db`: mel cepstral distortion. 25 coefficients from 80 mel bands, the usual `10/ln10 * sqrt(2)` scale,
  each signal's silent frames left out, the rest aligned by DTW in a 20% band. The same take gives 0.
- `mrstft`, `lsd_db`, `spectral_convergence`, `si_snr_db`: multi-resolution STFT distance, log-spectral distance,
  spectral convergence and scale-invariant SNR. SI-SNR needs sample alignment, so it is only for resynthesis.
- `tilt_error_db`: spectral tilt from 100 Hz to 8 kHz in dB per octave. Pink noise reads -3.0.

## Voice

- `speaker_similarity`: with a speaker encoder (`encoder_from`), the cosine of two embeddings. Without one, an
  envelope print: the mean of mel cepstra 1 to 12 over each signal's frames within 20 dB of its loudest, each
  frame's bands held within 50 dB of its loudest band, as a distance in dB turned into `exp(-d / 4 dB)`. One
  singer on two notes scores about 0.7, two singers on one vowel about 0.4. It also moves with the vowels sung,
  so it ranks systems on one set; it is not a speaker verification score.
- `identity_margin`: similarity to the target voice minus similarity to the source voice. Above 0 means the
  conversion moved toward the target. The `identity` system scores about -0.47 on the quick suite.
- `hnr_error_db`, `jitter_error`, `shimmer_error`: harmonics-to-noise ratio (Boersma's autocorrelation method)
  and frame-based jitter and shimmer, as differences from the reference.
- formants: LPC at 11.025 kHz. Within 12% on a source-filter vowel at 100 and 220 Hz; at 440 Hz it reports
  harmonics. No task scores formants.

## Clean output

- `glitches_per_minute`: clicks and dropouts away from the clip's edges. A click is a jump six times the local
  motion and 2.5 times the largest jump within 25 ms around it, so a bright voice's edge at every glottal closure
  is not one. Before 2.4.0 every period of a bright voice counted, and the repair step smoothed each of them.
- `quality_proxy`: a signal check (periodicity, clipping, quiet floor), not a listening test. A UTMOS predictor
  can be loaded with `load_predictor` where it is installed.
- `buzz_db`: energy at the codec's frame rate and its multiples.
- `consonant_clarity`: consonant onsets kept, against the reference.

## Sets of outputs

- `fad`: Frechet distance between the frame embeddings of two sets.
- `rtf`, `peak_memory_gb`: compute time per second of audio and peak accelerator memory.

## Statistics

Means come with 95% bootstrap intervals (2,000 resamples). Two runs are compared item by item with the sign test
and the Wilcoxon signed-rank test; `effect_size` is the paired Cohen's d.
