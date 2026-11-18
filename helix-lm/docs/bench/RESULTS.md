# Benchmark results so far

Measured in the development container on a CPU, helix-lm 2.4.0. TriStream's own numbers come from the GPU run
and are uploaded after each eval to `bench/LATEST.md` in `CorxLabs/Tristream-V2-SVS`.

## Baselines, quick suite (synthetic set, 6 items, 15 seconds)

| task | metric | identity | griffin-lim | harmonic-noise |
|---|---|---|---|---|
| resynthesis | mcd_db | 0.000 | 0.501 | 3.70 |
| resynthesis | f0_rmse_cents | 0.0 | 43.8 | 163.8 |
| resynthesis | vuv_error | 0.000 | 0.008 | 0.052 |
| conversion | speaker_similarity | 0.528 | | 0.571 |
| conversion | identity_margin | -0.472 | | +0.131 |
| conversion | ffe | 0.000 | | 0.220 |
| singing | note_accuracy | | | 0.924 |
| singing | onset_error_ms | | | 37.2 |
| pitch control | shift_error_cents | 350.0 | | 1.78 |

The identity system keeps the source's voice, so its margin is negative, as it should be. The harmonic-noise
converter moves toward the target voice (+0.13) but slips an octave at some onsets, which is most of its pitch
error.

## Pitch tracking (synthetic voices, vowel "a", three notes each)

| voice | YIN gross errors | YIN error (cents) | old tracker gross errors | old tracker error (cents) |
|---|---|---|---|---|
| low | 0% | 6.0 | 29% | 2173 |
| mid | 0% | 6.1 | 0% | 13.7 |
| high | 0% | 6.6 | 0.4% | 75.9 |

The prepare stage makes TriStream's pitch targets with YIN from 2.4.0 on.

## Fixes in 2.4.1 (a bug scan before the first GPU run)

- Device checks: a test tags every tensor that would be on the GPU and reports any op that mixes it with a CPU
  tensor. On 2.4.0 it found 9 places: the prepare stage's breath check (every clip), the glitch detector, the
  buzz, robotic and sync checks of the fail-safe loop, a low-pass filter, and the converter (audio from disk on
  the CPU, the model on the GPU). The fail-safe loop also drew GPU noise with a CPU generator. All fixed.
- The benchmark's register task crashed (a count read as a list), so the standard suite never finished.
- The pass-through system scored 17.7 register pops a minute on clean takes; now 0.
- Glitch check on a 4-minute song: 3.1 s in 2.4.0, 0.16 s now (block maxima and running sums).

## Fixes the tests found

- Voice similarity: the old fallback (cosine of the cepstra's mean and spread) scored a copy of a take at -0.6
  against the take and two different singers at 0.96. The envelope print scores the copy 0.60 against its take,
  another singer 0.41, and a conversion 0.83 against its target voice (0.43 against its source).
- Clicks: every period of a bright voice counted as a click (252 in two seconds of a clean voice), and the
  repair step smoothed each. Now 0, and a real click is still found.
- Shimmer was missing for any voice that ran to the end of a clip (one pitch frame more than amplitude frames).
- The harmonic-noise envelope sat 10 dB low under 1.5 kHz. With a peak-following envelope its copy went from
  3.68 to 2.94 dB MCD.
