# TriStream benchmarks

helix-lm 2.4 measures TriStream against simple baselines on the jobs it does: copying a voice through its codec
(resynthesis), turning one singer into another (conversion), singing written notes, and moving the pitch on
request. The numbers come with 95% intervals, a paired test against the last run, and one composite score.

The molab program runs the benchmark at the end of each eval stage and uploads the results next to the
checkpoints: `bench/LATEST.md` and `bench/latest.json` in `CorxLabs/Tristream-V2-SVS`, plus a folder per run.
A benchmark that cannot run is said in the log and the run goes on. With under an hour of the session left it
waits for the next eval.

## Run it

```bash
helix tristream bench run --suite quick                          # baselines only, about 15 seconds on a CPU
helix tristream bench run --suite standard --root /workspace/tristream --data vocalset=/workspace/tristream/raw/vocalset
helix tristream bench compare bench-out/results.json old/results.json   # what got worse, exit code 1 if any
helix tristream bench leaderboard run1/results.json run2/results.json
python -m helix_lm.scripts.tristream_bench --root /workspace/tristream   # what the molab program runs
```

## Suites

| suite | sets | tasks | items per set |
|---|---|---|---|
| quick | synthetic | resynthesis, conversion, singing from notes, pitch control | 6 |
| standard | synthetic, VocalSet test singers | the quick tasks, technique, robustness, register | 24 |
| full | adds NUS-48E, Opencpop, M4Singer (research licences, given with `--data`) | all, with long takes and speed | 60 |

## Systems

- `identity` returns its input. It is perfect at resynthesis and the worst possible converter (it keeps the
  source's voice), so it marks both ends of the scale.
- `griffin-lim` keeps the magnitude spectrum and rebuilds the phase. It is the floor for a codec round trip.
- `harmonic-noise` is a small source-filter vocoder: YIN pitch, a peak-following cepstral envelope, harmonics
  plus shaped noise. It converts by moving the envelope to the target voice and sings notes from a pitch curve.
  A trained model should beat it everywhere.
- `tristream` is the trained model from a molab root (`codec/codec.pt` and `runs/<size>/latest.pt`). A root
  without both files gives no TriStream row, never a pass-through scored as if it were the model. Outputs are
  cached per checkpoint, so a new checkpoint is always measured afresh.

## What a run gives

`results.json` holds every row (system, task, item, metrics) and the summary (mean, interval, count). A task
that fails is listed under `failed` with its error, and the other tasks still run.
`report.md` is the table per task. `helix tristream bench compare` lists every main metric that got worse
by more than its tolerance, with a Wilcoxon signed-rank test on the paired items, so noise is not read as a
regression. The composite (`helix_lm.music.composite`) puts each main metric between a floor (0) and a target
(1) and averages them with weights; it is for choosing checkpoints, not for papers.

See [bench/METRICS.md](bench/METRICS.md) for each measure and its limits, [bench/DATASETS.md](bench/DATASETS.md)
for the sets and their licences, and [bench/RESULTS.md](bench/RESULTS.md) for the numbers measured so far.
