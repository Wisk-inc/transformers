# Benchmark sets

| set | what | licence | commercial models | tasks |
|---|---|---|---|---|
| synthetic | voices built here from a pitch curve, a vowel's formants and breath noise; every answer known | Apache-2.0 | yes | resynthesis, conversion, singing, pitch control |
| VocalSet | 20 professional singers, 17 techniques, about 10 hours | CC BY 4.0 | yes | conversion, resynthesis, technique, register |
| NUS-48E | 12 singers, each song sung and its lyrics read | research only (licence agreement) | no | conversion, resynthesis |
| Opencpop | one singer, 100 Mandarin pop songs with notes | CC BY-NC-ND | no | singing, resynthesis |
| M4Singer | 20 singers in four voice types, Mandarin, with notes | CC BY-NC-SA | no | singing, conversion |

`helix_lm.bench.data.registry.usable_for(task, commercial_only=True)` lists the sets a commercial model may be
measured on.

## Splits

VocalSet's test singers are f2, f8, m3, m5 and m10 (`vocalset.TEST_SINGERS`). The molab program trains on the
others, so the benchmark never hears a singer the model learned. Conversion pairs never pair a singer with
themself. The research sets are only read when their folders are given (`--data opencpop=/path`).

## The synthetic set

`bench.data.synthetic.make_set` writes three voices (low, mid and high, each with its own vocal tract length,
tilt and breathiness) singing random melodies on five vowels, with second-order formant resonators of 80, 100
and 140 Hz bandwidth, vibrato and portamento. Each take has a `<id>.truth.json` with the notes and the pitch at
100 Hz. The same seed gives the same set. It measures what a real set cannot: exact pitch truth, exact note
timing, and pitch control without a singer's own errors.

## Manifests

`bench.data.manifest` writes the items with their sizes and checksums, so two runs can prove they were measured
on the same files.
