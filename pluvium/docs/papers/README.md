# Pluvium technical papers

Four short papers on the methods in Pluvium. Each states its method, how it was tested, what it found and where it
falls short. The numbers in papers 1 to 3 are written into the text by a script (`scripts/doc_papers.py`) from the
benchmark's result files, so the text cannot drift from the data.

| | Paper | The question |
|---|---|---|
| 1 | [Tail-aware pooling with online weights](01_blend_frontier.md) | Can a multi-model rain forecast be made better where it was weakest, without losing what it had? |
| 2 | [Forecasting beyond the training record](02_record_storms.md) | What should a calibrated forecast do when the storm is bigger than anything it was trained on? |
| 3 | [Sharp scenarios from smooth chances](03_scenarios.md) | Calibrated chances are smooth at long range. Can scenarios built from them look like weather? |
| 4 | [A learning consensus for tropical cyclone tracks](04_pluvium_cyclone.md) | How close can six public forecast systems come to the official hurricane forecast? |

## How to read the results

Rain is scored against CMORPH, a satellite estimate, on land, for April 2025 to March 2026, in six regions of the
world and in Jamaica. Training used April 2024 to March 2025. Hurricane tracks are scored against the National
Hurricane Center's best tracks, with settings chosen on three seasons and tested once on a fourth.

Papers 1 to 3 share a weakness that each of them states: the rain test year is the one on which the weak spots
were found. Every design choice was made inside the training year and written down before the test year was
scored, but a second look at a year is weaker evidence than a fresh year.

One step was settled late. The rule by which each place decides about the record-storm hand-over was fixed after
five of the seven places had been scored. Paper 2 gives that history, including the rule that failed.

One assumption turned out to be wrong. The online weights of paper 1 were tested with each day's rain known two
days later, and the satellite record is in fact published about four and a half months late. Section 4.7 of paper
1 repeats the test with that delay.

These are research results from one person's project. They have not been peer reviewed.

## Citing

```
Nathan (2026). Pluvium: rain and tropical cyclone forecasts from six operational systems.
https://pypi.org/project/pluvium/
```

## Related documents

- [The second full-year benchmark](../FRONTIER_BENCHMARK.md) and [the first](../FULL_YEAR_BENCHMARK.md)
- [The cyclone benchmark](../CYCLONE_BENCHMARK.md)
- [What did not work](../WHAT_DID_NOT_WORK.md)
- [How Pluvium works](../ARCHITECTURE.md)
