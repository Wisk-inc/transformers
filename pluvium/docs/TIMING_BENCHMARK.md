# When in the day it rains: what was measured

From version 0.5.0 the forecast splits each of its first three days into four six-hour periods and gives the chance
of rain (at least 1 mm) in each. This page says how that is computed, how it was checked for Jamaica, and where it
does not help. It needs no GPU.

## What is forecast

    chance(rain in a period) = chance(rain that day, calibrated by the blend)
                               x chance(the period is wet, given that the day is)

The second factor is a half-and-half mix of two sources:

- The systems' members: of the members that make the day wet, the share that make this period wet. The two
  ensembles and the four single forecasts each count the same.
- What is usual at the place: from six-hour satellite rain (CMORPH) over the blend's training year, how often each
  period was wet on the wet days of that month. `pluvium blend timing` computes it and stores it as `timing.npz`,
  6 KB for Jamaica.

Periods are six hours of UTC, because that is how every system publishes. They are labelled in the place's clock.
In Jamaica they are 7 pm to 1 am, 1 am to 7 am, 7 am to 1 pm and 1 pm to 7 pm. A blend with no `timing.npz` shows
no periods, for the reason in the first table.

## How it was checked

61 past runs, every sixth day from 16 May 2025, at Jamaica's 15 land grid cells: 2,745 days at a cell, 856 of them
wet. Truth is CMORPH summed to the same six-hour periods. What is usual was learned with the 7 days either side of
the scored day left out. Scores are Brier scores, so lower is better.

    python -m pluvium.benchmarks.timing_bench --root ~/pluvium-JM

### On wet days, which periods were wet

| Days 1 to 3 | Brier |
|---|---|
| Every period the same chance | 0.3344 |
| What is usual at the place | 0.1672 |
| The members alone | 0.1754 |
| Mix, members 25% | 0.1570 |
| **Mix, members 50% (used)** | **0.1543** |
| Mix, members 75% | 0.1591 |

| | Usual | Members alone | Mix 50% |
|---|---|---|---|
| Day 1 | 0.1650 | 0.1803 | 0.1564 |
| Day 2 | 0.1689 | 0.1878 | 0.1600 |
| Day 3 | 0.1679 | 0.1579 | 0.1462 |

Resampling the 61 runs: the mix is 7.8% better than what is usual (90% interval 3.6% to 11.9%). The members alone
are 4.9% worse (11.2% worse to 1.4% better). Every system on its own is worse than what is usual (0.1672): ENS
0.1785, GEFS 0.1825, GraphCast 0.2338, GFS 0.2597, AIFS 0.2620, HRES 0.2628.

### Is the wettest period the one forecast

| | Hit rate |
|---|---|
| A guess among four | 25% |
| What is usual | 66.5% |
| The members alone | 66.8% |
| Mix 50% | 66.7% |

No gain. In Jamaica the afternoon is the wettest period on most wet days, and the forecast seldom says otherwise.
The mix helps by giving better chances to the other periods, not by naming a different wettest one.

### The chance of rain in a period

This is the number the forecast prints. It depends on the daily chance as much as on the timing.

| Daily chance from | Timing from | Brier | Against what is usual |
|---|---|---|---|
| (the usual chance of rain in that period) | | 0.0840 | |
| The members | The members | 0.1430 | 70% worse |
| The members | Mix 50% | 0.1188 | 41% worse |
| The full-year blend, on its own training days | Mix 50% | 0.0695 | 17.3% better (12.3% to 21.8%) |
| The full-year blend, on its own training days | What is usual | 0.0728 | 13.4% better |

The last two rows are flattered: the blend had seen those days, and its daily chance alone was 25.9% better than
the usual daily chance there. The honest row is the next one. A blend trained on 97 days, scored on the 39 runs
outside them, had a daily chance 19.3% worse than usual, and its period chances were worse than usual too:

| 39 runs the blend had not seen | Brier |
|---|---|
| The usual chance of rain in that period | 0.0803 |
| Blend's day x mix 50% | 0.0849 (5.8% worse, 10.1% to 0.4% worse) |
| Blend's day x what is usual | 0.0867 |

In both cases the mix beat the same daily chance spread by what is usual: by 4.5% (1.9% to 7.4%) and by 2.0% (1.4%
to 2.6%).

## What this means

- The split of a wet day over its periods is better than the place's usual pattern by a measured 7.8%.
- The chance printed for a period is only as good as the daily chance it starts from. With a blend whose daily
  chance beats climatology it is better than the usual chance for that period; with a weak blend it is not.
- It rarely changes which period is most likely. Read it as "how likely is rain this afternoon", not as a forecast
  of the hour.
- Measured for Jamaica only, on one year, with a blend that had seen most of the scored days. A full-year blend
  scored on days it has not seen is still to be done.
- The hour-by-hour forecast is the Pluvium-500 network's job, and that needs the GPU training.
