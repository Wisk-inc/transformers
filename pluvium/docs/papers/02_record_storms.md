# Forecasting beyond the training record: a hand-over to the raw systems, decided by staged storms

**Pluvium technical paper 2.** Nathan, November 2026.

## Summary

A calibrated forecast is a map learned from the forecasts of its training period. On a storm larger than anything in that period the map extrapolates the lesson of ordinary days, which is that big forecasts usually verify smaller. Two days before Hurricane Melissa crossed Jamaica, every raw system gave an 83% to 100% chance of 50 mm and the first Pluvium Blend gave 66%.

We hand such a forecast back, in part, to the plain agreement of the raw systems. The share handed over is exactly zero inside the range of the training forecasts and rises beyond it, and it is scaled by how many systems agree. To test the idea without touching the test year, we stage record storms inside the training year: the wettest days are removed from training and then forecast.

With the four wettest days withheld, the calibrated forecast gave on average 10 mm where 22 mm fell. The hand-over improved the Brier score at 50 mm on those days in 5 of 7 places (Jamaica, the Caribbean, the Maritime Continent, the US Great Plains and the Southern Andes) and not in the Alps and the US Pacific Northwest.

That split is the main finding. The hand-over is not good or bad in general. It helps where the raw systems are the better guide to a storm the calibration has never seen, and hurts where they overcall heavy rain against the truth. So each place decides from its own staged storms: the hand-over is used only where it lowered the Brier score at 50 mm on them.

In the test year the hand-over was on in Jamaica, the Caribbean, the Maritime Continent, the US Great Plains and the Southern Andes and off in the Alps and the US Pacific Northwest. Mean Brier skill at 50 mm on days 1-3 over the 7 places is +0.1996 with no hand-over, +0.2089 with it always on and +0.2133 as the places decided. Each place's verdict had the sign of what the hand-over then did in the test year in 7 of 7 places, and in 2 of the 2 scored after the rule was final.

This paper also reports a rule that failed. Before settling on the staged storms we switched the hand-over by an audit of the training period's out-of-fold forecasts, a far larger sample. It would have had the right sign in 4 of 7 places. The hand-over touches one forecast in a thousand, so its effect on a year's score is a few tenths of a percent, and the sign of that number is not evidence.

The staged-storm rule was the first one written down, was replaced by the audit, and was restored when 5 of the 7 places (the Caribbean, the Maritime Continent, the Alps, the US Pacific Northwest and the Southern Andes) had been scored. It is not a blind result for them. The rain test year (April 2025 to March 2026) is the year on which the weak spots were found. Every design choice was made inside the training year and written down before the test year was scored, but a second look at a year is weaker evidence than a fresh year.

## 1. The problem

Calibration shrinks forecasts towards what usually happens, and it should: a raw 80 mm forecast usually verifies well below 80 mm. A record storm is exactly the case where that lesson is wrong, and it is the case that matters most. No amount of care in fitting helps, because the training period does not hold the event.

The raw systems are not bound in the same way. They are physics, or machine learning trained on forty years of reanalysis, and they can produce an event unlike anything in a one-year training set.

## 2. Method

### 2.1 The raw consensus

Every system has one vote, however many members it runs. A system's chance of a threshold is the share of its members that reach it, 0 or 1 for a single run, and the consensus is the mean over the systems present. On ordinary days this is a poorly calibrated forecast, which is why it is not the forecast.

### 2.2 Support: has the calibration seen a forecast like this one?

    score = ( log(1 + amount) - log(1 + edge) ) / width

`amount` is the raw consensus mean. `edge` is the 99.9th percentile of that quantity over the training rows, never below 20 mm, so one training row in a thousand lies beyond it. `width` is log 2: the score reaches one at about twice the edge.

### 2.3 The hand-over

    g = min( clip(score, 0, 1) * clip(2 * agree, 0, 1), 0.95 )
    final = (1 - g) * calibrated + g * raw consensus

`agree` is the share of systems that call for 25 mm or more. One system drawing a bullseye where the others are dry does not take the forecast away from the calibration. Inside the training range `g` is exactly zero, so ordinary days, and every score computed on them, are untouched by construction. An earlier version used a smooth gate, which is never exactly zero and touched 16% of ordinary rows.

### 2.4 Staging record storms

The remedy cannot be tuned on the test year, and ordinary days say nothing about it. So the situation is staged inside the training year. The four wettest days, at least five days apart, are removed from training along with the two days either side. Twelve ordinary days are removed as a control. A calibrated forecast is trained on the rest, and the withheld days are forecast with and without the hand-over.

**The verdict.** A place uses the hand-over if it lowered the Brier score at 50 mm on its withheld storms. The control days are not part of the verdict. They show that ordinary days are left alone. With no storm to judge by, the hand-over is kept, because a shrunk record storm is the costlier error. `pluvium blend train` stages the storms, prints the verdict and saves it with the model. The owner can overrule it (`--handover always` or `never`).

### 2.5 The yearly check, and why it does not decide

The staged storms are four days. A larger sample is at hand: every training row has an out-of-fold forecast, made by networks that never saw that row's days. The check applies the hand-over to all of them and compares

    Brier(with hand-over) against Brier(without),   at 50 mm, over all training rows

with the rows' sampling weights undone. It costs nothing, and we first used it as the switch. It does not work as one. The hand-over touches about one row in a thousand, most of them heavy days of a kind the networks have seen. Over a year its effect on the score is a few tenths of a percent in either direction, and in three of seven places the sign changes when the check is run on the one-network component's forecast in place of the pool's ([WHAT_DID_NOT_WORK.md](../WHAT_DID_NOT_WORK.md)). It is still run at every training and saved with the model, as the price of the hand-over in an ordinary year.

### 2.6 What the user is told

The forecast table carries three more columns. An alert tier for 50 mm (watch at a 10% chance, warning at 30%, extreme at 60%). A flag when the raw systems call for more than anything in the training period, in which case amounts should be read as a lower bound. And how rare the expected amount is at that place, from a generalised Pareto fit to the area's daily record (Pickands, 1975; Hosking and Wallis, 1987) with one shape for the area and a threshold and scale per place (Dalrymple, 1960). Beyond four times the length of the record the answer is only "rarer than once in that many years".

## 3. Results

### 3.1 Staged storms (training year)

| Place | Withheld days | Observed mean, mm | Calibrated amount, mm | Brier at 50 mm: calibrated -> with hand-over | Raw systems alone | Ordinary days handed over |
|---|---|---|---|---|---|---|
| Jamaica | 2024-07-03, 2024-11-26, 2024-11-05, 2024-09-23 | 48.1 | 15.5 | 0.3419 -> **0.3202** | 0.3197 | 0.00% |
| Caribbean | 2024-11-15, 2024-06-23, 2024-10-19, 2024-06-14 | 11.8 | 7.3 | 0.0468 -> **0.0460** | 0.0516 | 0.01% |
| Maritime Continent | 2025-03-19, 2024-11-26, 2025-01-26, 2025-01-10 | 21.6 | 14.0 | 0.0965 -> **0.0962** | 0.1129 | 0.08% |
| Alps | 2024-10-08, 2024-09-08, 2024-10-03, 2024-04-01 | 20.3 | 10.4 | 0.0770 -> 0.0781 | 0.0845 | 0.07% |
| US Pacific Northwest | 2024-11-13, 2024-11-22, 2025-01-31, 2024-05-04 | 18.9 | 7.3 | 0.0835 -> 0.0835 | 0.0953 | 0.01% |
| US Great Plains | 2024-11-18, 2024-11-03, 2025-04-05, 2024-04-28 | 20.7 | 11.9 | 0.1083 -> **0.1036** | 0.1017 | 0.02% |
| Southern Andes | 2024-08-01, 2025-03-29, 2024-09-01, 2024-04-02 | 15.1 | 4.3 | 0.0827 -> **0.0782** | 0.0666 | 0.14% |

The calibrated forecast shrinks every staged storm. The hand-over helped most where the raw systems alone beat the calibration on those days (Jamaica, the US Great Plains and the Southern Andes). Where they did not, it still helped a little in the Caribbean and the Maritime Continent, because it acts only on the rows that lie beyond the training range and on which the systems agree. It did not help in the Alps and the US Pacific Northwest, where the raw systems were well behind the calibration. In Jamaica the withheld days include Hurricane Beryl (3 July 2024).

### 3.2 Two rules from training data, and what happened

The staged storms decide. The yearly check is shown beside them. The last column is the test year: Brier skill at 50 mm on days 1-3 with the hand-over always on, minus without it. A verdict is bold where it had the sign of the test year.

| Place | Staged storms: Brier at 50 mm on the withheld storms | Their verdict | Yearly check: Brier at 50 mm over the training period | It would have said | Test year: what the hand-over did |
|---|---|---|---|---|---|
| Jamaica | -6.4% | **keep** | -0.20% | **keep** (at 4 of 4 refits: keep) | +0.0566 |
| Caribbean | -1.8% | **keep** | +0.41% | switch off (at 0 of 4 refits: keep) | +0.0048 |
| Maritime Continent | -0.4% | **keep** | +0.00% | switch off (at 2 of 4 refits: keep) | +0.0015 |
| Alps | +1.5% | **switch off** | +0.20% | **switch off** (at 0 of 4 refits: keep) | -0.0110 |
| US Pacific Northwest | +0.0% | **switch off** | +0.21% | **switch off** (at 0 of 4 refits: keep) | -0.0191 |
| US Great Plains | -4.3% | **keep** | -0.17% | **keep** (at 4 of 4 refits: keep) | +0.0152 |
| Southern Andes | -5.4% | **keep** | +0.07% | switch off (at 0 of 4 refits: keep) | +0.0175 |

The staged storms had the sign of the test year in 7 of 7 places and the yearly check in 4 of 7. The first count is partly hindsight (see the summary). The effects in the test year are several times larger than anything the yearly check measures, in both directions: a year's verdict on the hand-over is made by a few storms.

### 3.3 The test year, three ways

Brier skill at 50 mm, days 1-3. Bold is the best of the three.

| Place | No hand-over | Always on (as first declared) | As the staged storms decided (Pluvium Blend) | Their verdict |
|---|---|---|---|---|
| Jamaica | +0.4580 | **+0.5147** | **+0.5147** | keep |
| Caribbean | +0.0819 | **+0.0867** | **+0.0867** | keep |
| Maritime Continent | +0.0804 | **+0.0818** | **+0.0818** | keep |
| Alps | **+0.0908** | +0.0798 | **+0.0908** | switch off |
| US Pacific Northwest | **+0.3073** | +0.2882 | **+0.3073** | switch off |
| US Great Plains | +0.1135 | **+0.1287** | **+0.1287** | keep |
| Southern Andes | +0.2652 | **+0.2827** | **+0.2827** | keep |
| **Mean** | +0.1996 | +0.2089 | +0.2133 | |

Brier skill at 50 mm, days 4-7. Bold is the best of the three.

| Place | No hand-over | Always on (as first declared) | As the staged storms decided (Pluvium Blend) | Their verdict |
|---|---|---|---|---|
| Jamaica | +0.2427 | **+0.2469** | **+0.2469** | keep |
| Caribbean | **+0.0568** | +0.0559 | +0.0559 | keep |
| Maritime Continent | +0.0492 | **+0.0503** | **+0.0503** | keep |
| Alps | **+0.0325** | +0.0248 | **+0.0325** | switch off |
| US Pacific Northwest | **+0.1682** | +0.1589 | **+0.1682** | switch off |
| US Great Plains | **+0.0329** | +0.0287 | +0.0287 | keep |
| Southern Andes | +0.1323 | **+0.1326** | **+0.1326** | keep |
| **Mean** | +0.1021 | +0.0997 | +0.1022 | |

### 3.4 The tail as a whole, and what a warning meant

**Skill in the tail against the raw consensus of the six systems** (threshold-weighted CRPS above 25 mm; 1 is perfect, 0 no better than the raw consensus)

| Place | days 1-3 | days 4-7 | days 8-10 | days 11-14 |
|---|---|---|---|---|
| Jamaica | -0.003 | -0.032 | -0.019 | -0.222 |
| Caribbean | +0.064 | +0.053 | +0.053 | +0.047 |
| Maritime Continent | +0.065 | +0.059 | +0.050 | +0.046 |
| Alps | +0.125 | +0.081 | +0.118 | +0.065 |
| US Pacific Northwest | +0.206 | +0.141 | +0.161 | +0.108 |
| US Great Plains | +0.030 | +0.048 | +0.060 | +0.067 |
| Southern Andes | +0.352 | +0.233 | +0.192 | +0.166 |

**What a warning for 50 mm meant, days 1-3** (issued: share of all place-days; came: how often 50 mm or more then fell; caught: share of all 50 mm cases covered)

| Place | 50 mm days | Watch (10%+): issued / came / caught | Warning (30%+) | Extreme (60%+) |
|---|---|---|---|---|
| Jamaica | 0.83% | 1.18% / 44% / 63% | 0.57% / 81% / 55% | 0.40% / 98% / 47% |
| Caribbean | 0.77% | 1.24% / 19% / 30% | 0.20% / 39% / 10% | 0.04% / 69% / 3% |
| Maritime Continent | 3.07% | 5.55% / 18% / 33% | 0.57% / 42% / 8% | 0.07% / 77% / 2% |
| Alps | 0.32% | 0.58% / 18% / 33% | 0.10% / 31% / 10% | 0.00% / 18% / 0% |
| US Pacific Northwest | 0.27% | 0.62% / 31% / 71% | 0.23% / 53% / 46% | 0.06% / 67% / 15% |
| US Great Plains | 0.49% | 0.93% / 19% / 37% | 0.19% / 39% / 16% | 0.02% / 89% / 4% |
| Southern Andes | 0.51% | 1.23% / 30% / 72% | 0.46% / 49% / 44% | 0.13% / 66% / 17% |

### 3.5 Hurricane Melissa over Jamaica

Observed on 2025-10-28 (CMORPH): 154 mm on average, up to 447 mm; 100% of places had 50 mm or more and 96% had 100 mm or more.

| Days ahead | First Blend: chance of 50 mm, expected mm | New Blend: chance of 50 mm, of 100 mm, expected mm | Places under an extreme warning | Raw consensus: chance of 50 mm, mean mm |
|---|---|---|---|---|
| 2 | 66%, 63 | 92%, 75%, 195 | 97% | 94%, 173 |
| 4 | 42%, 50 | 47%, 21%, 78 | 35% | 47%, 77 |
| 8 | 9%, 16 | 10%, 1%, 17 | 0% | 18%, 28 |
| 12 | 2%, 5 | 1%, 0%, 5 | 0% | 23%, 41 |

**How rare was it?** A generalised Pareto fit to Jamaica's CMORPH record from 2020-01 to 2025-04 (5.1 years, 450 places, all before the test year) puts the daily amount reached once a year at 79 mm on average over the places, and once in five years at 123 mm. On 2025-10-28, 100% of places had more than their once-a-year amount, 78% more than their once-in-five-years amount, and 35% an amount rarer than once in 20 years, which is as far as a record of that length can say. The median place had a day expected about once in 13 years. These are rough figures from five years of satellite estimates.

## 4. What did not work

**Handing over everywhere.** That was the design as first written down. It was tested by staging storms in two regions, and both happened to be regions where it helps. Section 3.3 shows what it cost in the others.

**An audit of an ordinary year as the switch.** Section 2.5. A larger sample, and the wrong one: nearly all of it is heavy days the networks have seen, and the question is about the storm they have not.

**Discounting the raw systems first.** Scaling the raw chances by how often they verified at the top of the training forecasts, and never letting the hand-over lower a chance, still did harm on the Alps' staged storms and gave away most of the gain where the plain hand-over helps. A switch per place is simpler and did better.

**Starting the hand-over further out**, at the 99.99th percentile of the training forecasts or at their maximum, so that no training row could speak against it. On the staged storms that gave nearly all of the gain away: the gain comes from forecasts just beyond what the training period held. The numbers are in [WHAT_DID_NOT_WORK.md](../WHAT_DID_NOT_WORK.md).

## 5. Limits

- The rain test year (April 2025 to March 2026) is the year on which the weak spots were found. Every design choice was made inside the training year and written down before the test year was scored, but a second look at a year is weaker evidence than a fresh year.
- Where the staged storms switch the hand-over off, a record storm is forecast by the calibration alone, which shrinks it. The beyond-record flag and the count of raw systems calling for 50 mm are still shown.
- The verdict rests on four storms of one training year, one amount (50 mm) and one truth. A place can pass and still be hurt in a given year, or fail and then meet the storm that would have justified the hand-over.
- The rule was settled after most places had been scored. A fresh year in fresh places is the test it still needs.
- The hand-over raises the RMSE of the expected amount slightly where it is used.
- Whether the raw systems overcall depends on the truth. Against gauges in mountains the verdict could differ.
- Return periods come from one to a few years of satellite estimates and are rough.

## 6. Reproducing

```bash
pip install pluvium
pluvium blend train --root ~/pluvium-JM        # stages the storms and prints the verdict
```

The staged-storm experiment is `pluvium.tails.withheld.withheld_experiment` and its verdict `pluvium.tails.withheld.verdict`. The yearly check is `pluvium.tails.audit.audit_handover`. The test-year tables come from `python -m pluvium.benchmarks.frontier_bench`.

## References

- Dalrymple, T. (1960). Flood-frequency analyses. *US Geological Survey Water-Supply Paper* 1543-A.
- Gneiting, T., and Ranjan, R. (2011). Comparing density forecasts using threshold- and quantile-weighted scoring rules. *Journal of Business and Economic Statistics*, 29, 411-422.
- Hosking, J. R. M., and Wallis, J. R. (1987). Parameter and quantile estimation for the generalized Pareto distribution. *Technometrics*, 29, 339-349.
- Lerch, S., Thorarinsdottir, T. L., Ravazzolo, F., and Gneiting, T. (2017). Forecaster's dilemma: extreme events and forecast evaluation. *Statistical Science*, 32, 106-127.
- Pickands, J. (1975). Statistical inference using extreme order statistics. *Annals of Statistics*, 3, 119-131.
