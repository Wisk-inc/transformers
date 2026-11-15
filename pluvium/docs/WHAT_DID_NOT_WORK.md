# What did not work

Every idea below was built and tested. Most were tested inside the training year (April 2024 to March 2025), so the
test year was not used to decide. A few failed on the test data itself, and those are marked. Numbers are Brier skill
scores unless said otherwise. Higher is better for skill, lower for errors.

## Pluvium Blend

**More features for heavy rain.** Each system's median, maximum, neighbourhood maximum and the share of its
neighbourhood with 25 mm were added to the networks' inputs. No consistent gain at 25 or 50 mm in four regions.
Not adopted.

**Weights for every threshold.** The pool has one set of weights per band of thresholds (light, moderate, heavy).
Giving each of the 16 thresholds its own, tied to its neighbours by a smoothness penalty, changed nothing: in the
Alps 0.3084 against 0.3086 at 1 mm, and at 50 mm on days 1-3 it was lower (0.149 against 0.154). Three bands stayed.

**Weights that follow the season.** Letting the pool's weights swing with the time of year made it worse in
cross-validation: in the Alps 0.290 against 0.308 at 1 mm, and 0.085 against 0.113 on days 8-14. One year of
training cannot tell a season from that year's weather. The online weights do the job instead, from what has
verified.

**Weights that follow the regime.** Gates built from how wet and how split the components are helped the tail a
little (Alps, 50 mm, days 1-3: 0.161 against 0.149) and cost a little at 1 mm (0.3072 against 0.3084). Not adopted.

**Equal trust for every expert in the online layer.** Starting the online weights with no preference for the pool
was better at 1 mm in all four regions tested and slightly worse at 50 mm in three. Because the heavy tail was the
weak spot being fixed, the pool kept half the prior weight.

**A separate forecast of the expected amount.** A non-negative blend of the pool's amount, the components' amounts,
the climatology and the raw consensus, fitted for squared error. Gains of at most 0.7% in RMSE on days 1-7 and none
beyond. Not adopted. The code stays in `pluvium.pool.amount` for anyone who wants to try it on other data.

**Removing the climate's pattern from the expected amount.** At long range in the Alps the expected amount carries
about 1.2 times the climatological pattern, which dilutes the anomaly correlation. Taking the excess out raised the
correlation on days 11-14 from 0.21 to 0.27 and raised the RMSE by 0.4%. That trades one score for another, so it
was not adopted. The long-range anomaly correlation in mid-latitudes remains a weak spot (see the benchmark).

**Place-by-place offsets of the amount.** No effect on the anomaly correlation (0.206 against 0.211).

**A smooth gate for record storms.** The first hand-over used a sigmoid, which is never exactly zero: it touched
16% of ordinary training rows. It is now a hinge that is exactly zero inside the range of the training forecasts.

**Handing every record storm to the raw systems (test data).** The design written down before the test year handed
over everywhere. In the two mountain regions that was wrong. On days 1-3 at 50 mm the Alps scored 0.080 with the
hand-over and 0.091 without, and the US Pacific Northwest 0.288 against 0.307. The raw systems overcall heavy rain
over mountains against a satellite truth, so the calibration was right to discount them. The staged-storm experiment
had been run on two regions before the test, and both were regions where the hand-over helps. Run on all seven, it
shows the problem in training data: with the Alps' four largest storms withheld, the hand-over made the 50 mm score
worse (0.0781 against 0.0770). The remedy is the audit described in the benchmark: the hand-over is applied to the
training period's own out-of-fold forecasts and used only where it does no harm there.

**Discounting the raw systems before handing over.** A gentler remedy was tried first: scale the raw chances by how
often they verified at the top of the training forecasts, and never let the hand-over lower a chance. On the withheld
storms it still did harm in the Alps (0.0776 against 0.0770 calibrated) and gave away most of the gain in the
Southern Andes (0.0819 against 0.0782 for the plain hand-over) and part of it in the Great Plains, the Caribbean
and Jamaica. A switch decided by an audit is simpler and did better. The code stays in `pluvium.tails.override`
(`tail_trust`) and is not used.

**The hand-over is not free even where it is kept.** Its effect on the test year is shown part by part in the
benchmark ("What each part added"), including the scores where it cost something (test data).

## Pluvium Cyclone

**Correcting each system's bias.** Learned from one season and applied to the next, and learned online. Best
learner in two of four seasons, worst in the other two, 11% worse than a plain mean in one. A track bias is mostly
a property of the season's storms.

**Weights by gradient steps.** Exponentiated-gradient weights on the consensus error were behind weights from the
running error covariance in six of eight comparisons, and need a step size.

**Choosing settings on one season.** With 81 forecasts from 17 storms, the Atlantic 2024 season alone picked the
bias-corrected learner. Differences of 2% between learners at that sample size are noise.

**Single runs as points in the cloud.** With consensus weights on one track per deterministic run, a third of the
chance sat on a single line. Passage chances scored 0.76 and 0.77 on the two Atlantic seasons, below the raw member
cloud (0.79 and 0.78). Giving each system's track the ensembles' perturbations raised that to 0.82 and 0.81.

**A cone from a fixed prior.** The first cone held the centre 72% to 88% of the time against a target of 67%,
because long leads verify last and stayed on the prior. Rescaling the prior by what short leads show brought the
test season to 66% to 72%.

**Scenarios as single members.** The member nearest the middle of a group is a worse forecast than the consensus.
Scenarios are now the mean track of their group. They were flagged in only 5 of 151 test forecasts, which is too
few to judge them.

**Beating the official forecast.** Ahead at every lead on the Atlantic 2025 development season. Behind at every
lead on the East Pacific 2025 test season (test data).

## The lesson that repeats

Four times a choice that looked right on a small or already-seen sample did not hold on new data: season gates,
bias correction, the Atlantic win over the official forecast, and a hand-over tested on two regions that both
happened to like it. In each case the fix was the same. Decide on data kept apart, write the decision down, then
look once. And where a rule can be checked on every place's own training data, let each place check it.
