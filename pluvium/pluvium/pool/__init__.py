# Copyright 2026 Nathan. Apache-2.0.
"""
Pools: how Pluvium Blend's components become one forecast.

* :mod:`.gated` is the batch pool: a weighted logarithmic pool whose weights depend on the threshold, the lead and,
  through gates, on the season and the weather regime (:mod:`.gates`);
* :mod:`.online` re-weighs the experts after every verified day, with a guarantee against the best expert;
* :mod:`.cv` scores a design inside the training year by leaving out one month at a time, so designs are chosen
  without the test year.
"""

from .cv import brier_skill, leave_month_out, summarise
from .gated import GatedPool, fit_pool, predict_pool
from .gates import build_gates, gate_names
from .online import OnlineAggregator, run_online

__all__ = ["GatedPool", "OnlineAggregator", "brier_skill", "build_gates", "fit_pool", "gate_names", "leave_month_out",
           "predict_pool", "run_online", "summarise"]
