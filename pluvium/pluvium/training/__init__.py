# Copyright 2026 Nathan. Apache-2.0.
"""
How the network is taught, beyond the loss:

* :mod:`.curriculum` -- heavy rain is 0.2% of hours and thunder a few percent: uniform sampling barely shows them
  to the network (the version 2 benchmark's one loss was on heavy hours). Issue times are drawn with weights that
  favour windows with heavy rain and thunder, and tempered importance weights keep the loss close to unbiased;
* :mod:`.ema`        -- an exponential moving average of the weights, used for evaluation and release (smoother,
  better-generalising weights at no training cost);
* :mod:`.augment`    -- perturbations with a physical meaning: gaps in the satellite field, errors in airport
  reports, small moisture biases in the driving model;
* :mod:`.distill`    -- teaching a small model to forecast like the large one, for phones and CPUs;
* :mod:`.fit`        -- sizing a training step to the GPU's memory: the largest batch that fits, gradient
  checkpointing only where it pays, and more target cells per forecast with what is left.
"""

from .augment import Augmenter
from .curriculum import RareEventSampler
from .distill import distill_step
from .ema import EMA
from .fit import Fit, Probe, fill

__all__ = ["Augmenter", "EMA", "Fit", "Probe", "RareEventSampler", "distill_step", "fill"]
