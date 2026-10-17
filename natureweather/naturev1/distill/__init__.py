# Copyright 2026 Nathan. Apache-2.0.
"""
Multi-teacher distillation into NatureV1: Pangu-Weather, GraphCast, IFS HRES, Aurora, FGN (WeatherNext 2), FuXi,
GenCast, NeuralGCM from archives, AIFS from ECMWF open data, Earth-2 models run locally.

The teachers disagree on what a grid, a level and a step are, so nothing is compared until it has been
resampled into NatureV1's own analysis space:

==============  ==================================================================================================
catalog         every teacher's grid, levels, steps, years, archive paths, naming quirks and licence
align           conservative regridding, reduced-Gaussian / mesh binning, log-pressure levels, time alignment,
                names and units, NatureV1's channel layout and normalisation
sources         readers that return aligned fields: WeatherBench 2 archives, ECMWF open data, Earth2Studio
targets         skill-weighted multi-teacher mixture with an honest variance (ensemble spread + disagreement)
stage           aligned teacher forecasts staged on disk as memory-mapped arrays, resumable
train           rollout distillation loss (teacher mixture + ERA5 anchor + hurricane focus), curriculum, trainer
run             the command line: stage, then train, on one GPU or many
==============  ==================================================================================================
"""

from .catalog import ARCHIVED, TEACHERS, Teacher

__all__ = ["ARCHIVED", "TEACHERS", "Teacher"]
