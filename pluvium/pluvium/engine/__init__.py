# Copyright 2026 Nathan. Apache-2.0.
"""
Neural components of the version 3 engine, each aimed at a weakness of the version 2 decoder:

* :mod:`.refiner`        -- neighbouring towns were decoded independently, so two villages 3 km apart could get
  inconsistent hours; local attention among decoded points makes the field coherent and sharpens structure finer
  than the mesh (valleys, coasts);
* :mod:`.temporal_mixer` -- each hour was decoded from its own frame only; attention along a point's 40 future
  frames lets "the front arrives at hour 30" inform hours 28-32, removing hour-to-hour jitter in timing;
* :mod:`.lead_film`      -- skill falls with lead time; feature-wise modulation by lead, and a learned shrinkage
  of the transition logits towards climatology as lead grows, keep long-range probabilities honest;
* :mod:`.experts`        -- one head served fronts, tropical waves and afternoon storms alike; a soft mixture of
  experts lets each regime have its own mapping;
* :mod:`.terrain_patch`  -- thirteen numbers described each place's terrain; a small convolutional encoder reads a
  35 km elevation and coastline patch around every point;
* :mod:`.mc_heads`       -- a single network is overconfident where it has seen little; dropout before the heads,
  sampled at forecast time, gives epistemic uncertainty that the mixture of chains carries into every product.
"""

from .experts import RegimeExperts
from .lead_film import LeadFiLM, LeadShrink
from .mc_heads import HeadDropout
from .refiner import PointRefiner, knn_on_sphere
from .temporal_mixer import TemporalMixer
from .terrain_patch import TerrainPatchEncoder, terrain_patches

__all__ = ["HeadDropout", "LeadFiLM", "LeadShrink", "PointRefiner", "RegimeExperts", "TemporalMixer",
           "TerrainPatchEncoder", "knn_on_sphere", "terrain_patches"]
