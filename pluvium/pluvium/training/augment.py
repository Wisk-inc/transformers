# Copyright 2026 Nathan. Apache-2.0.
"""
Perturbations with a physical meaning, applied to training examples.

The model will meet imperfect inputs live; training on perfect ones teaches it to trust them too much. Three
perturbations, each mimicking a real failure:

* **satellite gaps** -- GOES and CMORPH have missing scans and limb and sun-glint gaps: one to three disks of 80-250 km
  in the observed rain field (the advected rain's source) are blanked;
* **report errors** -- an airport observer calls light rain moderate, or misses drizzle: a small share of known
  reports have their intensity class moved by one;
* **moisture bias** -- forecast models are systematically too moist or too dry by a few percent: total column water
  vapour and the specific humidities of a whole example are scaled by a factor near 1.
"""

from __future__ import annotations

import numpy as np

from ..util import great_circle_km


class Augmenter:
    def __init__(self, gap_chance: float = 0.3, report_noise: float = 0.03, moisture_sd: float = 0.04) -> None:
        self.gap_chance, self.report_noise, self.moisture_sd = gap_chance, report_noise, moisture_sd

    def satellite_gaps(self, values: np.ndarray, lat: np.ndarray, lon: np.ndarray, rng) -> np.ndarray:
        if values is None or rng.random() > self.gap_chance:
            return values
        out = values.copy()
        for _ in range(int(rng.integers(1, 4))):
            k = int(rng.integers(0, lat.size))
            radius = float(rng.uniform(80.0, 250.0))
            out[great_circle_km(lat, lon, lat[k], lon[k]) < radius] = np.nan
        return out

    def report_errors(self, codes: np.ndarray, rng) -> np.ndarray:
        codes = codes.astype(np.int16).copy()
        known_wet = (codes >= 0) & ((codes & 3) > 0) & ((codes >> 5) & 1).astype(bool)
        flip = known_wet & (rng.random(codes.shape) < self.report_noise)
        level = codes & 3
        moved = np.clip(level + rng.choice([-1, 1], size=codes.shape), 1, 3)
        codes = np.where(flip, (codes & ~3) | moved, codes)
        return codes.astype(np.int8)

    def moisture(self, fields: dict, upper: dict, rng) -> tuple[dict, dict]:
        f = float(np.exp(rng.normal(0.0, self.moisture_sd)))
        fields = {k: (v * f if k == "tcwv" else v) for k, v in fields.items()}
        upper = {k: (v * f if k.startswith("q") else v) for k, v in upper.items()}
        return fields, upper


__all__ = ["Augmenter"]
