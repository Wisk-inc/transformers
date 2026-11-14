# Copyright 2026 Nathan. Apache-2.0.
"""
Flood risk: rain is only half of it. The other half is where the water goes.

A town at the mouth of a valley floods from rain that fell on the hills above it, hours earlier; a town on a ridge
does not flood from the same storm. So each town gets, besides its own rain:

* its **catchment**: the cells of the terrain raster that drain through it, from D8 flow directions on the
  elevation (each cell drains to its steepest-descent neighbour, flats and pits filled first). The forecast rain over
  the catchment (the area-weighted mean of the nearby 8 km cells' totals) is what reaches the town;
* its **steepness**: steep catchments answer in an hour or two (flash floods), flat ones in a day;
* its **wetness before the storm**: the rain of the last days, if known (an antecedent precipitation index), since a
  soaked hillside sheds nearly all of the next storm's water.

The flash-flood level is a rule, not a trained model: the probability of short bursts (1, 3, 6 hours) and of 24-hour
totals beyond thresholds commonly used for heavy rain in the tropics (WMO's 50 mm in 24 hours; 75-150 mm storm totals
in Jamaica's flash-flood watches), raised for steep, wet, large catchments. It needs verification against flood
reports before anyone relies on it, and the forecast says so.
"""

from __future__ import annotations

import numpy as np

LEVELS = ("low", "moderate", "high", "very high")
#: (window, threshold mm, probability for each level above "low")
CRITERIA = (("1h", 40.0, (0.15, 0.35, 0.6)), ("3h", 60.0, (0.15, 0.35, 0.6)), ("6h", 75.0, (0.15, 0.3, 0.5)),
            ("24h", 100.0, (0.2, 0.4, 0.6)), ("24h", 50.0, (0.45, 0.7, 0.9)))
D8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def fill_depressions(elev: np.ndarray, land: np.ndarray, iterations: int = 200) -> np.ndarray:
    """Raise pits so every land cell drains to the sea (iterative minimum filter on the sea-anchored surface)."""
    from scipy import ndimage

    z = np.where(land, np.inf, elev)
    z[~land] = elev[~land]
    eps = 1e-3
    for _ in range(iterations):
        lowest = ndimage.grey_erosion(z, size=(3, 3), mode="nearest")
        new = np.where(land, np.maximum(elev, np.minimum(z, lowest + eps)), z)
        if np.allclose(new, z):
            break
        z = new
    return np.where(np.isfinite(z), z, elev)


def flow_network(elev: np.ndarray, land: np.ndarray) -> tuple[np.ndarray, list]:
    """D8 receivers (flat index, -1 for outlets to the sea) on the depression-filled surface, and the cells grouped
    in topological levels (every cell comes after all the cells that drain into it)."""
    h, w = elev.shape
    z = fill_depressions(elev, land)
    best = np.full((h, w), -1, np.int64)
    drop = np.zeros((h, w))
    for di, dj in D8:
        shifted = np.full((h, w), np.inf)
        si = slice(max(0, -di), h - max(0, di))
        sj = slice(max(0, -dj), w - max(0, dj))
        ti = slice(max(0, di), h - max(0, -di))
        tj = slice(max(0, dj), w - max(0, -dj))
        shifted[si, sj] = z[ti, tj]
        slope = (z - shifted) / (np.hypot(di, dj))
        better = (slope > drop) & land
        drop = np.where(better, slope, drop)
        idx = (np.arange(h)[:, None] + di) * w + (np.arange(w)[None, :] + dj)
        best = np.where(better, idx, best)
    receiver = best.ravel()
    n = receiver.size
    has = receiver >= 0
    indegree = np.bincount(receiver[has], minlength=n)
    frontier = np.flatnonzero(indegree == 0)
    levels = []
    while frontier.size:
        levels.append(frontier)
        rcv = receiver[frontier]
        rcv = rcv[rcv >= 0]
        if rcv.size == 0:
            break
        indegree -= np.bincount(rcv, minlength=n)
        cand = np.unique(rcv)
        frontier = cand[indegree[cand] == 0]
    return receiver, levels


def accumulate(values: np.ndarray, receiver: np.ndarray, levels: list) -> np.ndarray:
    """Sum of ``values`` (``(C,)`` or ``(B, C)``) over each cell and everything upstream of it."""
    acc = np.array(values, np.float64, copy=True)
    flat = acc.reshape(-1, acc.shape[-1])
    for frontier in levels:
        rcv = receiver[frontier]
        ok = rcv >= 0
        if ok.any():
            for row in flat:
                np.add.at(row, rcv[ok], row[frontier[ok]])
    return flat.reshape(acc.shape)


class Catchments:
    """
    For each town: the river cell it sits by (the largest upstream area within ``search_km``), that cell's upstream
    area and mean slope, and :meth:`catchment_mean` -- the area-weighted mean over everything upstream of it of a
    field forecast on the 8 km target cells (nested catchments are handled: a town downstream of another gets both).
    """

    def __init__(self, terrain, lat: np.ndarray, lon: np.ndarray, target_lat: np.ndarray, target_lon: np.ndarray,
                 search_km: float = 2.0, stride: int = 2) -> None:
        from scipy.spatial import cKDTree

        elev = terrain.elevation[::stride, ::stride].astype(np.float64)
        land = terrain.land[::stride, ::stride]
        tlat, tlon = terrain.lat[::stride], terrain.lon[::stride]
        km = 111.195 * abs(tlat[1] - tlat[0])
        area = (km * km * np.cos(np.radians(tlat))[:, None] * np.ones((1, tlon.size))) * land
        self.receiver, self.levels = flow_network(elev, land)
        self.area = area.ravel()
        self.acc_area = accumulate(self.area, self.receiver, self.levels)
        h, w = elev.shape
        ii = np.clip(np.searchsorted(tlat, lat), 0, h - 1) if tlat[1] > tlat[0] else np.clip(h - 1 - np.searchsorted(tlat[::-1], lat), 0, h - 1)
        jj = np.clip(np.searchsorted(tlon, lon), 0, w - 1)
        r = max(1, int(round(search_km / km)))
        acc2 = self.acc_area.reshape(h, w)
        self.outlet = np.zeros(np.size(lat), np.int64)
        for k, (i, j) in enumerate(zip(ii, jj)):
            a, b = slice(max(0, i - r), i + r + 1), slice(max(0, j - r), j + r + 1)
            win = np.where(land[a, b], acc2[a, b], -1)
            di, dj = np.unravel_index(int(np.argmax(win)), win.shape)
            self.outlet[k] = (a.start + di) * w + (b.start + dj)
        self.upstream_km2 = self.acc_area[self.outlet]
        gy, gx = np.gradient(elev, km * 1000.0)
        slope = np.hypot(gx, gy).ravel()
        self.slope = accumulate(slope * self.area, self.receiver, self.levels)[self.outlet] / np.maximum(
            self.upstream_km2, 1e-9)
        la, lo = np.meshgrid(tlat, tlon, indexing="ij")
        _, self.near = cKDTree(np.stack([target_lat, target_lon], -1)).query(np.stack([la.ravel(), lo.ravel()], -1))

    def catchment_mean(self, field: np.ndarray) -> np.ndarray:
        """Area-weighted mean over each town's catchment of a per-target-cell field ``(..., N_targets)``."""
        per_cell = np.asarray(field)[..., self.near] * self.area
        acc = accumulate(per_cell, self.receiver, self.levels)
        return acc[..., self.outlet] / np.maximum(self.upstream_km2, 1e-9)


def flood_risk(bursts: dict, totals24: dict, catchment24: np.ndarray | None = None, slope: np.ndarray | None = None,
               upstream_km2: np.ndarray | None = None, antecedent_mm: np.ndarray | None = None) -> dict:
    """
    Flood level per town (index into :data:`LEVELS`) and the reason, from the burst and total distributions of
    :func:`pluvium.products.summarize` (probabilities of exceeding thresholds), the catchment's forecast 24-hour rain
    (median, mm), its slope, size and the rain of the days before.
    """
    n = totals24["median"].shape[0]
    level = np.zeros(n, np.int64)
    reason = np.array([""] * n, dtype=object)
    for window, threshold, steps in CRITERIA:
        dist = bursts.get(int(window[:-1])) if window != "24h" else totals24
        if dist is None:
            continue
        key = f"p_ge_{int(threshold)}"
        p = dist.get(key)
        if p is None:
            p = (dist["p90"] >= threshold) * 0.1 + (dist["median"] >= threshold) * 0.5
        lv = np.searchsorted(np.array(steps), p, side="right")
        better = lv > level
        level = np.where(better, lv, level)
        text = np.char.add(np.char.mod("%.0f%% chance of ", 100 * np.asarray(p, np.float64)),
                           f"{threshold:.0f} mm in {window}")
        reason = np.where(better, text, reason)
    boost = np.zeros(n)
    if slope is not None:
        boost += (slope > 0.15).astype(float)                         # steep: water arrives within hours
    if upstream_km2 is not None and catchment24 is not None:
        boost += ((upstream_km2 > 50.0) & (catchment24 >= 50.0)).astype(float)
    if antecedent_mm is not None:
        boost += (antecedent_mm > 75.0).astype(float)
    raised = (level >= 1) & (boost >= 2)
    level = np.where(raised, np.minimum(level + 1, len(LEVELS) - 1), level)
    reason = np.where(raised, np.char.add(reason.astype(str), "; steep, soaked or large catchment"), reason)
    return {"level": level, "text": np.array(LEVELS)[level], "reason": reason}


__all__ = ["CRITERIA", "Catchments", "LEVELS", "accumulate", "fill_depressions", "flood_risk", "flow_network"]
