# Copyright 2026 Nathan. Apache-2.0.
"""
Scoring forecasts against what fell, and against the forecasts a person could have used instead.

Every score is computed the same way for Pluvium and for three references, so "skill" always means "better than
this simpler answer":

* **climatology** -- the chance of a wet hour for that place, month and local hour (from the training years);
* **persistence** -- "it will keep doing what it was doing in the last hour";
* **the driving model's own rain** -- wet when the coarse model (ERA5 in hindcasts, GFS/ECMWF live) puts at least
  0.2 mm in the hour at that point: the raw guidance Pluvium is meant to improve on.

Scores: Brier score of the hourly chance of rain (and its skill score against each reference), Brier score of
"any rain in the next 6/12/24 h", the critical success index of yes/no at 50%, the error of the forecast start
hour, and the continuous ranked probability score of 24-hour totals (from the sampled rain paths).
"""

from __future__ import annotations

import numpy as np

from .calibrate import LEAD_BANDS, reliability
from .dataset import WET_MM

WINDOWS = (6, 12, 24)


class Scoreboard:
    """Accumulates forecasts and outcomes; :meth:`report` gives every score."""

    def __init__(self) -> None:
        self.rows = {"p": [], "clim": [], "persist": [], "driving": [], "wet": [], "known": []}
        self.windows = {h: {"p": [], "clim": [], "persist": [], "driving": [], "wet": []} for h in WINDOWS}
        self.onset = {"pred": [], "obs": []}
        self.crps = {"model": [], "clim": []}
        self.intensity = {"rps": [], "rps_clim": [], "heavy_p": [], "heavy_clim": [], "heavy_obs": []}

    def add(self, p_wet: np.ndarray, truth: np.ndarray, clim: np.ndarray, last_hour: np.ndarray,
            driving: np.ndarray, any_rain: dict | None = None, start_median: np.ndarray | None = None,
            totals24: np.ndarray | None = None, clim_rate: np.ndarray | None = None, classes: np.ndarray | None = None,
            clim_classes: np.ndarray | None = None) -> None:
        """One forecast over points ``P`` and hours ``T``: ``p_wet (P, T)``, ``truth (P, T)`` mm/h (NaN unknown),
        ``clim (P, T)`` climatological wet chance, ``last_hour (P,)`` mm/h observed in the hour before (NaN unknown),
        ``driving (P, T)`` driving-model rain mm/h, ``any_rain {h: (P,)}``, ``start_median (P,)`` (hours),
        ``totals24 (S, P)`` sampled 24-hour totals, ``clim_rate (P, T)`` climatological mean rate."""
        known = np.isfinite(truth)
        wet = (np.nan_to_num(truth) >= WET_MM).astype(np.float32)
        if classes is not None and clim_classes is not None:
            from .weather import rate_class

            obs = rate_class(truth)                                            # (P, T), -1 unknown
            ok = obs >= 0
            onehot = np.eye(4)[np.clip(obs, 0, 3)]
            cdf_o = np.cumsum(onehot, -1)[..., :3]
            rps = ((np.cumsum(classes, -1)[..., :3] - cdf_o) ** 2).sum(-1) / 3.0
            rps_c = ((np.cumsum(clim_classes, -1)[..., :3] - cdf_o) ** 2).sum(-1) / 3.0
            self.intensity["rps"].append(rps[ok])
            self.intensity["rps_clim"].append(rps_c[ok])
            self.intensity["heavy_p"].append(classes[..., 3][ok])
            self.intensity["heavy_clim"].append(clim_classes[..., 3][ok])
            self.intensity["heavy_obs"].append((obs[ok] == 3).astype(np.float32))
        persist = np.broadcast_to((np.nan_to_num(last_hour) >= WET_MM).astype(np.float32)[:, None], wet.shape)
        persist_ok = np.isfinite(last_hour)[:, None] & known
        self.rows["p"].append(p_wet)
        self.rows["clim"].append(clim)
        self.rows["persist"].append(np.where(persist_ok, persist, clim))
        self.rows["driving"].append((driving >= WET_MM).astype(np.float32))
        self.rows["wet"].append(wet)
        self.rows["known"].append(known)
        # windows: "any rain in the next h hours", only where every hour of the window is known
        for h in WINDOWS:
            if h > wet.shape[1]:
                continue
            ok = known[:, :h].all(1)
            obs = wet[:, :h].max(1)
            p_clim = 1 - np.prod(1 - clim[:, :h], 1)       # independent hours: an upper bound, the honest naive answer
            d = (driving[:, :h] >= WET_MM).any(1).astype(np.float32)
            pr = (np.nan_to_num(last_hour) >= WET_MM).astype(np.float32)
            w = self.windows[h]
            w["p"].append((any_rain or {}).get(h, 1 - np.prod(1 - p_wet[:, :h], 1))[ok])
            w["clim"].append(p_clim[ok])
            w["persist"].append(np.where(np.isfinite(last_hour), pr, p_clim)[ok])
            w["driving"].append(d[ok])
            w["wet"].append(obs[ok])
        # onset: dry at t0 (known), rain starts within 24 h
        if start_median is not None:
            dry0 = known[:, 0] & (wet[:, 0] == 0)
            first = np.where(known[:, :24].all(1) & (wet[:, :24].max(1) > 0), wet[:, :24].argmax(1), -1)
            m = dry0 & (first > 0) & np.isfinite(start_median)
            self.onset["pred"].append(start_median[m])
            self.onset["obs"].append(first[m].astype(np.float64))
        # 24-hour totals
        if totals24 is not None and truth.shape[1] >= 24:
            ok = known[:, :24].all(1)
            y = np.nan_to_num(truth[:, :24]).sum(1)
            self.crps["model"].append(crps_samples(totals24[:, ok], y[ok]))
            if clim_rate is not None:
                c = (clim[:, :24] * 0 + clim_rate[:, :24]).sum(1)
                self.crps["clim"].append(np.abs(c[ok] - y[ok]))       # a point forecast's CRPS is its absolute error

    # ------------------------------------------------------------------------------------- report --
    def report(self) -> dict:
        if not self.rows["p"]:
            return {}
        cat = {k: np.concatenate([x.reshape(x.shape[0], -1) for x in v], 0) for k, v in self.rows.items()}
        known = cat["known"]
        out = {"hours_scored": int(known.sum()), "wet_share": float(cat["wet"][known].mean())}
        T = cat["p"].shape[1]

        def brier(p, mask):
            return float(np.mean((p[mask] - cat["wet"][mask]) ** 2)) if mask.any() else float("nan")

        for name in ("p", "clim", "persist", "driving"):
            out[f"brier_{name}"] = brier(cat[name], known)
        for ref in ("clim", "persist", "driving"):
            out[f"bss_vs_{ref}"] = 1 - out["brier_p"] / max(out[f"brier_{ref}"], 1e-12)
        bands = {}
        for a, b in LEAD_BANDS:
            if a >= T:
                continue
            m = np.zeros_like(known)
            m[:, a:b] = known[:, a:b]
            bands[f"{a}-{b}h"] = {"brier": brier(cat["p"], m), "brier_clim": brier(cat["clim"], m),
                                  "bss_vs_clim": 1 - brier(cat["p"], m) / max(brier(cat["clim"], m), 1e-12)}
        out["by_lead"] = bands
        obs = (cat["wet"] > 0) & known
        for level in (0.5, 0.3):
            yes = (cat["p"] >= level) & known
            hits, misses, false = (yes & obs).sum(), (~yes & obs).sum(), (yes & ~obs).sum()
            tag = int(level * 100)
            out[f"csi_{tag}"] = float(hits / max(hits + misses + false, 1))
            out[f"pod_{tag}"] = float(hits / max(hits + misses, 1))
            out[f"far_{tag}"] = float(false / max(hits + false, 1))
        out["reliability"] = reliability(cat["p"][known], cat["wet"][known])
        for h, w in self.windows.items():
            if not w["p"]:
                continue
            c = {k: np.concatenate(v) for k, v in w.items()}
            if c["wet"].size == 0:
                continue
            row = {"n": int(c["wet"].size), "wet_share": float(c["wet"].mean())}
            for name in ("p", "clim", "persist", "driving"):
                row[f"brier_{name}"] = float(np.mean((c[name] - c["wet"]) ** 2))
            row["bss_vs_clim"] = 1 - row["brier_p"] / max(row["brier_clim"], 1e-12)
            row["bss_vs_driving"] = 1 - row["brier_p"] / max(row["brier_driving"], 1e-12)
            out[f"any_rain_{h}h"] = row
        if self.onset["pred"]:
            pred, obs_h = np.concatenate(self.onset["pred"]), np.concatenate(self.onset["obs"])
            if pred.size:
                err = np.abs(pred - obs_h)
                out["start_hour_error"] = {"n": int(err.size), "median_h": float(np.median(err)),
                                           "within_2h": float((err <= 2).mean()), "within_3h": float((err <= 3).mean())}
        if self.intensity["rps"]:
            c = {k: np.concatenate(v) for k, v in self.intensity.items()}
            out["rps"] = float(c["rps"].mean())
            out["rps_clim"] = float(c["rps_clim"].mean())
            out["rpss_vs_clim"] = 1 - out["rps"] / max(out["rps_clim"], 1e-12)
            hb = float(np.mean((c["heavy_p"] - c["heavy_obs"]) ** 2))
            hc = float(np.mean((c["heavy_clim"] - c["heavy_obs"]) ** 2))
            out["heavy"] = {"brier": hb, "brier_clim": hc, "bss_vs_clim": 1 - hb / max(hc, 1e-12),
                            "share": float(c["heavy_obs"].mean())}
        if self.crps["model"]:
            m = np.concatenate(self.crps["model"])
            out["crps_24h_mm"] = float(m.mean())
            if self.crps["clim"]:
                c = np.concatenate(self.crps["clim"])
                out["crps_24h_clim_mm"] = float(c.mean())
                out["crpss_24h_vs_clim"] = 1 - out["crps_24h_mm"] / max(out["crps_24h_clim_mm"], 1e-12)
        return out


def crps_samples(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS of an ensemble ``(S, N)`` for outcomes ``(N,)`` (the energy form, sorted for O(S log S))."""
    s = np.sort(samples, 0)
    n = s.shape[0]
    term1 = np.abs(s - y[None]).mean(0)
    k = np.arange(1, n + 1)[:, None]
    term2 = ((2 * k - n - 1) * s).sum(0) / (n * n)       # E|X - X'| / 2
    return term1 - term2


def format_report(r: dict) -> str:
    """The scores as a few readable lines."""
    if not r:
        return "no forecasts scored"
    lines = [f"hours scored {r['hours_scored']:,}  (wet {r['wet_share']:.1%})",
             f"hourly Brier {r['brier_p']:.4f}   climatology {r['brier_clim']:.4f}   persistence {r['brier_persist']:.4f}"
             f"   driving rain {r['brier_driving']:.4f}",
             f"skill vs climatology {r['bss_vs_clim']:+.1%}   vs persistence {r['bss_vs_persist']:+.1%}"
             f"   vs driving rain {r['bss_vs_driving']:+.1%}",
             f"yes/no at 50%: CSI {r['csi_50']:.3f}  hit rate {r['pod_50']:.1%}  false alarms {r['far_50']:.1%};  at 30%: "
             f"CSI {r['csi_30']:.3f}  hit rate {r['pod_30']:.1%}  false alarms {r['far_30']:.1%}"]
    for band, b in r.get("by_lead", {}).items():
        lines.append(f"  lead {band:>6}: Brier {b['brier']:.4f} (climatology {b['brier_clim']:.4f}, skill "
                     f"{b['bss_vs_clim']:+.1%})")
    for h in WINDOWS:
        w = r.get(f"any_rain_{h}h")
        if w:
            lines.append(f"any rain in {h:>2} h: Brier {w['brier_p']:.4f}  climatology {w['brier_clim']:.4f}  driving "
                         f"{w['brier_driving']:.4f}  skill vs climatology {w['bss_vs_clim']:+.1%}")
    if "start_hour_error" in r:
        s = r["start_hour_error"]
        lines.append(f"start hour: median error {s['median_h']:.1f} h, within 2 h {s['within_2h']:.0%}, within 3 h "
                     f"{s['within_3h']:.0%} ({s['n']:,} rain starts)")
    if "rps" in r:
        lines.append(f"intensity (dry/light/moderate/heavy): RPS {r['rps']:.4f}  climatology {r['rps_clim']:.4f}  "
                     f"skill {r['rpss_vs_clim']:+.1%}")
        h = r["heavy"]
        lines.append(f"heavy hours (>= 7.6 mm/h, {h['share']:.2%} of hours): Brier {h['brier']:.5f}  climatology "
                     f"{h['brier_clim']:.5f}  skill {h['bss_vs_clim']:+.1%}")
    if "crps_24h_mm" in r:
        lines.append(f"24 h totals: CRPS {r['crps_24h_mm']:.2f} mm" + (
            f" (climatology {r['crps_24h_clim_mm']:.2f}, skill {r['crpss_24h_vs_clim']:+.1%})"
            if "crps_24h_clim_mm" in r else ""))
    return "\n".join(lines)


__all__ = ["Scoreboard", "crps_samples", "format_report"]
