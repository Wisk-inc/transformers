# Copyright 2026 Nathan. Apache-2.0.
"""Maps of a swarm forecast and its impacts (needs ``matplotlib``)."""

from __future__ import annotations

import numpy as np


def impact_figure(run: dict, worldlines: int = 300, scene=None, extent=None):
    """
    Two (or three) panels for a :func:`naturev1.swarm.live.forecast_storm` run: the probability of hurricane-force
    wind per cell with the consensus track, the cone, a sample of the swarm's worldlines (coloured by steering depth)
    and the satellite fixes; the expected rain; and, if ``scene`` (a GOES scene) is given, the infrared image now.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    impact, result, case = run["impact"], run["result"], run["case"]
    region = impact.region if impact is not None else None
    fc = result["swarm"]["forecast"]
    panels = 2 + (scene is not None)
    fig, axes = plt.subplots(1, panels, figsize=(7.5 * panels, 6.2), constrained_layout=True)
    track = np.vstack([[case.lat, case.lon], result["track"]])
    if extent is None:
        lat_all = np.concatenate([track[:, 0], fc.positions[:, -1, 0]])
        lon_all = np.concatenate([track[:, 1], fc.positions[:, -1, 1]])
        extent = (np.nanpercentile(lon_all, 2) - 4, np.nanpercentile(lon_all, 98) + 4,
                  np.nanpercentile(lat_all, 2) - 3, np.nanpercentile(lat_all, 98) + 3)
    titles = ("P(hurricane-force wind)", "Expected storm-total rain (mm)", "GOES infrared now")
    for k, ax in enumerate(np.atleast_1d(axes)):
        ax.set_facecolor("#dfe8ee")
        if region is not None and k < 2:
            grid = impact.cell_maps["p64"] if k == 0 else impact.cell_maps["mean_rain_mm"]
            land = np.where(region.land, 1.0, np.nan)
            ax.pcolormesh(region.lon, region.lat, land, cmap="Greys", vmin=0, vmax=4, shading="auto")
            mesh = ax.pcolormesh(region.lon, region.lat, grid, cmap="YlOrRd" if k == 0 else "GnBu",
                                 vmin=0, vmax=1 if k == 0 else max(np.nanpercentile(grid, 99), 50), shading="auto")
            fig.colorbar(mesh, ax=ax, shrink=0.75)
        if k == 2 and scene is not None:
            sc = ax.scatter(scene.longitude, scene.latitude, c=scene.values[:, 0], s=1, cmap="Greys", vmin=190, vmax=300)
            fig.colorbar(sc, ax=ax, shrink=0.75, label="K")
        if fc.track is not None and k == 0:
            pick = np.random.default_rng(0).choice(fc.track.shape[0], min(worldlines, fc.track.shape[0]), replace=False,
                                                   p=fc.weights / fc.weights.sum())
            depth = fc.track_depth[pick] if fc.track_depth is not None else np.full(fc.track[pick].shape[:2], 0.5)
            for i, p in enumerate(pick):
                ax.plot(fc.track[p, :, 1], fc.track[p, :, 0], lw=0.4, alpha=0.25,
                        color=plt.cm.viridis(float(np.nanmean(depth[i]))))
        ax.plot(track[:, 1], track[:, 0], "-o", color="black", ms=3, lw=1.6, label="NatureV1 consensus")
        for j, (name, m) in enumerate(run.get("models", {}).items()):
            t = np.asarray(m["track"])
            if np.isfinite(t[:, 0]).any():
                ok = np.isfinite(t[:, 0])
                path = np.vstack([[case.lat, case.lon], t[ok]])
                ax.plot(path[:, 1], path[:, 0], "-", color=plt.cm.tab10(j % 10), lw=0.8, alpha=0.8, label=m.get("title", name))
        model = run.get("model_track")
        if model is not None and np.isfinite(model).any():
            ok = np.isfinite(model[:, 0])
            path = np.vstack([[case.lat, case.lon], model[ok]])
            ax.plot(path[:, 1], path[:, 0], "--", color="dimgray", lw=1.2, label="models' mean")
        cone = result["swarm"].get("cone_km")
        if cone is not None and k == 0:
            for (la, lo), r in zip(result["track"], cone):
                if np.isfinite(la):       # a true distance circle: wider in longitude by 1 / cos(latitude)
                    ax.add_patch(Ellipse((lo, la), 2 * r / 111.2 / np.cos(np.radians(la)), 2 * r / 111.2, fill=False,
                                         lw=0.6, ls="--", color="black", alpha=0.6))
        fixes = run.get("fixes_before", []) + run.get("fixes_after", [])
        if fixes:
            ax.scatter([f.lon for f in fixes], [f.lat for f in fixes], s=8, c="deepskyblue", edgecolor="navy", lw=0.3,
                       label="centre fixes", zorder=5)
        if np.isfinite(case.truth.get("lat", [np.nan])[0]) if len(case.truth.get("lat", [])) else False:
            ok = case.truth["valid"]
            ax.plot(case.truth["lon"][ok], case.truth["lat"][ok], "-", color="crimson", lw=1.2, label="best track (truth)")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect(1.0 / np.cos(np.radians(np.mean(extent[2:]))))
        ax.set_title(f"{titles[k]} -- {case.name or case.storm_id} from {str(run['analysis_time'])[:16]}Z", fontsize=10)
        if k == 0:
            ax.legend(loc="lower left", fontsize=7)
    return fig


def landfall_figure(run: dict, summary: dict, country: str = "Jamaica", extent=None, top: int = 14):
    """
    One country, parish by parish: peak sustained wind given a hit (colour), the consensus track with its date and hour
    every 12 hours, the landfall point, and a panel with when the centre comes ashore (median and 10-90%), the
    intensity then, the peak intensity, and the parishes ranked by peak wind with gust, rain and arrival times.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    impact, result, case = run["impact"], run["result"], run["case"]
    region = impact.region
    ids = [u for u, c in enumerate(region.countries) if c == country]
    by_unit = {u["unit"]: u for u in impact.units}
    value = np.full(len(region.names) + 1, np.nan)
    for u in ids:
        if u in by_unit:
            value[u] = by_unit[u].get("peak_kt_if_hit", by_unit[u]["peak_kt"])
    grid = np.where(region.unit >= 0, value[np.where(region.unit >= 0, region.unit, len(region.names))], np.nan)
    grid = np.where(np.isin(region.unit, ids), grid, np.nan)
    if extent is None:
        ii, jj = np.nonzero(np.isin(region.unit, ids))
        extent = (region.lon[jj.min()] - 1.2, region.lon[jj.max()] + 1.2, region.lat[ii.min()] - 1.0, region.lat[ii.max()] + 1.0)
    # the map on top at full width, the text panel below it: a wide parish table no longer squeezes the map
    fig = plt.figure(figsize=(15, 12.5))
    gs = GridSpec(2, 1, height_ratios=[1.0, 0.95], figure=fig, left=0.05, right=0.98, top=0.95, bottom=0.02, hspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor("#dfe8ee")
    land = np.where(region.land, 1.0, np.nan)
    ax.pcolormesh(region.lon, region.lat, land, cmap="Greys", vmin=0, vmax=4, shading="auto")
    mesh = ax.pcolormesh(region.lon, region.lat, grid, cmap="YlOrRd", vmin=34, vmax=140, shading="auto")
    fig.colorbar(mesh, ax=ax, shrink=0.85, pad=0.02, label="peak sustained wind if hit (kt)")
    ax.contour(region.lon, region.lat, np.where(region.land, region.unit, -1), levels=np.arange(-0.5, len(region.names)), colors="k", linewidths=0.3)
    t0 = np.datetime64(run["analysis_time"], "m")
    track = np.vstack([[case.lat, case.lon], result["track"]])
    hours = np.concatenate([[0.0], np.asarray(result["leads"], float)])
    ok = np.isfinite(track[:, 0])
    ax.plot(track[ok, 1], track[ok, 0], "-o", color="black", ms=3, lw=1.6, label="NatureV1 consensus")
    for h, (la, lo) in zip(hours[ok], track[ok]):
        if h % 12 == 0 and extent[0] <= lo <= extent[1] and extent[2] <= la <= extent[3]:
            when = t0 + np.timedelta64(int(h * 60), "m")
            ax.annotate(str(when)[5:16].replace("T", " ") + "Z", (lo, la), textcoords="offset points", xytext=(6, 4), fontsize=7)
    cons = summary.get("consensus") or {}
    if cons.get("time"):
        ax.plot(cons["lon"], cons["lat"], marker="*", color="crimson", ms=14, label=f"landfall {cons['time'][5:16]}")
    if len(case.truth.get("lat", [])) and np.isfinite(case.truth["lat"]).any():
        v = case.truth["valid"]
        ax.plot(case.truth["lon"][v], case.truth["lat"][v], "-", color="crimson", lw=1.2, label="best track (truth)")
    for u in ids:
        cells = np.nonzero(region.unit == u)
        if cells[0].size and u in by_unit:
            ax.text(float(region.lon[cells[1]].mean()), float(region.lat[cells[0]].mean()), region.names[u].replace("Saint", "St").split(" Parish")[0],
                    fontsize=7, ha="center", va="center", color="black")
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3]); ax.set_aspect(1.0 / np.cos(np.radians(np.mean(extent[2:]))))
    ax.set_title(f"{case.name or case.storm_id}: {country}, parish by parish -- forecast from {str(t0)[:16].replace('T', ' ')}Z", fontsize=12)
    ax.legend(loc="lower left", fontsize=8)
    panel = fig.add_subplot(gs[1, 0])
    panel.axis("off")
    from .landfall import format_landfall

    panel.text(0.0, 1.0, format_landfall(summary, top=top), family="monospace", fontsize=8.2, va="top", ha="left", transform=panel.transAxes)
    panel.text(0.0, 0.0, "Research model, not an operational forecast. Storm surge is not modelled.", fontsize=8, va="bottom", transform=panel.transAxes)
    return fig


__all__ = ["impact_figure", "landfall_figure"]

