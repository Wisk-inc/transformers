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


__all__ = ["impact_figure"]
