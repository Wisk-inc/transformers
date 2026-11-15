# Copyright 2026 Nathan. Apache-2.0.
"""Track error by lead for the early guidance: Pluvium Cyclone against the National Hurricane Center's official
forecast and its two consensus aids, on the forecasts all four made. One panel per 2025 season.

    python scripts/chart_cyclone.py cyclone.json docs/img/cyclone_track_error.png
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = [("PLVI", "Pluvium Cyclone", "#2a78d6"), ("OFCL", "NHC official forecast", "#eb6834"),
          ("TVCN", "NHC consensus TVCN", "#1baf7a"), ("HCCA", "NHC corrected consensus HCCA", "#eda100")]
PANELS = [("East Pacific 2025", "East Pacific 2025: the test season"),
          ("North Atlantic 2025", "North Atlantic 2025: a development season")]

result = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
fig.patch.set_facecolor(SURFACE)
for ax, (key, title) in zip(axes, PANELS):
    rows = result["seasons"][key]["homogeneous_early"]
    leads = sorted(int(k) for k in rows)
    ax.set_facecolor(SURFACE)
    ends = []
    for tech, label, color in SERIES:
        ys = [rows[str(k)][tech] for k in leads]
        ax.plot(leads, ys, color=color, lw=2, solid_joinstyle="round", solid_capstyle="round", zorder=3)
        ax.plot(leads[-1], ys[-1], "o", ms=6, color=color, mec=SURFACE, mew=2, zorder=4)
        ends.append((ys[-1], {"PLVI": "Pluvium", "OFCL": "Official"}.get(tech, tech), color))
    ends.sort()
    placed = []
    for y, label, _ in ends:                                    # end labels, nudged apart so they never collide
        y_at = y if not placed or y - placed[-1] >= 9 else placed[-1] + 9
        placed.append(y_at)
        ax.text(leads[-1] + 3, y_at, label, fontsize=8, color=INK2, va="center")
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=6)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks(leads)
    ax.set_xlim(leads[0] - 4, leads[-1] + 26)
    ax.set_xlabel("Hours ahead", color=INK2)
    ax.text(leads[0], 0.96, f"{rows[str(leads[0])]['n']} forecasts at {leads[0]} h, {rows[str(leads[-1])]['n']} at {leads[-1]} h",
            fontsize=8, color=MUTED, transform=ax.get_xaxis_transform(), va="top")
axes[0].set_ylabel("Mean track error, nautical miles", color=INK2)
axes[0].set_ylim(0, None)
handles = [plt.Line2D([], [], color=c, lw=2, marker="o", ms=6, mec=SURFACE, mew=2) for _, _, c in SERIES]
fig.legend(handles, [label for _, label, _ in SERIES], loc="upper left", ncol=4, frameon=False, fontsize=9,
           bbox_to_anchor=(0.01, 1.0), handlelength=2.2)
fig.text(0.01, -0.02, "Early guidance (what is in hand at advisory time), on the forecasts all four made. Truth: the National "
         "Hurricane Center's best tracks. Lower is better.", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.01, 1, 0.92))
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
print("saved", out)
