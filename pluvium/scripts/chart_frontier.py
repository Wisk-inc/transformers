# Copyright 2026 Nathan. Apache-2.0.
"""Two charts for the frontier benchmark, from its result files.

    python scripts/chart_frontier.py RESULTS FIRST docs/img

``frontier_heavy_rain.png``: Brier skill for 50 mm or more on days 1-3, place by place: the first Blend, the new
Blend and the best other system. ``frontier_handover.png``: what the record-storm hand-over did in the test year in
each place, and what the audit had said about it from training data.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pluvium.benchmarks import fullyear as F
from pluvium.frontier import report as R

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"                    # the validated order: Pluvium is always blue
ORDER = ("jamaica", "caribbean", "maritime", "alps", "pacific_nw", "great_plains", "andes_south")
SHORT = {"jamaica": "Jamaica", "caribbean": "Caribbean", "maritime": "Maritime Continent", "alps": "Alps",
         "pacific_nw": "US Pacific Northwest", "great_plains": "US Great Plains", "andes_south": "Southern Andes"}
GROUP = "days 1-3"

results_dir, first_dir, out_dir = (Path(p) for p in sys.argv[1:4])
new = {k: json.loads((results_dir / f"{k}.json").read_text()) for k in ORDER if (results_dir / f"{k}.json").exists()}
old = {k: json.loads((first_dir / f"{k}.json").read_text()) for k in new if (first_dir / f"{k}.json").exists()}
places = [p for p in ORDER if p in new][::-1]                             # first place on top
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})


def frame(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def skill(res: dict, model: str) -> float:
    return F._value(res["groups"][GROUP].get(model), "bss", "50")


# ---------------------------------------------------------------------------------------- heavy rain --
fig, ax = plt.subplots(figsize=(8.2, 0.52 * len(places) + 1.9))
fig.patch.set_facecolor(SURFACE)
frame(ax)
for y, place in enumerate(places):
    res = new[place]
    mine = skill(res, "stack")
    rival, who = R.best_rival(res["groups"][GROUP], "bss", "50", True)
    first = skill(old[place], F._top(old[place])) if place in old else float("nan")
    if first == first:
        ax.plot([first, mine], [y, y], color=AXIS, lw=2, solid_capstyle="round", zorder=2)
        ax.plot(first, y, "o", ms=8, color=GREEN, mec=SURFACE, mew=2, zorder=3)
    ax.plot(rival, y, "D", ms=7, color=ORANGE, mec=SURFACE, mew=2, zorder=3)
    ax.plot(mine, y, "o", ms=9, color=BLUE, mec=SURFACE, mew=2, zorder=4)
    label = F.LABELS.get(str(who).replace("_raw", ""), str(who)) + (" raw" if str(who).endswith("_raw") else "")
    right = max(v for v in (mine, rival, first) if v == v)
    ax.text(right + 0.006, y, f"{mine:+.3f}   best other: {label} {rival:+.3f}", fontsize=8, color=INK2, va="center")
ax.set_yticks(range(len(places)))
ax.set_yticklabels([SHORT[p] for p in places], color=INK)
ax.set_ylim(-0.6, len(places) - 0.4)
lo, hi = ax.get_xlim()
ax.set_xlim(min(lo, -0.005), hi + 0.42 * (hi - lo))
ax.axvline(0, color=AXIS, lw=1)
ax.set_xlabel("Brier skill score for 50 mm or more, days 1-3 (higher is better; 0 is the climatology)")
handles = [plt.Line2D([], [], color=BLUE, marker="o", ms=9, mec=SURFACE, mew=2, lw=0),
           plt.Line2D([], [], color=GREEN, marker="o", ms=8, mec=SURFACE, mew=2, lw=0),
           plt.Line2D([], [], color=ORANGE, marker="D", ms=7, mec=SURFACE, mew=2, lw=0)]
fig.legend(handles, ["Pluvium Blend 0.3", "First Pluvium Blend", "Best other system, raw or post-processed"],
           loc="upper left", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.01, 1.0))
fig.text(0.01, -0.03, "Test year April 2025 to March 2026, truth CMORPH. The same year the first Blend was scored on.",
         fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.01, 1, 0.93))
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "frontier_heavy_rain.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)
print("saved", out_dir / "frontier_heavy_rain.png")

# ------------------------------------------------------------------------------------------ hand-over --
rows = []
for place in places:
    res = new[place]
    verdicts = [bool(a["keep"]) for a in res.get("audits", {}).get("stack", [])]
    effect = skill(res, "declared") - skill(res, "pool_online")
    if verdicts and effect == effect:
        rows.append((place, effect, verdicts))
if rows:
    fig, ax = plt.subplots(figsize=(8.2, 0.5 * len(rows) + 1.9))
    fig.patch.set_facecolor(SURFACE)
    frame(ax)
    span = max(abs(e) for _, e, _ in rows) or 1e-3
    for y, (place, effect, verdicts) in enumerate(rows):
        on = verdicts[0]
        ax.barh(y, effect, height=0.5, color=BLUE if on else ORANGE, zorder=3)
        said = "kept" if all(verdicts) else ("switched off" if not any(verdicts) else
                                              f"kept at {sum(verdicts)} of {len(verdicts)} refits")
        side = 1 if effect >= 0 else -1
        ax.text(effect + side * 0.03 * span, y, f"{effect:+.4f}   audit: {said}", fontsize=8, color=INK2, va="center",
                ha="left" if side > 0 else "right")
    ax.axvline(0, color=INK2, lw=1, zorder=4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([SHORT[p] for p, _, _ in rows], color=INK)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(-2.4 * span, 2.4 * span)
    ax.set_xlabel("Change in Brier skill at 50 mm on days 1-3 from handing record storms to the raw systems, test year")
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    fig.legend(handles, ["Audit on training data said: keep the hand-over", "Audit said: switch it off"], loc="upper left",
               ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.01, 1.0))
    fig.text(0.01, -0.03, "Bars to the right: the hand-over helped in the test year. The audit saw training data only. "
             "Colour is its verdict at the first refit.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.01, 1, 0.93))
    fig.savefig(out_dir / "frontier_handover.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("saved", out_dir / "frontier_handover.png")
