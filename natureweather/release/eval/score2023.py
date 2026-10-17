"""Score the 2023 (El Nino) end-to-end banks against the same metrics on the 2017/2019/2021 held-out storms."""
import sys
from pathlib import Path

import numpy as np
import torch
from ihelix import fibonacci_sphere
from naturev1.model import NatureConfig, NatureV1
from naturev1.storms import _great_circle_km, _head_outputs, consensus_track, track_members

S = Path(sys.argv[1])
ck = torch.load(S / "hub/checkpoints/stage2/latest.pt", map_location="cpu", weights_only=False)
cfg = NatureConfig.from_dict(ck["config"]); LEADS = cfg.lead_times_hours
model = NatureV1(cfg, fibonacci_sphere(cfg.latent_points, num_neighbours=cfg.latent_neighbours,
                                       cluster_size=cfg.latent_cluster))
model.load_state_dict(ck["model"]); model.eval()
KT = 0.514444


def evaluate(bank, hours=("00",), issued=None):
    t = bank["targets"]
    with torch.no_grad():
        out = _head_outputs(model, bank)
        mem = track_members(out, t, LEADS)
    blend = consensus_track(mem, model.track_consensus)        # the shipped weights: no refitting on 2023
    c, truth, valid = t["storm_center"].float(), t["track_target"].float(), t["track_valid"] > 0
    keep = torch.ones(len(c), dtype=torch.bool)
    if issued is not None:
        keep = torch.tensor([s[11:13] in hours for s in issued])
    rows = {}
    for l, h in enumerate(LEADS):
        ok = valid[:, l] & keep
        if ok.sum() < 5:
            continue
        a0, a1 = c[ok, 0] + truth[ok, l, 0], c[ok, 1] + truth[ok, l, 1]

        def err(d):
            return float(_great_circle_km(a0, a1, c[ok, 0] + d[:, 0], c[ok, 1] + d[:, 1]).mean())

        both = ok & (t["intensity_mask"][:, l, 0] > 0) & torch.isfinite(t["current_wind_kt"])
        wt, wp = t["intensity_target"][both, l, 0] / KT, out["intensity_mean"][both, l, 0] / KT
        lm = (t["landfall_mask"][:, l] > 0) & keep
        rows[h] = dict(n=int(ok.sum()), consensus=err(blend[ok, l]), model=err(mem[ok, l, 0]),
                       steering=err(mem[ok, l, 2]), extrap=err(mem[ok, l, 3]), still=err(torch.zeros_like(truth[ok, l])),
                       wind=float((wp - wt).abs().mean()) if both.any() else float("nan"),
                       persist=float((t["current_wind_kt"][both] - wt).abs().mean()) if both.any() else float("nan"))
    return rows, out, blend, mem


def table(title, rows):
    print(f"\n{title}")
    print(f"{'lead':>5} {'n':>4} {'NatureV1':>9} {'steering':>9} {'straight':>9} {'stay-put':>9} {'wind err':>9} {'persist':>8}")
    for h, r in rows.items():
        print(f"{h:>4}h {r['n']:>4} {r['consensus']:>7.0f}km {r['steering']:>7.0f}km {r['extrap']:>7.0f}km "
              f"{r['still']:>7.0f}km {r['wind']:>7.1f}kt {r['persist']:>6.1f}kt")


banks = sorted((S / "elnino2023").glob("*.pt"))
merged = None
for path in banks:
    b = torch.load(path, weights_only=False)
    if merged is None:
        merged = {"features": [], "features_far": [], "targets": {k: [] for k in b["targets"]}, "issued": [], "name": []}
    merged["features"].append(b["features"]); merged["features_far"].append(b["features_far"])
    for k, v in b["targets"].items():
        merged["targets"][k].append(v)
    merged["issued"] += b["issued"]; merged["name"] += [path.stem] * len(b["issued"])
el = {"features": torch.cat(merged["features"]), "features_far": torch.cat(merged["features_far"]),
      "targets": {k: torch.cat(v) for k, v in merged["targets"].items()}}
print(f"2023 storms scored: {', '.join(p.stem.title() for p in banks)} ({len(merged['issued'])} forecasts, issued 00Z)")

old = torch.load(S / "hub/checkpoints/storm_bank_val.pt", weights_only=False)
r23, out23, blend23, mem23 = evaluate(el)
rold, *_ = evaluate(old)
table("2023 -- strong El Nino, never seen by any part of the model (full model from raw ERA5):", r23)
table("2017 / 2019 / 2021 -- La Nina / neutral held-out seasons, same metrics, same shipped weights:", rold)

print("\nper storm, NatureV1 consensus track error (km) at 1 / 2 / 3 days, against physics steering:")
t = el["targets"]; c, truth, valid = t["storm_center"].float(), t["track_target"].float(), t["track_valid"] > 0
for name in dict.fromkeys(merged["name"]):
    idx = torch.tensor([i for i, n in enumerate(merged["name"]) if n == name])
    cells = []
    for h in (24, 48, 72):
        l = LEADS.index(h)
        ok = idx[valid[idx, l]]
        if len(ok) == 0:
            cells.append("    -");
            continue
        a0, a1 = c[ok, 0] + truth[ok, l, 0], c[ok, 1] + truth[ok, l, 1]
        e = float(_great_circle_km(a0, a1, c[ok, 0] + blend23[ok, l, 0], c[ok, 1] + blend23[ok, l, 1]).mean())
        s = float(_great_circle_km(a0, a1, c[ok, 0] + mem23[ok, l, 2, 0], c[ok, 1] + mem23[ok, l, 2, 1]).mean())
        cells.append(f"{e:5.0f} ({s:4.0f})")
    print(f"  {name.title():<9} n={len(idx):>2}   " + "   ".join(cells))

print("\nlandfall probability (2023): Brier vs base rate, and AUC")
for l, h in enumerate(LEADS):
    k = t["landfall_mask"][:, l] > 0
    y = t["landfall_target"][k, l].float()
    if k.sum() < 10 or y.sum() == 0 or y.sum() == len(y):
        continue
    p = torch.sigmoid(out23["landfall_logit"][k, l])
    bm, br = float(((p - y) ** 2).mean()), float(((y.mean() - y) ** 2).mean())
    r = p.argsort().argsort().float() + 1
    auc = float((r[y > 0].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (1 - y).sum()))
    print(f"  {h:>4}h n {int(k.sum()):>3}  landfalls {int(y.sum()):>3}  skill vs base rate {1 - bm / br:+.0%}  AUC {auc:.2f}")
