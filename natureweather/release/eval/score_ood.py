import sys, glob, torch
from pathlib import Path
from ihelix import fibonacci_sphere
from naturev1.model import NatureConfig, NatureV1
from naturev1.storms import _head_outputs, track_members, consensus_track, _great_circle_km
S = sys.argv[1]; KT = 0.514444
ck = torch.load(f"{S}/hub/checkpoints/stage2/latest.pt", map_location="cpu", weights_only=False)
cfg = NatureConfig.from_dict(ck["config"]); L = cfg.lead_times_hours
m = NatureV1(cfg, fibonacci_sphere(cfg.latent_points, num_neighbours=cfg.latent_neighbours, cluster_size=cfg.latent_cluster)); m.load_state_dict(ck["model"]); m.eval()
print(f"{'storm':<18} {'n':>3} | 24h km: NatV1 steer line | 72h km: NatV1 steer line | wind MAE 24h: model persist | 72h north-south: predicted vs actual")
for path in sorted(glob.glob(f"{S}/ood/*.pt")):
    b = torch.load(path, weights_only=False); t = b["targets"]
    with torch.no_grad(): o = _head_outputs(m, b); mem = track_members(o, t, L)
    blend = consensus_track(mem, m.track_consensus); c, tr, v = t["storm_center"], t["track_target"], t["track_valid"] > 0
    cells = []
    for h in (24, 72):
        l = L.index(h); k = v[:, l]
        if not k.any(): cells.append("    -     -     -"); continue
        a0, a1 = c[k, 0] + tr[k, l, 0], c[k, 1] + tr[k, l, 1]
        e = lambda d: float(_great_circle_km(a0, a1, c[k, 0] + d[:, 0], c[k, 1] + d[:, 1]).mean())
        cells.append(f"{e(blend[k, l]):5.0f} {e(mem[k, l, 2]):5.0f} {e(mem[k, l, 3]):5.0f}")
    l = L.index(24); k = v[:, l] & (t["intensity_mask"][:, l, 0] > 0)
    wm = float(((o["intensity_mean"][k, l, 0] - t["intensity_target"][k, l, 0]) / KT).abs().mean()) if k.any() else float("nan")
    wp = float((t["current_wind_kt"][k] - t["intensity_target"][k, l, 0] / KT).abs().mean()) if k.any() else float("nan")
    l = L.index(72); k = v[:, l]
    ns = f"{float(blend[k, l, 0].mean()):+5.1f} vs {float(tr[k, l, 0].mean()):+5.1f} deg" if k.any() else "-"
    print(f"{Path(path).stem:<18} {len(b['issued']):>3} | {cells[0]} | {cells[1]} | {wm:11.1f} {wp:7.1f} | {ns}")
    if "OTIS" in path:
        print("   Otis, forecast wind vs actual (kt), by issue time:")
        for i, issued in enumerate(b["issued"]):
            f = [round(float(o['intensity_mean'][i, L.index(h), 0]) / KT) for h in (12, 24)]
            a = [round(float(t['intensity_target'][i, L.index(h), 0]) / KT) if t['intensity_mask'][i, L.index(h), 0] > 0 else None for h in (12, 24)]
            ri = torch.sigmoid(o["ri_logits"][i]).tolist()
            print(f"     {issued}  now {float(t['current_wind_kt'][i]):3.0f} kt | +12h {f[0]} vs {a[0]} | +24h {f[1]} vs {a[1]} | P(RI 25/30/35 kt) {ri[0]:.0%}/{ri[1]:.0%}/{ri[2]:.0%}")
