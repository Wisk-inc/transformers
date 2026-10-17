import sys, copy, torch
from ihelix import fibonacci_sphere
from naturev1.model import NatureConfig, NatureV1
from naturev1.storms import _head_outputs, track_members, consensus_track, _great_circle_km
S = sys.argv[1]; torch.manual_seed(0)
ck = torch.load(f"{S}/hub/checkpoints/stage2/latest.pt", map_location="cpu", weights_only=False)
cfg = NatureConfig.from_dict(ck["config"]); L = cfg.lead_times_hours; KT = 0.514444
m = NatureV1(cfg, fibonacci_sphere(cfg.latent_points, num_neighbours=cfg.latent_neighbours, cluster_size=cfg.latent_cluster)); m.load_state_dict(ck["model"]); m.eval()
base = torch.load(f"{S}/hub/checkpoints/storm_bank_val.pt", weights_only=False)
t0 = base["targets"]; N = base["features"].shape[0]; perm = torch.randperm(N)
def score(bank):
    t = bank["targets"]
    with torch.no_grad():
        o = _head_outputs(m, bank); mem = track_members(o, t, L)
    blend = consensus_track(mem, m.track_consensus)
    c, tr, v = t0["storm_center"], t0["track_target"], t0["track_valid"] > 0     # truth is always the real storm's
    res = []
    for h in (24, 72):
        l = L.index(h); k = v[:, l]
        a0, a1 = c[k, 0] + tr[k, l, 0], c[k, 1] + tr[k, l, 1]
        res.append(float(_great_circle_km(a0, a1, c[k, 0] + blend[k, l, 0], c[k, 1] + blend[k, l, 1]).mean()))
    for h in (24, 72):
        l = L.index(h); k = (v[:, l]) & (t0["intensity_mask"][:, l, 0] > 0)
        res.append(float(((o["intensity_mean"][k, l, 0] - t0["intensity_target"][k, l, 0]) / KT).abs().mean()))
    return res
def variant(**changes):
    b = {"features": base["features"].clone(), "features_far": base["features_far"].clone(), "targets": {k: v.clone() for k, v in t0.items()}}
    for key, fn in changes.items():
        if key in ("features", "features_far"): b[key] = fn(b[key])
        else: b["targets"][key] = fn(b["targets"][key])
    return b
mean = lambda x: x.mean(0, keepdim=True).expand_as(x).clone()
shuffle = lambda x: x[perm]
cases = [("full model (nothing removed)", {}),
         ("backbone features (near storm) -> average", {"features": mean}),
         ("backbone features (near) from a different storm", {"features": shuffle}),
         ("far pool (1000 km) -> average", {"features_far": mean}),
         ("ALL backbone features -> average", {"features": mean, "features_far": mean}),
         ("physical environment (shear, steering, SST) -> average", {"storm_environment": mean}),
         ("storm state (wind, pressure, 12h motion) -> average", {"storm_state": mean}),
         ("12h motion only -> zero", {"storm_state": lambda x: torch.cat([x[:, :2], torch.zeros_like(x[:, 2:4]), x[:, 4:]], 1)}),
         ("current wind/pressure only -> average", {"storm_state": lambda x: torch.cat([mean(x)[:, :2], x[:, 2:]], 1)}),
         ("steering-track prior -> zero (no physics prior)", {"track_prior": torch.zeros_like})]
ref = None
print(f"{'what was removed':<56} {'track 24h':>9} {'track 72h':>9} {'wind 24h':>8} {'wind 72h':>8}")
for label, ch in cases:
    r = score(variant(**ch)); ref = ref or r
    d = [f"{x:7.0f}km" if i < 2 else f"{x:6.1f}kt" for i, x in enumerate(r)]
    pct = [f"{100*(x/y-1):+4.0f}%" for x, y in zip(r, ref)]
    print(f"{label:<56} " + " ".join(f"{a}({p})" for a, p in zip(d, pct)))
