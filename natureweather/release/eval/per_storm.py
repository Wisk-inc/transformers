import sys, torch, numpy as np
from ihelix import fibonacci_sphere
from naturev1.model import NatureConfig, NatureV1
from naturev1.storms import _head_outputs, track_members, _cross_validated_consensus, _great_circle_km
from naturev1.besttrack import parse_hurdat2
S = sys.argv[1]
ck = torch.load(f"{S}/hub/checkpoints/stage2/latest.pt", map_location="cpu", weights_only=False)
cfg = NatureConfig.from_dict(ck["config"])
model = NatureV1(cfg, fibonacci_sphere(cfg.latent_points, num_neighbours=cfg.latent_neighbours, cluster_size=cfg.latent_cluster))
model.load_state_dict(ck["model"]); model.eval()
bank = torch.load(f"{S}/hub/checkpoints/storm_bank_val.pt", map_location="cpu", weights_only=False)
t = bank["targets"]; leads = cfg.lead_times_hours
with torch.no_grad():
    out = _head_outputs(model, bank); mem = track_members(out, t, leads)
c = t["storm_center"].float(); truth = t["track_target"].float(); valid = t["track_valid"] > 0
seasons = t["season"]; blend = _cross_validated_consensus(mem, truth, valid, c[:, 0], seasons)
# name each sample from HURDAT2 by season + exact centre
rows = {}
for tr in parse_hurdat2(f"{S}/hurdat/hurdat2.txt"):
    if tr.year in (2017, 2019, 2021):
        for i in range(len(tr)):
            rows.setdefault((tr.year, round(float(tr.latitude[i]), 1), round(float(tr.longitude[i]), 1)), []).append((tr.name, str(tr.time[i])[:16], float(tr.max_wind_kt[i])))
names, times = [], []
for k in range(len(c)):
    hit = rows.get((int(seasons[k]), round(float(c[k, 0]), 1), round(float(c[k, 1]), 1)))
    names.append(hit[0][0] if hit else "?"); times.append(hit[0][1] if hit else "?")
print("matched", sum(n != "?" for n in names), "of", len(names))
def err(k, l, d):
    return float(_great_circle_km(c[k, 0] + truth[k, l, 0], c[k, 1] + truth[k, l, 1], c[k, 0] + d[0], c[k, 1] + d[1]))
L = {h: i for i, h in enumerate(leads)}
print(f"\n{'storm':<10}{'year':>5}{'n':>5} | 24h: {'NatV1':>6}{'steer':>6}{'line':>6} | 72h: {'NatV1':>6}{'steer':>6}{'line':>6}")
for name in ["HARVEY", "IRMA", "JOSE", "MARIA", "NATE", "OPHELIA", "DORIAN", "HUMBERTO", "LORENZO", "BARRY", "IDA", "LARRY", "SAM", "ELSA", "HENRI", "NICHOLAS"]:
    ks = [k for k in range(len(c)) if names[k] == name]
    if not ks: continue
    def avg(l, m):
        v = [err(k, L[l], (blend[k, L[l]] if m == "c" else mem[k, L[l], m])) for k in ks if valid[k, L[l]]]
        return f"{np.mean(v):6.0f}" if v else "     -"
    print(f"{name:<10}{int(seasons[ks[0]]):>5}{len(ks):>5} | 24h: {avg(24,'c')}{avg(24,2)}{avg(24,3)} | 72h: {avg(72,'c')}{avg(72,2)}{avg(72,3)}")
# one detailed forecast: Dorian, issued 2019-08-30 18Z if present, else nearest
def detail(name, when):
    ks = [k for k in range(len(c)) if names[k] == name and times[k].startswith(when)]
    if not ks: print(f"\n{name} {when}: not in bank"); return
    k = ks[0]; probs = torch.softmax(out["mode_logits"][k], -1) if "mode_logits" in out else None
    print(f"\n{name}, issued {times[k]}Z from {float(c[k,0]):.1f}N {-float(c[k,1]):.1f}W" + (f"; top scenario {float(probs.max()):.0%}" if probs is not None else ""))
    print(f"{'lead':>5} {'actual':>14} {'NatureV1':>14} {'miss km':>8} {'steering km':>12} {'line km':>8}  {'wind fcst/actual':>16}")
    for h in leads:
        l = L[h]
        if not valid[k, l]: continue
        a = c[k] + truth[k, l]; f = c[k] + blend[k, l]
        wf = float(out["intensity_mean"][k, l, 0]) / 0.514444; wa = float(t["intensity_target"][k, l, 0]) / 0.514444
        print(f"{h:>4}h {float(a[0]):5.1f}N {-float(a[1]):5.1f}W {float(f[0]):5.1f}N {-float(f[1]):5.1f}W {err(k,l,blend[k,l]):8.0f} {err(k,l,mem[k,l,2]):12.0f} {err(k,l,mem[k,l,3]):8.0f}  {wf:7.0f} / {wa:3.0f} kt")
detail("DORIAN", "2019-08-30T18"); detail("IRMA", "2017-09-06T00"); detail("IDA", "2021-08-28T00"); detail("MARIA", "2017-09-18T00")
print("\nkeys", list(out.keys()))
