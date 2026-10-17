import sys, torch
from ihelix import fibonacci_sphere
from naturev1.model import NatureConfig, NatureV1
from naturev1.storms import _head_outputs, track_members, consensus_track, _great_circle_km
S = sys.argv[1]
ck = torch.load(f"{S}/hub/checkpoints/stage2/latest.pt", map_location="cpu", weights_only=False)
cfg = NatureConfig.from_dict(ck["config"]); leads = cfg.lead_times_hours
model = NatureV1(cfg, fibonacci_sphere(cfg.latent_points, num_neighbours=cfg.latent_neighbours, cluster_size=cfg.latent_cluster))
model.load_state_dict(ck["model"]); model.eval()
new = torch.load(f"{S}/maria_e2e_bank.pt", weights_only=False)
old = torch.load(f"{S}/hub/checkpoints/storm_bank_val.pt", weights_only=False)
oc = old["targets"]["storm_center"]
def run(bank):
    with torch.no_grad():
        out = _head_outputs(model, bank); mem = track_members(out, bank["targets"], leads)
    return out, consensus_track(mem, model.track_consensus), mem
o_new, c_new, m_new = run(new); o_old, c_old, m_old = run(old)
t = new["targets"]
for k in range(new["features"].shape[0]):
    j = int(((oc - t["storm_center"][k]).abs().sum(1)).argmin())
    cos = torch.nn.functional.cosine_similarity(new["features"][k], old["features"][j], 0)
    cosf = torch.nn.functional.cosine_similarity(new["features_far"][k], old["features_far"][j], 0)
    c = t["storm_center"][k]; tr, ok = t["track_target"][k], t["track_valid"][k] > 0
    p = torch.softmax(o_new["mode_logits"][k], -1).max()
    print(f"\nMaria from {float(c[0]):.1f}N {-float(c[1]):.1f}W | backbone features vs saved bank: cosine near {cos:.5f}, far {cosf:.5f} | top scenario {p:.0%}")
    print(f"{'lead':>5} {'actual':>13} {'end-to-end':>13} {'miss':>6} {'bank-based':>13} {'miss':>6} {'steer':>6}  wind e2e/bank/actual")
    for l, h in enumerate(leads):
        if not ok[l]: continue
        a = c + tr[l]; f = c + c_new[k, l]; g = oc[j] + c_old[j, l]
        e = lambda q: float(_great_circle_km(a[0], a[1], q[0], q[1]))
        print(f"{h:>4}h {float(a[0]):5.1f}N {-float(a[1]):5.1f}W {float(f[0]):5.1f}N {-float(f[1]):5.1f}W {e(f):6.0f} {float(g[0]):5.1f}N {-float(g[1]):5.1f}W {e(g):6.0f} {e(c + m_new[k, l, 2]):6.0f}  "
              f"{float(o_new['intensity_mean'][k,l,0])/0.514444:4.0f} / {float(o_old['intensity_mean'][j,l,0])/0.514444:4.0f} / {float(t['intensity_target'][k,l,0])/0.514444:4.0f} kt")
