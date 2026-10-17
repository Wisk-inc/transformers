import sys, torch, numpy as np
from ihelix import fibonacci_sphere
from naturev1.model import NatureConfig, NatureV1
from naturev1.storms import _head_outputs
S = sys.argv[1]
ck = torch.load(f"{S}/hub/checkpoints/stage2/latest.pt", map_location="cpu", weights_only=False)
cfg = NatureConfig.from_dict(ck["config"]); L = cfg.lead_times_hours
m = NatureV1(cfg, fibonacci_sphere(cfg.latent_points, num_neighbours=cfg.latent_neighbours, cluster_size=cfg.latent_cluster)); m.load_state_dict(ck["model"]); m.eval()
print("trained steps per head:", {k: v for k, v in m.trained_heads().items()})
b = torch.load(f"{S}/hub/checkpoints/storm_bank_val.pt", weights_only=False); t = b["targets"]
with torch.no_grad(): o = _head_outputs(m, b)
def auc(p, y):
    p, y = np.asarray(p), np.asarray(y).astype(bool)
    if y.all() or not y.any(): return float("nan")
    r = p.argsort().argsort() + 1
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())
print("\nLANDFALL (will the centre cross a coast by this lead?)")
print(f"{'lead':>5} {'n':>5} {'rate':>6} {'Brier model':>12} {'Brier base-rate':>16} {'skill':>7} {'AUC':>6}")
for l, h in enumerate(L):
    k = t["landfall_mask"][:, l] > 0
    if k.sum() < 20: continue
    y = t["landfall_target"][k, l].float(); p = torch.sigmoid(o["landfall_logit"][k, l])
    bm = float(((p - y) ** 2).mean()); br = float(((y.mean() - y) ** 2).mean())
    print(f"{h:>4}h {int(k.sum()):>5} {float(y.mean()):6.1%} {bm:12.4f} {br:16.4f} {1-bm/br:7.1%} {auc(p,y):6.2f}")
print("\nRAPID INTENSIFICATION")
print("  ri_target", tuple(t["ri_target"].shape), "logits", tuple(o["ri_logits"].shape))
for j in range(t["ri_target"].shape[1]):
    k = t["ri_mask"][:, j] > 0
    if k.sum() < 20: continue
    y = t["ri_target"][k, j].float(); p = torch.sigmoid(o["ri_logits"][k, j])
    bm = float(((p - y) ** 2).mean()); br = float(((y.mean() - y) ** 2).mean())
    print(f"  threshold {j}: n {int(k.sum())}, events {int(y.sum())} ({float(y.mean()):.1%}), Brier {bm:.4f} vs base-rate {br:.4f} (skill {1-bm/br:+.1%}), AUC {auc(p,y):.2f}")
print("\nSIZE: radius of max wind (n mi) and wind radii")
for name, pred, tgt in [("RMW", o["eyewall_rmw_nmi"], t["rmw_target"]), ("eyewall_target vs peak wind", o["eyewall_peak_wind_kt"], t["eyewall_target"])]:
    for l in (0, 3, 6):
        k = torch.isfinite(tgt[:, l]) & (tgt[:, l] > 0)
        if k.sum() < 20: continue
        y, p = tgt[k, l], pred[k, l] if pred.dim() > 1 else pred[k]
        print(f"  {name} +{L[l]}h: n {int(k.sum())}, model MAE {float((p-y).abs().mean()):.1f}, predict-the-average MAE {float((y-y.mean()).abs().mean()):.1f}")
wr, tw = o["wind_radii_nmi"], t["wind_radii_target"]
for l in (0, 3, 6):
    k = torch.isfinite(tw[:, l]) & (tw[:, l] > 0)
    y, p = tw[:, l][k], wr[:, l][k]
    print(f"  wind radii +{L[l]}h: n {int(k.sum())}, model MAE {float((p-y).abs().mean()):.1f} n mi, predict-the-average MAE {float((y-y.mean()).abs().mean()):.1f}")
