import sys, glob, torch, numpy as np
from ihelix import fibonacci_sphere
from naturev1.model import NatureConfig, NatureV1
from naturev1.storms import _head_outputs
S = sys.argv[1]; torch.manual_seed(0)
ck = torch.load(f"{S}/hub/checkpoints/stage2/latest.pt", map_location="cpu", weights_only=False)
cfg = NatureConfig.from_dict(ck["config"]); L = cfg.lead_times_hours; KT = 0.514444
m = NatureV1(cfg, fibonacci_sphere(cfg.latent_points, num_neighbours=cfg.latent_neighbours, cluster_size=cfg.latent_cluster)); m.load_state_dict(ck["model"]); m.eval()
def cat_banks(paths):
    bs = [torch.load(p, weights_only=False) for p in paths]
    return {"features": torch.cat([b["features"] for b in bs]), "features_far": torch.cat([b["features_far"] for b in bs]),
            "targets": {k: torch.cat([b["targets"][k] for b in bs]) for k in bs[0]["targets"]}}
sets = {"held-out 2017/19/21": torch.load(f"{S}/hub/checkpoints/storm_bank_val.pt", weights_only=False),
        "2023 El Nino": cat_banks(sorted(glob.glob(f"{S}/elnino2023/*.pt")))}
def logpdf(x, mu, ls, rho):        # x (...,2) degrees
    s = ls.exp(); z = (x - mu) / s; r = rho.clamp(-0.99, 0.99); om = 1 - r**2
    q = (z[..., 0]**2 - 2*r*z[..., 0]*z[..., 1] + z[..., 1]**2) / om
    return -0.5 * (q + 2*ls.sum(-1) + torch.log(om) + 2*np.log(2*np.pi)), q
for name, bank in sets.items():
    t = bank["targets"]
    with torch.no_grad(): o = _head_outputs(m, bank)
    w = o["mode_logits"].softmax(-1); top = w.argmax(-1)
    print(f"\n=== {name} ===")
    print(f"{'lead':>5} {'n':>5} | track: mix50% mix90% top-cone95% | wind 90% interval | wind bias kt")
    for l, h in enumerate(L):
        ok = t["track_valid"][:, l] > 0
        idx = ok.nonzero().flatten()
        if len(idx) < 20: continue
        truth = t["track_target"][idx, l]
        mu, ls, rho = o["displacement"][idx, :, l], o["log_scale"][idx, :, l], o["correlation"][idx, :, l]
        lp_c, q = logpdf(truth[:, None], mu, ls, rho)
        lp_truth = torch.logsumexp(lp_c + w[idx].log(), -1)
        # sample the mixture to find its highest-density thresholds
        n = 2000
        comp = torch.multinomial(w[idx], n, replacement=True)                    # (N, n)
        g = lambda a: torch.gather(a, 1, comp[..., None].expand(-1, -1, a.shape[-1]) if a.dim() == 3 else comp)
        mus, lss, rhs = g(mu), g(ls), g(rho)
        e1, e2 = torch.randn(len(idx), n), torch.randn(len(idx), n)
        r_ = rhs.clamp(-0.99, 0.99)
        x = torch.stack([e1, r_*e1 + (1-r_**2).sqrt()*e2], -1) * lss.exp() + mus
        lp_s = torch.logsumexp(logpdf(x[:, :, None], mu[:, None], ls[:, None], rho[:, None])[0] + w[idx].log()[:, None], -1)
        c50 = (lp_truth >= lp_s.quantile(0.5, dim=1)).float().mean(); c90 = (lp_truth >= lp_s.quantile(0.1, dim=1)).float().mean()
        topq = q[torch.arange(len(idx)), top[idx]]
        cone = (topq <= 5.991).float().mean()
        wm = t["intensity_mask"][:, l, 0] > 0
        widx = (ok & wm).nonzero().flatten()
        mean, sd = o["intensity_mean"][widx, l, 0], o["intensity_log_var"][widx, l, 0].mul(0.5).exp()
        tw = t["intensity_target"][widx, l, 0]
        cov = ((tw - mean).abs() <= 1.645 * sd).float().mean()
        bias = ((mean - tw) / KT).mean()
        print(f"{h:>4}h {len(idx):>5} |        {c50:6.0%} {c90:6.0%}  {cone:9.0%}  |      {cov:6.0%}       | {bias:+6.1f}")
    # intensity bias by category
    cw = t["current_wind_kt"]
    print("wind error by current strength (forecast minus actual, kt):")
    for lo, hi, lab in ((0, 64, "tropical storm (<64 kt)"), (64, 96, "Cat 1-2 (64-95 kt)"), (96, 400, "Cat 3-5 (96+ kt)")):
        row = []
        for h in (24, 72):
            l = L.index(h); k = (cw >= lo) & (cw < hi) & (t["intensity_mask"][:, l, 0] > 0) & (t["track_valid"][:, l] > 0)
            if k.sum() < 5: row.append("   n/a"); continue
            err = (o["intensity_mean"][k, l, 0] - t["intensity_target"][k, l, 0]) / KT
            row.append(f"{h}h bias {float(err.mean()):+5.1f} MAE {float(err.abs().mean()):4.1f} (n {int(k.sum())})")
        print(f"  {lab:<24} " + " | ".join(row))
    # landfall and RI reliability
    for key, logit, target, mask, cols in (("landfall by 72h", "landfall_logit", "landfall_target", "landfall_mask", L.index(72)),
                                          ("RI +30 kt/24h", "ri_logits", "ri_target", "ri_mask", 1)):
        k = t[mask][:, cols] > 0
        p, y = torch.sigmoid(o[logit][k, cols]), t[target][k, cols].float()
        bins = [0, .1, .3, .5, .7, 1.01]; parts = []
        for a, b2 in zip(bins[:-1], bins[1:]):
            s = (p >= a) & (p < b2)
            if s.sum(): parts.append(f"{a:.0%}-{min(b2,1):.0%}: said {float(p[s].mean()):.0%} got {float(y[s].mean()):.0%} (n {int(s.sum())})")
        print(f"{key} reliability: " + "; ".join(parts))
