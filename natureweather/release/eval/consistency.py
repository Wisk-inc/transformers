import sys, torch
from ihelix import fibonacci_sphere
from naturev1.model import NatureConfig, NatureV1
from naturev1.storms import _head_outputs
S = sys.argv[1]
ck = torch.load(f"{S}/hub/checkpoints/stage2/latest.pt", map_location="cpu", weights_only=False)
cfg = NatureConfig.from_dict(ck["config"]); L = cfg.lead_times_hours
m = NatureV1(cfg, fibonacci_sphere(cfg.latent_points, num_neighbours=cfg.latent_neighbours, cluster_size=cfg.latent_cluster)); m.load_state_dict(ck["model"]); m.eval()
val = torch.load(f"{S}/hub/checkpoints/storm_bank_val.pt", weights_only=False)
tr = torch.load(f"{S}/hub/checkpoints/storm_bank_train.pt", weights_only=False)
sv, st = set(val["targets"]["season"].tolist()), set(tr["targets"]["season"].tolist())
print("LEAKAGE: validation seasons", sorted(sv), "| training seasons", min(st), "-", max(st), f"({len(st)} seasons)",
      "| overlap:", sorted(sv & st) or "none", "| test seasons 2018/2020/2022 in training:", sorted({2018, 2020, 2022} & st) or "none")
print("weather backbone trained on 2011-2015 ERA5; storm-head training seasons inside that window:", sorted(s for s in st if 2011 <= s <= 2015))
with torch.no_grad(): o = _head_outputs(m, val)
N = o["landfall_logit"].shape[0]
p = torch.sigmoid(o["landfall_logit"])
drops = (p[:, 1:] < p[:, :-1] - 0.01).any(1).float().mean()
print(f"\nCONSISTENCY: landfall 'by this lead' probability falls as lead grows in {drops:.0%} of forecasts "
      f"(a cumulative probability must never fall); worst drop {float((p[:, :-1] - p[:, 1:]).max()):.0%}")
ri = torch.sigmoid(o["ri_logits"])
bad = ((ri[:, 1] > ri[:, 0] + 0.01) | (ri[:, 2] > ri[:, 1] + 0.01)).float().mean()
print(f"RI: P(bigger jump) exceeds P(smaller jump) in {bad:.0%} of forecasts (must never happen)")
wr = o["wind_radii_nmi"]                          # (N, L, 3, 4)
inv = ((wr[:, :, 1] > wr[:, :, 0] + 1) | (wr[:, :, 2] > wr[:, :, 1] + 1)).any(-1).any(-1).float().mean()
print(f"wind radii: a 50/64 kt radius larger than the 34/50 kt one in {inv:.0%} of forecasts (must never happen)")
neg = (wr < 0).any(-1).any(-1).any(-1).float().mean()
print(f"wind radii: negative radius in {neg:.0%} of forecasts")
pk = o["eyewall_peak_wind_kt"]; im = o["intensity_mean"][..., 0] / 0.514444
print(f"two intensity heads disagree: |eyewall peak wind - intensity head| mean {float((pk - im).abs().mean()):.1f} kt, "
      f">20 kt apart in {float(((pk - im).abs() > 20).float().mean()):.0%} of lead-forecasts")
rmw = o["eyewall_rmw_nmi"]; r34 = wr[:, :, 0].amax(-1)
print(f"radius of max wind larger than the 34 kt radius in {float((rmw > r34).float().mean()):.0%} of lead-forecasts (physically impossible)")
pr = o["intensity_mean"][..., 1]
print("intensity head second channel (pressure) range:", round(float(pr.min()), 2), "to", round(float(pr.max()), 2))
