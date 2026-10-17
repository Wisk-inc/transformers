import sys, torch, numpy as np
from naturev1.model import NatureConfig, NatureV1
from naturev1.storms import storm_scorecard, _head_outputs, track_members
from naturev1.besttrack import parse_hurdat2
S = sys.argv[1]
ck = torch.load(f"{S}/hub/checkpoints/stage2/latest.pt", map_location="cpu", weights_only=False)
cfg = NatureConfig.from_dict(ck["config"])
from ihelix import fibonacci_sphere
mesh = fibonacci_sphere(cfg.latent_points, num_neighbours=cfg.latent_neighbours, cluster_size=cfg.latent_cluster)
model = NatureV1(cfg, mesh); model.load_state_dict(ck["model"]); model.eval()
bank = torch.load(f"{S}/hub/checkpoints/storm_bank_val.pt", map_location="cpu", weights_only=False)
print(storm_scorecard(model, bank, cfg.lead_times_hours))
torch.save({"out": _head_outputs(model, bank), "members": track_members(_head_outputs(model, bank), bank["targets"], cfg.lead_times_hours)}, f"{S}/preds.pt")
