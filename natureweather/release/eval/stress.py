import sys, math, datetime as dt, torch, numpy as np
from ihelix import fibonacci_sphere
from naturev1.model import NatureConfig, NatureV1
from naturev1.forecast import build_forecast
from naturev1.guards import check_forecast
S = sys.argv[1]
ck = torch.load(f"{S}/hub/checkpoints/stage2/latest.pt", map_location="cpu", weights_only=False)
cfg = NatureConfig.from_dict(ck["config"]); L = cfg.lead_times_hours
m = NatureV1(cfg, fibonacci_sphere(cfg.latent_points, num_neighbours=cfg.latent_neighbours, cluster_size=cfg.latent_cluster)); m.load_state_dict(ck["model"]); m.eval()
b = torch.load(f"{S}/hub/checkpoints/storm_bank_val.pt", weights_only=False); t = b["targets"]
k = int(((t["storm_center"] - torch.tensor([14.0, -58.0])).abs().sum(1)).argmin())   # Maria, 18 Sep 2017
trained = m.trained_heads()
def run(label, feat=None, far=None, centre=None, state=None, env=None, prior=None, inputs=(), trained=trained):
    f = b["features"][k:k+1].clone() if feat is None else feat
    g = b["features_far"][k:k+1].clone() if far is None else far
    c = t["storm_center"][k:k+1].clone() if centre is None else torch.tensor([centre], dtype=torch.float32)
    s = t["storm_state"][k:k+1].clone() if state is None else state
    e = t["storm_environment"][k:k+1].clone() if env is None else env
    p = t["track_prior"][k:k+1].clone() if prior is None else prior
    try:
        with torch.no_grad():
            out = m.storm_heads(f, c, s, e, p, g)
        bundle = build_forecast(out, L, dt.datetime(2017, 9, 18), storm_center=tuple(float(x) for x in c[0]),
                                trained=trained, inputs=inputs, consensus={"weights": m.track_consensus, "storm_state": s[0]})
        probs = check_forecast(bundle)
        sc = bundle["track_scenarios"]; top = sc[0] if isinstance(sc, list) else None
        last = top["track"][-1] if top else None
        cons = bundle.get("track_consensus", {}).get("track", [])
        inten = bundle.get("intensity", [])
        w = [round(r["max_wind_ms"] / 0.514444) for r in inten] if isinstance(inten, list) else "n/a"
        ri = bundle.get("rapid_intensification", {})
        print(f"\n[{label}]  trustworthy={bundle['trustworthy']}  guard problems={len(probs)}")
        if top: print(f"  top scenario {top['probability']:.0%}: +120h at {last['latitude']:.1f}, {last['longitude']:.1f}; cone95 {last.get('cone_radius_km_95', float('nan')):.0f} km")
        if cons: print(f"  consensus +120h at {cons[-1]['latitude']:.1f}, {cons[-1]['longitude']:.1f}")
        print(f"  wind kt by lead {w}; RI p30 {ri.get('probability_30kt', 'n/a')}")
        for p_ in probs[:4]: print("   guard:", p_)
    except Exception as err:
        print(f"\n[{label}]  CRASH {type(err).__name__}: {err}")
run("baseline: Maria 18 Sep 2017")
run("centre at the dateline 15N 179.5E", centre=[15.0, 179.5])
run("centre at 88N (polar)", centre=[88.0, -58.0])
run("southern hemisphere 15S 160E", centre=[-15.0, 160.0])
run("NaN centre", centre=[float("nan"), float("nan")])
st = t["storm_state"][k:k+1].clone(); st[0, 0] = (400 - 0) / 1; run("storm state: 400 kt current wind (raw, unscaled)", state=st)
st = t["storm_state"][k:k+1].clone(); st[0, 0] = 20.0; st[0, 1] = -20.0; run("storm state: extreme scaled wind/pressure (+20 sd)", state=st)
run("garbage backbone features (noise x10)", feat=torch.randn(1, 512) * 10, far=torch.randn(1, 512) * 10)
run("all-zero features", feat=torch.zeros(1, 512), far=torch.zeros(1, 512))
e = t["storm_environment"][k:k+1].clone() * 50; run("environment x50 (absurd shear/steering)", env=e)
run("steering prior shifted 30 degrees", prior=t["track_prior"][k:k+1].clone() + 30.0)
run("no training record supplied", trained=None)
run("satellite input path (never trained)", inputs=("satellite",))
