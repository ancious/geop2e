"""e8g — offline scorer for e8e_bench output: geometric success criterion
(p2m median < 0.5 mm AND p95 < 2.0 mm; symmetry-free, matches 'perfect
registration' semantics), RRE_sym as secondary diagnostic with the detected
discrete equivalence class. Produces the final per-part and pooled tables."""
import json, sys, os
import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "results/e8e_bench.json"
GEO_SUCC = dict(median=0.5, p95=2.0)
d = json.load(open(SRC))
methods = ["init", "icp_p2p", "icp_p2plane", "gicp", "trimmed_icp", "geop2e_e8", "geop2e_all", "geop2e_cal_oracle"]
out = {"criterion": GEO_SUCC, "per_part": {}, "pooled": {}}
for pk, p in d["parts"].items():
    rows = {}
    for m in methods:
        rr = [t[m] for t in p["trials"] if m in t and "median" in t[m]]
        if not rr:
            continue
        med = [r["median"] for r in rr]
        p95 = [r["p95"] for r in rr]
        rre = [r.get("rre_deg", -1) for r in rr]
        rt = [r.get("runtime_s", 0) for r in rr]
        rows[m] = dict(n=len(rr), median_mean=float(np.mean(med)), median_max=float(np.max(med)),
                       p95_med=float(np.median(p95)), geo_success=float(np.mean([(a < GEO_SUCC["median"] and b < GEO_SUCC["p95"]) for a, b in zip(med, p95)])),
                       rre_med_deg=float(np.median(rre)), runtime_med_s=float(np.median(rt)))
    out["per_part"][pk] = dict(cad=d["protocol"]["parts"].get(pk.replace("scan", ""), "?"),
                               ref=p["ref_p2m"], n_sym=p.get("n_symmetries"), rows=rows)
allr = {}
for m in methods:
    med, p95, rt = [], [], []
    for pk, p in d["parts"].items():
        for t in p["trials"]:
            if m in t and "median" in t[m]:
                med.append(t[m]["median"]); p95.append(t[m]["p95"]); rt.append(t[m].get("runtime_s", 0))
    if med:
        allr[m] = dict(n=len(med), median_mean=float(np.mean(med)), median_max=float(np.max(med)),
                       p95_med=float(np.median(p95)),
                       geo_success=float(np.mean([(a < GEO_SUCC["median"] and b < GEO_SUCC["p95"]) for a, b in zip(med, p95)])),
                       runtime_med_s=float(np.median(rt)))
out["pooled"] = allr
o = SRC.replace(".json", "_scored.json")
json.dump(out, open(o, "w"), indent=1)
print(f"{'method':18s} {'n':>3s} {'med(mean)':>9s} {'med(max)':>8s} {'p95med':>7s} {'geoSucc':>7s} {'rt(s)':>6s}")
for m, r in allr.items():
    print(f"{m:18s} {r['n']:3d} {r['median_mean']:9.3f} {r['median_max']:8.3f} {r['p95_med']:7.2f} {r['geo_success']*100:6.0f}% {r['runtime_med_s']:6.1f}")
print("saved", o)
