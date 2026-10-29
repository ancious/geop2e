"""E3d — reanalysis per GPT-sol-max verification prescriptions (2026-08-27).

From the recorded 320-run rows (results/e3_hv_ablation_full.json):
  1. RAW-vs-SYM endpoint table: per cell/arm success under the RAW endpoint
     (RRE_raw<1deg & RTE_raw<1mm) alongside the symmetry-aware endpoint, and how
     many rows changed endpoint class (the judge found 53/200 changed; verify).
  2. Actual activation + hypothesis counts per cell (from full_hv rows:
     triggered rate, mean/max n_hypotheses — the judge found 9 not 13; verify).
  3. Runtime ratio full_hv/no_hv when BOTH arms activated-trigger cells.
  4. E5 reanalysis: per-scan fixed-scan initialization sensitivity with
     median/SD/IQR/range/bootstrap-CI, max-iteration-capped frequency, overlap,
     runtime — labelled exactly 'fixed-scan computational initialization
     sensitivity' (no acquisition/metrology implication).
No new registrations. Pure reanalysis; never overwrites (timestamped suffix).
"""
import os, sys, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "results", "e3_hv_ablation_full.json")


def boot_ci(v, stat=np.median, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    if len(v) == 0:
        return [None, None]
    bs = [float(stat(v[rng.integers(0, len(v), len(v))])) for _ in range(B)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    d = json.load(open(SRC))
    rows = d["rows"]
    out = {"source": os.path.basename(SRC)}

    # 1-3: synthetic cells
    cells = sorted({r["cell"] for r in rows if r["cell"].startswith("loss")})
    synth = {}
    changed_total = 0
    for c in cells:
        cr = [r for r in rows if r["cell"] == c and r["ok"]]
        e = {}
        for arm in ("no_hv", "full_hv"):
            ar = [r for r in cr if r["arm"] == arm]
            succ_sym = sum(r["success"] for r in ar)
            succ_raw = sum(r["rre_raw"] < 1.0 and r["rte_raw"] < 1.0 for r in ar)
            e[arm] = dict(n=len(ar), success_sym=f"{succ_sym}/{len(ar)}",
                          success_raw=f"{succ_raw}/{len(ar)}",
                          rows_sym_only=sum(1 for r in ar if r["success"] and not (r["rre_raw"] < 1.0 and r["rte_raw"] < 1.0)),
                          mean_runtime_s=float(np.mean([r["runtime_s"] for r in ar])))
            if arm == "full_hv":
                e[arm].update(triggered=f"{sum(r['triggered'] for r in ar)}/{len(ar)}",
                              mean_n_hypotheses=float(np.mean([r["n_hypotheses"] for r in ar])),
                              max_n_hypotheses=int(max(r["n_hypotheses"] for r in ar)),
                              mean_n_cand_evaluated=float(np.mean([r["n_cand_evaluated"] for r in ar])))
            changed_total += e[arm]["rows_sym_only"]
        t_full = np.array([r["runtime_s"] for r in cr if r["arm"] == "full_hv"])
        t_no = np.array([r["runtime_s"] for r in cr if r["arm"] == "no_hv"])
        e["runtime_ratio_full/no (median)"] = float(np.median(t_full) / max(np.median(t_no), 1e-12))
        synth[c] = e
    out["synthetic"] = synth
    out["rows_whose_success_requires_symmetry_quotient"] = changed_total

    # 4: E5 reanalysis from real full_hv rows
    e5 = {}
    for c in sorted({r["cell"] for r in rows if r["cell"].startswith("real_")}):
        ar = [r for r in rows if r["cell"] == c and r["arm"] == "full_hv" and r["ok"]]
        v = np.array([r["p2m_median_mm"] for r in ar])
        e5[c] = dict(n=len(ar),
                     p2m_median=float(np.median(v)), p2m_sd=float(np.std(v)),
                     p2m_iqr=[float(np.percentile(v, 25)), float(np.percentile(v, 75))],
                     p2m_range=[float(v.min()), float(v.max())],
                     p2m_median_boot95=boot_ci(v),
                     inlier_pct_median=float(np.median([r["inlier_pct"] for r in ar])),
                     runtime_s_median=float(np.median([r["runtime_s"] for r in ar])),
                     hit_max_iter=f"{sum(1 for r in ar if r.get('iters') == 60)}/{len(ar)}",
                     nonzero_final_update=f"{sum(1 for r in ar if (r.get('update_norm') or 0) > 1e-6)}/{len(ar)}",
                     label="fixed-scan computational initialization sensitivity "
                           "(NOT acquisition repeatability / metrological uncertainty)")
    out["e5_reanalysis"] = e5

    p = os.path.join(HERE, "results", "e3d_reanalysis.json")
    if os.path.exists(p):
        p += f".{os.getctime(p):.0f}.json"
    json.dump(out, open(p, "w"), indent=2, default=float)
    print(f"saved {p}")
    print(json.dumps(out["synthetic"], indent=1, default=float)[:1500])
    print("rows requiring symmetry quotient:", changed_total)


if __name__ == "__main__":
    main()
