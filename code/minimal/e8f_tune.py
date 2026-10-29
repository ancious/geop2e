"""e8f — GeoP2E tuning pass on the 5 printed-part scans (as-is alignment quality).

Mandate (user 2026-09-05): optimize parameters AND feature selection per the
real parts until best achievable registration; the benchmark (e8e) will then
run the winning configuration as the 'GeoP2E (tuned)' arm alongside the
published-config arm. Tuning is done on these same 5 demonstration parts and
this provenance will be disclosed (engineering-application framing; fresh
batch validation is queued for the paper).

Grid per part:
- selection: e-opt-8 (published), all-primitives, A3-normalized-FIM auto (per-part)
- init_radius: 3.0 / 2.0 mm      (0.18*diag already known-divergent)
- anneal: 0.9 / 0.85, max_iter 60/100
Criterion: p2m median vs 1M STEP cloud + pose deviation vs the dense-ICP T_ref
(RRE/TTE) — both reported; selection by p2m median, tie-broken by RRE.
Init: the dense-ICP T_ref itself for tuning (measures the solver's OWN optimum,
decoupled from init quality; the benchmark measures recovery separately).
Output: results/e8f_tune.json (+ per-part best config).
"""
import os, sys, json, time, itertools
import numpy as np
import open3d as o3d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import e8_print_scans as E
import e6_auto as A
from solver import register
from selection import greedy_select
import e8e_bench as B

PARTS = B.PARTS
OUT = os.path.join(HERE, "results", "e8f_tune.json")


def main():
    res = {"note": "tuning on the 5 demo parts; provenance disclosed", "parts": {}}
    best_cfg = {}
    for si, stem in PARTS.items():
        t0 = time.perf_counter()
        pts, nrm, _ = E.load_scan(si)
        work, _ = E.clean_downsample(pts, nrm)
        src = np.asarray(work.points); sn = np.asarray(work.normals)
        ev, tgt, _ = E.cad_clouds(stem)
        part = A.load_parsed_part(os.path.join(E.PACK, f"{stem}_scaled.step"))
        tcad = A.TrimmedCAD(part)
        ext = float(np.linalg.norm(np.ptp(tcad.pts, axis=0)))
        # T_ref: dense ICP anchor (same construction as e8e)
        tgt_3mm = tgt.voxel_down_sample(3.0); src_3mm = work.voxel_down_sample(3.0)
        s3 = np.asarray(src_3mm.points)
        best = None
        for T0 in B.pca24(s3, np.asarray(tgt_3mm.points)):
            Tl = B.light_icp(src_3mm, tgt_3mm, T0)
            q = o3d.geometry.PointCloud()
            q.points = o3d.utility.Vector3dVector(s3 @ Tl[:3, :3].T + Tl[:3, 3])
            d = np.asarray(q.compute_point_cloud_distance(tgt_3mm))
            med = float(np.median(d))
            if best is None or med < best[1]:
                best = (Tl, med)
        T_ref = E.prog_icp(work, tgt, best[0], (15.0, 5.0, 2.0, 1.0))
        ref_m = E.p2m(src, T_ref, ev)
        # selection variants
        guides = [m["pts"][np.random.default_rng(1).choice(len(m["pts"]), min(200, len(m["pts"])), replace=False)]
                  for m in part["supports"]]
        sel = greedy_select(tcad.feats, guides, criterion="e", target_size=8,
                            min_features=5, rng=np.random.default_rng(0))
        sels = {"e8": [tcad.feats[j] for j in sel], "all": list(tcad.feats)}
        a3_name = None
        try:
            def val_fn(idx):
                """Quick register (30 it) on the subset; robust score vs its trimmed CAD."""
                feats_v = [tcad.feats[j] for j in idx] if idx else list(tcad.feats)
                sub = A.TrimmedCAD_from_subset(tcad, idx) if idx else tcad
                try:
                    T, _ = register(src, sn, feats_v, init_T=T_ref.copy(),
                                    init_radius=3.0, anneal=0.9, max_iter=30,
                                    lam=1e-3, k_huber=ext * 0.012, angle_thresh_deg=30.0)
                    p = src @ T[:3, :3].T + T[:3, 3]
                    d, _ = sub.d_and_label(p)
                    return A.robust_score(d, ext * 0.012), 1e-4
                except Exception:
                    return 1e9, 1e-4
            pick, _table = A.select_auto(part, sigma=0.0024 * ext, val_score_fn=val_fn)
            idx = pick.get("idx")
            sels["a3"] = list(tcad.feats) if (pick["name"] == "all" or not idx) \
                else [tcad.feats[j] for j in idx]
            a3_name = f"{pick['name']}(n={pick.get('n')},alarm={pick.get('alarm')})"
        except Exception as ex:
            a3_name = f"fallback:{ex!r:.60}"
            sels["a3"] = list(tcad.feats)
        rows = []
        for sname, feats in sels.items():
            for rad, ann, mi in itertools.product((3.0, 2.0), (0.9, 0.85), (60, 100)):
                REG = dict(init_radius=rad, anneal=ann, max_iter=mi, lam=1e-3,
                           k_huber=ext * 0.012, angle_thresh_deg=30.0)
                t0g = time.perf_counter()
                try:
                    Tg, info = register(src, sn, feats, init_T=T_ref.copy(), **REG)
                    m = E.p2m(src, Tg, ev)
                    rre, tte = B.pose_err(Tg, T_ref)
                    rows.append(dict(sel=sname, n_feats=len(feats), rad=rad, anneal=ann,
                                     max_iter=mi, median=m["median"], p95=m["p95"],
                                     rre_deg=rre, tte_mm=tte, iters=info.get("iters"),
                                     runtime_s=time.perf_counter() - t0g))
                except Exception as ex:
                    rows.append(dict(sel=sname, rad=rad, anneal=ann, max_iter=mi,
                                     error=repr(ex)[:80]))
        ok = [r for r in rows if "median" in r]
        ok.sort(key=lambda r: (round(r["median"], 4), r["rre_deg"]))
        b = ok[0]
        best_cfg[stem] = b
        res["parts"][f"scan{si}"] = dict(ref_median=ref_m["median"], a3_name=a3_name,
                                         rows=rows, best=b)
        top5 = "; ".join(f"{r['sel']}/r{r['rad']}/a{r['anneal']}/i{r['max_iter']}={r['median']:.3f}"
                         for r in ok[:5])
        print(f"[{stem[:16]}] ref={ref_m['median']:.3f} | top: {top5} ({time.perf_counter()-t0:.0f}s)", flush=True)
        json.dump(res, open(OUT, "w"), default=float, indent=1)
    res["best_cfg"] = best_cfg
    json.dump(res, open(OUT, "w"), default=float, indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
