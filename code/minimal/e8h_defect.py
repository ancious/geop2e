"""e8h — defect-regime benchmark on the 5 real printed-part scans (GeoP2E's
home scenario per the paper's core claim). Companion to e8e (intact regime).

Protocol: identical to e8e v3 EXCEPT each perturbed trial cloud is cropped by a
random half-space at the retention quantile (70% and 50% levels), simulating
structural loss on REAL scan data. All methods refine from the SAME shared
coarse init recomputed on the cropped cloud (PCA-24 + prog ICP). Metrics:
p2m median/p95 vs the FULL CAD 1M cloud (misalignment of the intact region
dominates), geometric success (median<0.5mm AND p95<2.0mm), runtime.
RRE omitted (symmetry ambiguity; e8g handles scoring).
Output: results/e8h_defect.json (never overwrites).
"""
import os, sys, json, time
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import e8_print_scans as E
import e6_auto as A
import e8e_bench as B
from solver import register
from selection import greedy_select
from baselines import trimmed_icp

PARTS = B.PARTS
N_TRIALS = 10
SEED0 = 20260906
RETENTIONS = (0.7, 0.5)
GEO_REG = B.GEO_REG
CALIB = B.CALIB
OUT = os.path.join(HERE, "results", "e8h_defect.json")


def main():
    global OUT
    if os.path.exists(OUT):
        OUT = OUT.replace(".json", f".{int(time.time())}.json")
    res = {"protocol": dict(parts=PARTS, n_trials=N_TRIALS, retentions=RETENTIONS,
                            seed0=SEED0, geo_reg=GEO_REG, calib=CALIB,
                            crop="random-axis half-space at retention quantile, applied to perturbed cloud",
                            init="shared PCA-24 + prog ICP on cropped cloud",
                            metrics="p2m vs FULL CAD 1M cloud; geometric success median<0.5 AND p95<2.0"),
           "parts": {}}
    for si, stem in PARTS.items():
        t0p = time.perf_counter()
        pts, nrm, _ = E.load_scan(si)
        work, _ = E.clean_downsample(pts, nrm)
        src = np.asarray(work.points); sn = np.asarray(work.normals)
        ev, tgt, _ = E.cad_clouds(stem)
        tgt_3mm = tgt.voxel_down_sample(3.0)
        part = A.load_parsed_part(os.path.join(E.PACK, f"{stem}_scaled.step"))
        tcad = A.TrimmedCAD(part)
        ext = float(np.linalg.norm(np.ptp(tcad.pts, axis=0)))
        guides = [m["pts"][np.random.default_rng(1).choice(len(m["pts"]), min(200, len(m["pts"])), replace=False)]
                  for m in part["supports"]]
        sel = greedy_select(tcad.feats, guides, criterion="e", target_size=8,
                            min_features=5, rng=np.random.default_rng(0))
        feats8 = [tcad.feats[j] for j in sel]
        feats_all = list(tcad.feats)
        REG = dict(GEO_REG, k_huber=ext * 0.012)
        prec = {"ext": ext, "trials": []}
        res["parts"][f"scan{si}"] = prec
        rng = np.random.default_rng(SEED0 + si * 1000)
        for tr in range(N_TRIALS * len(RETENTIONS)):
            ret = RETENTIONS[tr // N_TRIALS]
            ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
            ang = rng.uniform(*B.ROT_RANGE)
            Rg = Rot.from_rotvec(ax * np.deg2rad(ang)).as_matrix()
            tg = rng.uniform(-1, 1, 3) * B.TRANS_FRAC * ext
            pts_p = src @ Rg.T + tg
            nrm_p = sn @ Rg.T
            # defect: half-space crop on random axis at retention quantile
            ca = rng.normal(size=3); ca /= np.linalg.norm(ca)
            q = np.quantile(pts_p @ ca, ret)
            keep = (pts_p @ ca) >= q
            pts_c, nrm_c = pts_p[keep], nrm_p[keep]
            w_c = o3d.geometry.PointCloud()
            w_c.points = o3d.utility.Vector3dVector(pts_c)
            w_c.normals = o3d.utility.Vector3dVector(nrm_c)
            s3_c = w_c.voxel_down_sample(3.0)
            if not s3_c.has_normals():
                s3_c.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8.0, max_nn=30))
            row = dict(trial=tr % N_TRIALS, retention=ret, ang_deg=float(ang), n_pts=int(keep.sum()))
            t0 = time.perf_counter()
            T0 = B.coarse_init(w_c, tgt, s3_c, tgt_3mm)
            row["init"] = dict(runtime_s=time.perf_counter() - t0, **E.p2m(pts_c, T0, ev))
            cal = CALIB[si]
            feats_cal = feats8 if cal["sel"] == "e8" else feats_all
            methods = {}
            t0 = time.perf_counter()
            methods["icp_p2plane"] = (o3d.pipelines.registration.registration_icp(
                w_c, tgt, 2.0, T0,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=100)).transformation,
                time.perf_counter() - t0)
            t0 = time.perf_counter()
            Tm, _ = trimmed_icp(pts_c, np.asarray(tgt.points), init=T0, threshold=3.0, trim_frac=0.3, max_iter=50)
            methods["trimmed_icp"] = (Tm, time.perf_counter() - t0)
            for name, feats, rg in (("geop2e_e8", feats8, REG),
                                    ("geop2e_cal_oracle", feats_cal,
                                     dict(REG, lam=cal["lam"], angle_thresh_deg=cal["ang"]))):
                t0 = time.perf_counter()
                try:
                    Tm, _ = register(pts_c, nrm_c, feats, init_T=T0.copy(), **rg)
                    methods[name] = (Tm, time.perf_counter() - t0)
                except Exception as ex:
                    methods[name] = (np.eye(4), time.perf_counter() - t0, repr(ex)[:80])
            for name, val in methods.items():
                Tm, rt = val[0], val[1]
                m = E.p2m(pts_c, Tm, ev)
                row[name] = dict(runtime_s=rt, **m,
                                 success=bool(m["median"] < 0.5 and m["p95"] < 2.0))
                if len(val) > 2:
                    row[name]["error"] = val[2]
            prec["trials"].append(row)
            print(f"  tr{tr%N_TRIALS:02d} ret={ret:.1f} n={keep.sum():6d} | " +
                  " ".join(f"{k}={row[k]['median']:.3f}" for k in ("icp_p2plane", "geop2e_cal_oracle") if k in row),
                  flush=True)
            json.dump(res, open(OUT, "w"), default=float, indent=1)
        print(f"[{stem[:18]}] done ({time.perf_counter()-t0p:.0f}s)", flush=True)
    print("saved", OUT)


if __name__ == "__main__":
    main()
