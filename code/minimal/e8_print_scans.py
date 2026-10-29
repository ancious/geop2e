"""e8 — first-pass inspection + registration of the 5 physically printed & scanned parts.

Exploratory first look (NOT the paper-grade protocol; formal E8 will be
pre-registered and pass the GPT/Opus gate before any number enters the paper).

v2 fixes over the aborted first run:
  - FGR alone fails on feature-poor large cylinders (best median 13.8 mm = misaligned).
    Matching now uses multi-start: PCA 4-proper-flip coarse alignments + FGR,
    each refined by progressive ICP (15 -> 5 -> 2 mm), best by p2m median.
  - ICP target = per-CAD downsampled eval cloud WITH normals (voxel 1.0 mm);
    p2m evaluated against the 1M-pt STEP-sampled cloud (no normals needed).
  - GeoP2E stage uses TrimmedCAD(part).feats / .pts (not part['feats']).
Pipeline per scan: load -> outlier-clean -> voxel 0.5 -> match vs 5 CADs ->
GeoP2E production (parsed primitives, matching-best init, auto_params) vs
FGR/PCA+ICP p2plane baseline -> median/p95 point-to-mesh mm.
Never overwrites: results/e8_print_scans_firstpass.json (+ .N if exists).
"""
import os, sys, json, time
import numpy as np
import open3d as o3d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import e6_auto as A
from solver import register

DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "五件")
PACK = os.path.join(os.path.dirname(HERE), "print_pack_20_v2")
OUT = os.path.join(HERE, "results", "e8_print_scans_firstpass.json")
FB = ["52890_7ae554d9_1", "21644_aa203dc5_3", "29726_73b01e1f_0",
      "21638_0a984848_73", "25365_14519ce1_0"]
VOX_WORK, VOX_FGR, VOX_TGT = 0.5, 2.5, 1.0
EVAL_SAMPLE = 1_000_000


def load_scan(i):
    p = os.path.join(DATA, f"{i}.asc")
    skip = 0
    with open(p) as f:
        for _ in range(5):
            line = f.readline()
            if line.startswith("#") or line.strip() == "":
                skip += 1
            else:
                break
    a = np.loadtxt(p, skiprows=skip)
    pts, nrm = a[:, :3].copy(), a[:, 3:6].copy()
    ok = np.isfinite(pts).all(1) & np.isfinite(nrm).all(1) & (np.linalg.norm(nrm, axis=1) > 0.5)
    pts, nrm = pts[ok], nrm[ok] / np.linalg.norm(nrm[ok], axis=1, keepdims=True)
    return pts, nrm, len(a)


def clean_downsample(pts, nrm):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc.normals = o3d.utility.Vector3dVector(nrm)
    n0 = len(pts)
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    n1 = len(pc.points)
    work = pc.voxel_down_sample(VOX_WORK)
    if not work.has_normals():
        work.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=30))
    return work, (n0, n1, len(work.points))


def cad_clouds(stem):
    """(1M eval cloud from STEP re-meshed at 0.02mm, downsampled ICP target with normals)."""
    from OCP.STEPControl import STEPControl_Reader
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.StlAPI import StlAPI_Writer
    from OCP.IFSelect import IFSelect_RetDone
    cache = os.path.join(HERE, "artifacts", f"e8_eval_{stem}_0p02.stl")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    if not os.path.exists(cache):
        r = STEPControl_Reader()
        assert r.ReadFile(os.path.join(PACK, f"{stem}_scaled.step")) == IFSelect_RetDone
        r.TransferRoots()
        shape = r.OneShape()
        BRepMesh_IncrementalMesh(shape, 0.02, False, 0.1, True)
        w = StlAPI_Writer(); w.ASCIIMode = False
        assert w.Write(shape, cache)
    m = o3d.io.read_triangle_mesh(cache)
    npz = cache.replace('.stl', '_eval.npz')
    if os.path.exists(npz):
        arr = np.load(npz)['pts']
    else:
        arr = np.asarray(m.sample_points_uniformly(number_of_points=EVAL_SAMPLE).points)
        np.savez_compressed(npz, pts=arr)   # deterministic reuse across runs
    ev = o3d.geometry.PointCloud()
    ev.points = o3d.utility.Vector3dVector(arr)
    tgt = ev.voxel_down_sample(VOX_TGT)
    tgt.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=VOX_TGT * 6, max_nn=30))
    return ev, tgt, m.get_surface_area()


def pca_candidates(src_np, tgt_np, src_pc, tgt_pc, tgt_ds_voxel):
    """PCA 4-proper-flip coarse alignments (scale known: both in mm)."""
    cs, ct = src_np.mean(0), tgt_np.mean(0)
    As = np.linalg.svd(src_np - cs, full_matrices=False)[2].T       # columns = scan axes
    At = np.linalg.svd(tgt_np - ct, full_matrices=False)[2].T
    # handedness canonicalization (2026-09-07 fix; see e8e_bench.pca24)
    if np.linalg.det(As) < 0:
        As[:, 2] *= -1
    if np.linalg.det(At) < 0:
        At[:, 2] *= -1
    cands = []
    for f in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
        R = At @ np.diag(f) @ As.T
        if np.linalg.det(R) < 0:
            continue
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = ct - R @ cs
        cands.append(T)
    return cands


def fgr_T(src_pc, tgt_pc, voxel, max_corr):
    sd = src_pc.voxel_down_sample(voxel)
    td = tgt_pc.voxel_down_sample(voxel)
    for p_ in (sd, td):
        if not p_.has_normals():
            p_.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 6, max_nn=30))
    fs = o3d.pipelines.registration.compute_fpfh_feature(
        sd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
    ft = o3d.pipelines.registration.compute_fpfh_feature(
        td, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
    opt = o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=max_corr, use_absolute_scale=True)
    try:
        return o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            sd, td, fs, ft, opt).transformation, True
    except Exception:
        return np.eye(4), False


def prog_icp(src_pc, tgt_pc, T0, thrs=(15.0, 5.0, 2.0)):
    T = T0.copy()
    for thr in thrs:
        T = o3d.pipelines.registration.registration_icp(
            src_pc, tgt_pc, thr, T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=80)).transformation
    return T


def p2m(src_pts, T, tgt_eval):
    q = o3d.geometry.PointCloud()
    q.points = o3d.utility.Vector3dVector(src_pts @ T[:3, :3].T + T[:3, 3])
    d = np.asarray(q.compute_point_cloud_distance(tgt_eval))
    return dict(median=float(np.median(d)), p95=float(np.percentile(d, 95)),
                inlier_pct=float(100 * np.mean(d < 1.0)))


def main():
    res = {"protocol": dict(vox_work=VOX_WORK, matching="best of {FGR, PCA-4flip} + progICP 15/5/2 vs STEP 1M cloud",
                            outlier="statistical 20/2.0", eval_sample=EVAL_SAMPLE, eval_mesh="STEP re-meshed 0.02mm",
                            note="EXPLORATORY first pass; not paper-grade"), "parts": {}}
    cad = {}
    for stem in FB:
        t0 = time.perf_counter()
        ev, tgt, area = cad_clouds(stem)
        cad[stem] = (ev, tgt, area)
        print(f"CAD {stem[:20]:20s} area={area:.0f}mm2 tgt={len(tgt.points):,} ({time.perf_counter()-t0:.0f}s)", flush=True)
    for i in range(1, 6):
        t0 = time.perf_counter()
        pts, nrm, n_rows = load_scan(i)
        work, (n0, n1, n2) = clean_downsample(pts, nrm)
        src = np.asarray(work.points); sn = np.asarray(work.normals)
        print(f"scan {i}: rows={n_rows:,} -> clean {n1:,} -> work {n2:,} ({time.perf_counter()-t0:.0f}s)", flush=True)
        rec = dict(n_rows=n_rows, n_clean=n1, n_work=n2, scan_bbox=np.round(np.ptp(src, axis=0), 1).tolist())
        scores = {}
        for stem in FB:
            ev, tgt, _ = cad[stem]
            tn = np.asarray(tgt.points)
            t0c = time.perf_counter()
            cands = pca_candidates(src, tn, work, tgt, VOX_FGR)
            Tf, fgr_ok = fgr_T(work, tgt, VOX_FGR, 5.0)
            cands.append(Tf)
            best = None
            for T0 in cands:
                T = prog_icp(work, tgt, T0)
                m = p2m(src, T, ev)
                if best is None or m["median"] < best[1]["median"]:
                    best = (T, m, "pca" if T0 is not Tf else "fgr")
            scores[stem] = dict(T=best[0].tolist(), init=best[2], fgr_ok=fgr_ok,
                                runtime_s=time.perf_counter() - t0c, **best[1])
            print(f"   vs {stem[:20]:20s} [{best[2]:3s}] median={best[1]['median']:7.3f}mm "
                  f"p95={best[1]['p95']:7.2f} in1mm={best[1]['inlier_pct']:5.1f}%", flush=True)
        best_stem = min(scores, key=lambda s: scores[s]["median"])
        rec["match_scores"] = {k: {kk: vv for kk, vv in v.items() if kk != "T"} for k, v in scores.items()}
        rec["matched_cad"] = best_stem
        print(f"scan {i} MATCHED -> {best_stem} (median {scores[best_stem]['median']:.3f}mm)", flush=True)
        # ---- GeoP2E production on matched CAD (init = best matching T) ----
        ev, _, area = cad[best_stem]
        part = A.load_parsed_part(os.path.join(PACK, f"{best_stem}_scaled.step"))
        tcad = A.TrimmedCAD(part)
        tgt_pts = tcad.pts
        ext = float(np.linalg.norm(np.ptp(tgt_pts, axis=0)))
        REG = dict(init_radius=ext * 0.18, anneal=0.9, max_iter=60, lam=1e-3,
                   k_huber=ext * 0.012, angle_thresh_deg=30.0)
        guides = [m["pts"][np.random.default_rng(1).choice(len(m["pts"]), min(200, len(m["pts"])), replace=False)]
                  for m in part["supports"]]
        from selection import greedy_select
        sel = greedy_select(tcad.feats, guides, criterion="e", target_size=8,
                            min_features=5, rng=np.random.default_rng(0))
        feats = [tcad.feats[j] for j in sel]
        T0 = np.array(scores[best_stem]["T"])
        rec["coarse_init_p2m"] = p2m(src, T0, ev)
        t0g = time.perf_counter()
        try:
            T_geo, info = register(src, sn, feats, init_T=T0.copy(), **REG)
            rec["geop2e"] = dict(runtime_s=time.perf_counter() - t0g, iters=info.get("iters"),
                                 n_feats=len(feats), **p2m(src, T_geo, ev))
        except Exception as ex:
            rec["geop2e"] = {"error": repr(ex)[:150]}
        # baseline: progressive p2plane ICP (same init) vs support cloud
        tgt_pc = o3d.geometry.PointCloud(); tgt_pc.points = o3d.utility.Vector3dVector(tgt_pts)
        tgt_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=VOX_TGT * 6, max_nn=30))
        t0b = time.perf_counter()
        T_icp = prog_icp(work, tgt_pc, T0, (5.0, 2.0, 1.0))
        rec["icp_baseline"] = dict(runtime_s=time.perf_counter() - t0b, **p2m(src, T_icp, ev))
        res["parts"][f"scan{i}"] = rec
        g = rec.get("geop2e", {})
        print(f"  coarse={rec['coarse_init_p2m']['median']:.3f} | GeoP2E median={g.get('median', -1):.3f}mm "
              f"p95={g.get('p95', -1):.2f} | ICP-baseline {rec['icp_baseline']['median']:.3f}mm "
              f"[{time.perf_counter()-t0:.0f}s total]", flush=True)
        json.dump(res, open(OUT, "w"), default=float, indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
