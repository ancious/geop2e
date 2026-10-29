"""E4 — Cranfield non-gear real-data study (R1: general applicability beyond gears).

All 50 filtered scans across 9 industrial parts, methods: GeoP2E (mesh-fitted
primitives), FGR (+ICP p2plane) context baseline, zero-shot RPMNet (guji adapter).
Disclosed limitation (title-level): STL meshes -> RANSAC-fitted primitives, NOT
native B-Rep parsing; GT poses are the benchmark's ICP-refined estimates.
Mirror/near-180deg outcomes are reported explicitly (3dphy's related method
failed systematically there). Per-scan + per-part + aggregate; no pooled
superiority claims.

Frozen Cranfield config (units METERS): FGR voxel 0.005, ICP thr 0.01 (repo's
own recall limit), solver params scale-derived via auto_params.
Usage: python e4_cranfield.py [--skip-rpmnet] [--fit-only]
"""
import os, sys, json, time, glob, subprocess, uuid
import numpy as np
import open3d as o3d

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PARENT)

import e0_common as e0
from features import PlaneFeature, CylinderFeature, SphereFeature
from solver import estimate_normals
from selection import greedy_select, spectral, selected_info
from e3_hv_ablation import auto_params, make_reg_fn, verify_shared_init

CR = e0.CRANFIELD
SCANS_DIR = os.path.join(CR, "datasets/point_clouds/filtered")
CAD_DIR = os.path.join(CR, "datasets/CAD")
GT_DIR = os.path.join(CR, "datasets/ground_truth/filtered")
CFG = {"fgr_voxel": 0.005, "fgr_max_corr": 0.01, "icp_thr": 0.01,
       "ransac_plane_thr": 0.002, "ransac_min_pts_frac": 0.02,
       "max_planes": 12, "max_cyls": 8, "max_spheres": 4,
       "cad_sample": 20000, "scan_voxel": 0.002}
GUJI_PY = os.path.expanduser("~/miniconda3/envs/guji/bin/python")
RPMNET_ADAPTER = os.path.join(HERE, "e1c_rpmnet.py")


# --------------------------------------------------------------- primitive fit
def fit_primitives(mesh_path, cache_dir):
    """Iterative RANSAC plane/cylinder/sphere extraction from the CAD mesh.
    Cached per part (npz inlier masks are big; cache stores fitted params +
    coverage stats). Returns (feats, stats)."""
    part = os.path.splitext(os.path.basename(mesh_path))[0]
    cache = os.path.join(cache_dir, f"fit_{part}.json")
    if os.path.exists(cache):
        d = json.load(open(cache))
        # records are in RAW STL coords; transformed at build time by caller via prep
        return d["records"], d["stats"]
    import pyransac3d as p3d
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    pts = np.asarray(mesh.sample_points_uniformly(number_of_points=50000).points)
    remaining = pts.copy()
    records = []
    # return structures (introspected): Plane.fit -> (eq[a,b,c,d], inlier_idx);
    # Cylinder.fit -> (center, axis, radius, inlier_idx); Sphere.fit -> (center, radius, inlier_idx)
    for kind, max_n in (("plane", CFG["max_planes"]), ("cylinder", CFG["max_cyls"]),
                        ("sphere", CFG["max_spheres"])):
        for _ in range(max_n):
            min_pts = CFG["ransac_min_pts_frac"] * len(pts)
            if len(remaining) < min_pts:
                break
            try:
                if kind == "plane":
                    eq, idx = p3d.Plane().fit(remaining, CFG["ransac_plane_thr"])
                    nrm = np.array(eq[:3], float); nrm /= np.linalg.norm(nrm)
                    d = float(eq[3]) / np.linalg.norm(np.array(eq[:3], float))
                    idx = np.asarray(idx, int)
                    if len(idx) < min_pts:
                        break
                    records.append(dict(kind="plane", normal=nrm.tolist(), d=d, n_inlier=int(len(idx))))
                elif kind == "cylinder":
                    center, axis, radius, idx = p3d.Cylinder().fit(remaining, CFG["ransac_plane_thr"])
                    center = np.array(center, float); axis = np.array(axis, float)
                    axis /= np.linalg.norm(axis); radius = float(radius)
                    idx = np.asarray(idx, int)
                    if not np.isfinite(radius) or radius <= 0 or len(idx) < min_pts:
                        break
                    records.append(dict(kind="cylinder", center=center.tolist(),
                                        axis=axis.tolist(), radius=radius, n_inlier=int(len(idx))))
                else:
                    center, radius, idx = p3d.Sphere().fit(remaining, CFG["ransac_plane_thr"])
                    center = np.array(center, float); radius = float(radius)
                    idx = np.asarray(idx, int)
                    if not np.isfinite(radius) or radius <= 0 or len(idx) < min_pts:
                        break
                    records.append(dict(kind="sphere", center=center.tolist(),
                                        radius=radius, n_inlier=int(len(idx))))
            except Exception:
                break
            keep = np.ones(len(remaining), bool); keep[idx] = False
            remaining = remaining[keep]
    covered = 1.0 - len(remaining) / len(pts)
    stats = dict(part=part, n_primitives=len(records),
                 kinds={k: sum(1 for r in records if r["kind"] == k) for k in ("plane", "cylinder", "sphere")},
                 coverage_frac=round(float(covered), 4),
                 frame="raw STL coordinates (transformed to prepared frame at load)",
                 note="RANSAC-fitted from sampled mesh; NOT native B-Rep")
    json.dump({"records": records, "stats": stats}, open(cache, "w"), indent=2)
    return records, stats


def feats_from_records(records, prep=None):
    """Build features; if prep=(c, s) is given, transform RAW-STL-frame primitive
    parameters into the PREPARED template frame (final-verification fix: the fit
    cache lives in raw STL coordinates, registration runs on p->(p-c)/s):
      plane:    d' = (d + n·c)/s           (normal unchanged)
      cyl/sph:  q' = (q - c)/s,  r' = r/s  (axis unchanged)
    """
    feats = []
    for r in records:
        if r["kind"] == "plane":
            n = np.array(r["normal"]); d = r["d"]
            if prep is not None:
                c, s = prep
                d = (d + float(n @ c)) / s
            feats.append(PlaneFeature(n, d))
        elif r["kind"] == "cylinder":
            q = np.array(r["center"]); ax = np.array(r["axis"]); rad = r["radius"]
            if prep is not None:
                c, s = prep
                q = (q - c) / s; rad = rad / s
            feats.append(CylinderFeature(q, ax, rad))
        else:
            q = np.array(r["center"]); rad = r["radius"]
            if prep is not None:
                c, s = prep
                q = (q - c) / s; rad = rad / s
            feats.append(SphereFeature(q, rad))
    return feats


# --------------------------------------------------------------- GT handling
def prepare_template(mesh, n_sample, scale=1000.0):
    """Repo-exact template preparation (data_manager.prepare_stl): sample, then
    p -> (p - c)/scale with c = the sampled cloud's own center (open3d
    scale-about-center then translate(-center) nets to exactly this). Returns
    (prepared_points, prepared_mesh) where the mesh is transformed identically
    so p2m evaluation lives in the SAME prepared frame."""
    pc = mesh.sample_points_uniformly(number_of_points=n_sample)
    pts = np.asarray(pc.points)
    c = pts.mean(axis=0)
    prep = (pts - c) / scale
    mesh2 = o3d.geometry.TriangleMesh(mesh)
    mesh2.translate(-c)
    mesh2.scale(1.0 / scale, center=mesh2.get_center() * 0)  # about origin now
    return prep, mesh2, c


def load_gt(part, scan_nmb):
    """Benchmark GT: 'transformation' is (K,4,4) — K>=1 DECLARED solution matrices
    (K>1 = the benchmark's symmetric solution set; matches the repo's
    symmetric_errors gt_symm semantics). Returns the declared solution list."""
    p = os.path.join(GT_DIR, f"{part}_{scan_nmb}.json")
    d = json.load(open(p))
    T = np.array(d["transformation"], float)
    if T.ndim == 2:
        T = T[None]
    if T.ndim != 3 or T.shape[1:] != (4, 4):
        raise ValueError(f"unexpected GT shape {T.shape} in {p}")
    # 'frame': 'template to source'. Our solver outputs T_pred: source->template.
    # evaluate_alignment(T_pred, S) needs S = template->source = the DECLARED GT
    # itself (T_pred @ GT ~ I). Fix of the previous double-inversion error.
    return [T[i] for i in range(len(T))], sorted(d.keys())


def gt_error(T_pred, sols):
    """Min rotation error over the benchmark's GT solution set."""
    from solver import evaluate_alignment
    best = (np.inf, np.inf)
    for S in sols:
        r, t = evaluate_alignment(T_pred, S)
        if r < best[0]:
            best = (r, t)
    return best


# --------------------------------------------------------------- arms
def fgr_arm(src_pc, tgt_pc):
    sd = src_pc.voxel_down_sample(CFG["fgr_voxel"])
    td = tgt_pc.voxel_down_sample(CFG["fgr_voxel"])
    for p in (sd, td):
        p.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=CFG["fgr_voxel"] * 2, max_nn=30))
    fs = o3d.pipelines.registration.compute_fpfh_feature(sd, o3d.geometry.KDTreeSearchParamHybrid(radius=CFG["fgr_voxel"] * 5, max_nn=100))
    ft = o3d.pipelines.registration.compute_fpfh_feature(td, o3d.geometry.KDTreeSearchParamHybrid(radius=CFG["fgr_voxel"] * 5, max_nn=100))
    opt = o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=CFG["fgr_max_corr"], use_absolute_scale=True)
    return o3d.pipelines.registration.registration_fgr_based_on_feature_matching(sd, td, fs, ft, opt).transformation


def icp_refine(src_pc, tgt_pc, T0):
    est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    crit = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=100)
    return o3d.pipelines.registration.registration_icp(src_pc, tgt_pc, CFG["icp_thr"], T0, est, crit).transformation


def main(skip_rpmnet=False, fit_only=False):
    scan_files = sorted(glob.glob(os.path.join(SCANS_DIR, "*_Source.ply")))
    print(f"{len(scan_files)} filtered scans")
    parts = sorted({os.path.basename(f).rsplit("_", 2)[0] for f in scan_files})
    print("parts:", parts)

    cache_dir = os.path.join(HERE, "artifacts", "cranfield_fits")
    os.makedirs(cache_dir, exist_ok=True)
    fits = {}
    for part in parts:
        stl = os.path.join(CAD_DIR, f"{part}.stl")
        fits[part] = fit_primitives(stl, cache_dir)
        print(f"  fit {part}: {fits[part][1]['kinds']} coverage={fits[part][1]['coverage_frac']}")
    if fit_only:
        return

    # ---- canonical persisted inputs (verification-#2 prescriptions): per-part CAD
    # sample, per-scan preprocessed source (RPMNet pairs reuse the SAME source),
    # canonical per-scan FGR matrix (computed once, reused by FGR+ICP and GeoP2E)
    art = os.path.join(HERE, "artifacts", "cranfield_canon")
    os.makedirs(art, exist_ok=True)
    cad_samples = {}
    part_meshes = {}
    part_prep = {}
    for part in parts:
        npz = os.path.join(art, f"cad3_{part}.npz")
        # scale factor from the benchmark GT json (per-part; seen 100 and 1000)
        gscale = float(json.load(open(os.path.join(
            GT_DIR, f"{part}_1.json")))["scale"])
        m = o3d.io.read_triangle_mesh(os.path.join(CAD_DIR, f"{part}.stl"))
        m.compute_vertex_normals()
        if os.path.exists(npz):
            z = np.load(npz)
            tgt_pts, tgt_nrm, c = z["pts"], z["nrm"], z["c"]
            assert abs(float(z["scale"]) - gscale) < 1e-9
        else:
            pc0 = m.sample_points_uniformly(number_of_points=CFG["cad_sample"])
            c = np.asarray(pc0.points).mean(axis=0)          # CANONICAL center, persisted
            pts = (np.asarray(pc0.points) - c) / gscale
            pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(pts)
            pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=CFG["scan_voxel"] * 5, max_nn=30))
            tgt_pts, tgt_nrm = np.asarray(pc.points).copy(), np.asarray(pc.normals).copy()
            np.savez_compressed(npz, pts=tgt_pts, nrm=tgt_nrm, c=c, scale=gscale)
        # deterministic prepared mesh from the PERSISTED c (rerun-independent)
        prep_mesh = o3d.geometry.TriangleMesh(m)
        prep_mesh.translate(-c)
        prep_mesh.scale(1.0 / gscale, center=(0.0, 0.0, 0.0))
        cad_samples[part] = (tgt_pts, tgt_nrm)
        part_meshes[part] = prep_mesh
        part_prep[part] = (np.asarray(c, float), float(gscale))

    prepped = {}
    for f in scan_files:
        base = os.path.basename(f).replace("_Source.ply", "")
        npz = os.path.join(art, f"src_{base}.npz")
        if os.path.exists(npz):
            z = np.load(npz); src = z["pts"]
        else:
            pc = o3d.io.read_point_cloud(f).voxel_down_sample(CFG["scan_voxel"])
            src = np.asarray(pc.points).copy()
            np.savez_compressed(npz, pts=src)
        prepped[base] = src

    pairs_dir = os.path.join(HERE, "artifacts", "rpmnet_pairs_e4b")
    os.makedirs(pairs_dir, exist_ok=True)
    rpm_results, rpm_meta = {}, {}
    if not skip_rpmnet:
        for f in scan_files:
            base = os.path.basename(f).replace("_Source.ply", "")
            part = base.rsplit("_", 1)[0]
            np.savez(os.path.join(pairs_dir, f"{base}.npz"),
                     src=prepped[base], tgt=cad_samples[part][0])
        uniq = os.path.join(HERE, "results", f"e4_rpmnet.{uuid.uuid4().hex[:8]}.json")
        r = subprocess.run([GUJI_PY, RPMNET_ADAPTER, "--pairs", pairs_dir, "--out", uniq],
                           capture_output=True, text=True, cwd=CR, timeout=7200)
        if r.returncode != 0 or not os.path.exists(uniq):
            print(f"RPMNet adapter FAILED rc={r.returncode}: {r.stdout[-200:]} {r.stderr[-200:]}")
        else:
            rec = json.load(open(uniq))
            rpm_results = rec["results"]
            rpm_meta = {k: rec.get(k) for k in ("ckpt", "device", "note")}

    rows = []
    diag_m = {}
    for f in scan_files:
        base = os.path.basename(f).replace("_Source.ply", "")
        part, scan_nmb = base.rsplit("_", 1)
        feats, fstats = feats_from_records(fits[part][0], prep=part_prep[part]), fits[part][1]
        src = prepped[base]
        if len(src) < 500 or len(feats) < 4:
            rows.append(dict(scan=base, part=part, status="skipped",
                             reason=f"src={len(src)} feats={len(feats)}"))
            continue
        tgt_pts, tgt_nrm = cad_samples[part]
        cad_mesh = part_meshes[part]          # PREPARED mesh (scaled+centered) for p2m
        diag = float(np.linalg.norm(np.ptp(tgt_pts, axis=0)))
        diag_m.setdefault(part, diag)
        sols, gt_keys = load_gt(part, scan_nmb)   # already inverted to source->template
        p2m = e0.make_p2m_evaluator(cad_mesh)

        def eval_m(pts_m, T):
            """Cranfield units are METERS (verification-#2 fix): report p2m in mm
            and use a scale-aware inlier threshold (repo voxel 0.01 m)."""
            d = p2m(pts_m @ T[:3, :3].T + T[:3, 3]) * 1000.0
            return {"p2m_median_mm": float(np.median(d)),
                    "p2m_p95_mm": float(np.percentile(d, 95)),
                    "inlier_pct": float(100.0 * np.mean(d < 10.0))}   # 10 mm = 0.01 m

        # GT direction validation (convention check, NOT per-scan tuning): which
        # family, applied to the SOURCE points, aligns them onto the prepared
        # template: the declared GT (template->source, i.e. our eval solutions)
        # or its inverse. The winner's INVERSE is the source->template family;
        # evaluate_alignment then uses the winner directly (see load_gt note).
        def fam_dist(fam):
            ds = [float(np.median(p2m(src @ S[:3, :3].T + S[:3, 3]))) for S in fam]
            return min(ds)
        # apply to source the INVERSE of each declared GT (source->template candidates)
        inv_fam = [np.linalg.inv(S) for S in sols]
        d_fwd = fam_dist(inv_fam)                 # declared direction is right for src->template
        d_bwd = fam_dist(sols)                    # inverted interpretation
        gt_dir = "declared" if d_fwd <= d_bwd else "inverted"
        if gt_dir == "inverted":
            # declared family actually maps source->template; eval solutions are its inverse
            sols = inv_fam

        src_pc = o3d.geometry.PointCloud(); src_pc.points = o3d.utility.Vector3dVector(src)
        src_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=CFG["scan_voxel"] * 5, max_nn=30))
        tgt_pc = o3d.geometry.PointCloud()
        tgt_pc.points = o3d.utility.Vector3dVector(tgt_pts)
        tgt_pc.normals = o3d.utility.Vector3dVector(tgt_nrm)
        sn = estimate_normals(src, radius=CFG["scan_voxel"] * 2.5, max_nn=30)

        rec = dict(scan=base, part=part, n_pts=len(src), n_gt_solutions=len(sols),
                   gt_keys=gt_keys, gt_direction_validated=gt_dir,
                   gt_fwd_med_m=round(d_fwd, 6), gt_bwd_med_m=round(d_bwd, 6))

        # canonical persisted FGR per scan (verification-#2 prescription)
        fgr_path = os.path.join(art, f"canonical_fgr_{base}.npy")
        if os.path.exists(fgr_path):
            T_fgr = np.load(fgr_path); t_fgr = 0.0
        else:
            t0 = time.perf_counter(); T_fgr = fgr_arm(src_pc, tgt_pc); t_fgr = time.perf_counter() - t0
            np.save(fgr_path, T_fgr)
        rec["FGR"] = dict(**eval_m(src, T_fgr), runtime_s=t_fgr)
        t0 = time.perf_counter(); T_i = icp_refine(src_pc, tgt_pc, T_fgr); t_ref = time.perf_counter() - t0
        rec["FGR+ICP"] = dict(**eval_m(src, T_i), runtime_s=t_fgr + t_ref)
        rre_f, rte_f = gt_error(T_i, sols)
        rec["FGR+ICP"].update(rre_deg=float(rre_f), rte=float(rte_f),
                              success=bool(rre_f < 5.0 and rte_f < 0.02 * diag))

        # GeoP2E: E-opt selection on fitted feats, register from the CANONICAL FGR
        # start, production HV semantics (trigger reported per scan)
        rng0 = np.random.default_rng(e0.MASTER_SEED)
        # primitive-frame assertion (final-verification prescription): at the
        # canonical FGR start, count features with >=10 radius-gated matches;
        # <2 gated features means the frame transform failed -> hard error row
        p_chk = src @ T_fgr[:3, :3].T + T_fgr[:3, 3]
        gate_r = 0.18 * float(np.linalg.norm(np.ptp(tgt_pts, axis=0)))
        n_gated = sum(1 for f0 in feats
                      if int(np.sum(np.abs(f0.compute_residual(p_chk)) < gate_r)) >= 10)
        if n_gated < 2:
            rec["GeoP2E"] = {"error": f"primitive-frame assertion failed (n_gated={n_gated})"}
            rows.append(rec)
            print(f"{base}: FRAME ASSERTION FAIL n_gated={n_gated}", flush=True)
            continue
        ext = float(np.linalg.norm(np.ptp(tgt_pts, axis=0)))
        guide = [f.generate_points(120, noise_std=0.0, extent=ext * 0.15, rng=rng0) for f in feats]
        sel_idx = greedy_select(feats, guide, criterion="e", target_size=8, min_features=5, rng=rng0)
        sub = [feats[i] for i in sel_idx]
        sub_guide = [guide[i] for i in sel_idx]
        REG = auto_params(tgt_pts)
        reg_fn = make_reg_fn(REG)
        t0 = time.perf_counter()
        try:
            T_geo, info, hv_log = verify_shared_init(
                sub, sub_guide, tgt_pts, src, sn, reg_fn, init_T=T_fgr)
            t_geo = t_fgr + (time.perf_counter() - t0)
            rre_g, rte_g = gt_error(T_geo, sols)
            rec["GeoP2E"] = dict(**eval_m(src, T_geo), runtime_s=t_geo,
                                 rre_deg=float(rre_g), rte=float(rte_g),
                                 success=bool(rre_g < 5.0 and rte_g < 0.02 * diag),
                                 n_selected=len(sel_idx), **hv_log)
            # PREDECLARED secondary basin-diagnostic arm (not a rescue): common
            # ICP refinement from the GeoP2E endpoint
            t0 = time.perf_counter()
            T_gi = icp_refine(src_pc, tgt_pc, T_geo)
            rre_gi, rte_gi = gt_error(T_gi, sols)
            rec["GeoP2E+ICP(secondary)"] = dict(**eval_m(src, T_gi),
                                                runtime_s=t_geo + (time.perf_counter() - t0),
                                                rre_deg=float(rre_gi), rte=float(rte_gi),
                                                success=bool(rre_gi < 5.0 and rte_gi < 0.02 * diag))
        except Exception as ex:
            rec["GeoP2E"] = {"error": repr(ex)[:200]}
        rr = rpm_results.get(base, {})
        if "T" in rr:
            T_r = np.array(rr["T"])
            rre_r, rte_r = gt_error(T_r, sols)
            rec["RPMNet"] = dict(**eval_m(src, T_r),
                                 runtime_s_infer_only=rr.get("time_s", 0.0),
                                 rre_deg=float(rre_r), rte=float(rte_r),
                                 success=bool(rre_r < 5.0 and rte_r < 0.02 * diag))
        else:
            rec["RPMNet"] = {"error": rr.get("error", "missing")[:150]}
        rec["fit_stats"] = fstats
        rows.append(rec)
        g = rec.get("GeoP2E", {})
        print(f"{base}: gt_dir={gt_dir} FGR+ICP rre={rec['FGR+ICP']['rre_deg']:.1f} "
              f"GeoP2E rre={g.get('rre_deg', float('nan')):.1f} trig={g.get('triggered')} "
              f"nhyp={g.get('n_hypotheses')} p2m={g.get('p2m_median_mm', float('nan')):.2f}mm", flush=True)

    # aggregate (all scans incl. skips reported; success = joint RRE<5deg AND
    # RTE<2%·diagonal, PREDECLARED; mirror/near-180 outcomes explicit)
    agg = {}
    for row in rows:
        p = row.get("part")
        if not p or row.get("status") == "skipped":
            continue
        agg.setdefault(p, []).append(row)
    summary = {}
    for p, rs in agg.items():
        s = {"n_scans": len(rs), "n_skipped": sum(1 for r in rows if r.get("part") == p and r.get("status") == "skipped")}
        for m in ("FGR+ICP", "GeoP2E", "GeoP2E+ICP(secondary)", "RPMNet"):
            ok = [r[m] for r in rs if m in r and "rre_deg" in r[m]]
            if ok:
                s[m] = dict(n=len(ok),
                            success_joint=f"{sum(1 for x in ok if x.get('success'))}/{len(ok)}",
                            rre_median_deg=float(np.median([x["rre_deg"] for x in ok])),
                            rre_max_deg=float(np.max([x["rre_deg"] for x in ok])),
                            near180_count=sum(1 for x in ok if x["rre_deg"] > 170),
                            p2m_median_mm=float(np.median([x.get("p2m_median_mm", np.nan) for x in ok])))
        summary[p] = s

    out = {"protocol": {
            "scope": "prespecified applicability & failure map of production GeoP2E under "
                     "mesh-fitted primitives — NOT a superiority or native-B-Rep-generality claim",
            "units": "Cranfield metres; p2m reported in mm; inlier threshold 10mm (repo voxel 0.01m)",
            "gt": "benchmark ICP-refined estimates; direction VALIDATED per scan (declared vs inverse, recorded); "
                  "symmetry solution set used when provided",
            "success_rule": "PREDECLARED joint: RRE<5deg AND RTE<2%*CAD-diagonal",
            "secondary_arm": "GeoP2E+ICP predeclared as basin-diagnostic only, never a rescue",
            "config": CFG,
            "canonical_inputs": "per-part CAD sample + per-scan preprocessed source + per-scan FGR persisted under artifacts/cranfield_canon/",
            "disclosure": "evaluated under research use; no redistribution (repo has no LICENSE)"},
           "rpmnet_meta": rpm_meta,
           "summary_per_part": summary, "rows": rows}
    p = os.path.join(HERE, "results", "e4_cranfield.json")
    if os.path.exists(p):
        p += f".{uuid.uuid4().hex[:8]}.json"
    e0.save_json(out, p)
    print(json.dumps(summary, indent=1, default=float)[:2000])


if __name__ == "__main__":
    main(skip_rpmnet="--skip-rpmnet" in sys.argv, fit_only="--fit-only" in sys.argv)
