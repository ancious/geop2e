"""E2 — controlled Table-A5 rebuild (R2: "A controlled comparison is needed, or
these superiority claims should be removed").

ALL methods, including GeoP2E itself, run on IDENTICAL inputs: the persisted
20k-pt CAD target (artifacts/gear_target_20260827.npz), the same evaluation
mesh, the same per-scan shared FGR coarse start (voxel 1.0 frozen; ONE setting,
no aggregation), same source scans, same units. Global methods that generate
their own start (TEASER++, Super4PCS, RPMNet) are reported raw AND with the
common point-to-plane ICP refinement (threshold 2.0, max 100 iters) so global-
vs-refined stages are separable. RPMNet runs zero-shot via the guji-env adapter
(subprocess; partial_0.7 checkpoint; disclosed domain mismatch). NO 1.6-6x
ratio is computed anywhere. Per-scan rows only.

Usage: python e2_controlled_table.py [--skip-rpmnet]
"""
import os, sys, json, time, subprocess, tempfile, glob
import numpy as np
import open3d as o3d

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PARENT)

import e0_common as e0
from solver import estimate_normals
from selection import greedy_select, spectral, selected_info
from baselines import trimmed_icp
from gear_baseline_extended import sparse_icp

GUJI_PY = os.path.expanduser("~/miniconda3/envs/guji/bin/python")
RPMNET_ADAPTER = os.path.join(HERE, "e1c_rpmnet.py")
CRANFIELD_ROOT = os.path.expanduser("~/projects/new/3dphy/data/real/cranfield")
S4PCS_BIN = os.path.expanduser("~/tmp/builds/opengr/build/apps/Super4PCS/Super4PCS")
S4_DELTA = 1.0          # frozen (mm); smoke-validated on this CAD
TEASER_NOISE_BOUND = 1.0  # frozen (mm) = FPFH voxel scale; library defaults else


def write_ply_f32(path, pts):
    pts = np.asarray(pts, np.float32)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(pts))
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in pts:
            f.write("%.4f %.4f %.4f\n" % (p[0], p[1], p[2]))


def icp_refine(src_pc, tgt_pc, T_init):
    est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    crit = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=e0.CONFIGS["icp_max_iter"])
    return o3d.pipelines.registration.registration_icp(
        src_pc, tgt_pc, e0.CONFIGS["icp_threshold"], T_init, est, crit).transformation


def teaser_arm(src, tgt_pts, tgt_nrm, corr_src, corr_tgt):
    import teaserpp_python as tp
    params = tp.RobustRegistrationSolver.Params()
    params.noise_bound = TEASER_NOISE_BOUND
    params.estimate_scaling = False   # rigid SE(3) only (review fix: default True invalidates R,t)
    solver = tp.RobustRegistrationSolver(params)
    solver.solve(np.ascontiguousarray(corr_src.T), np.ascontiguousarray(corr_tgt.T))
    s = solver.getSolution()
    T = np.eye(4)
    T[:3, :3] = np.asarray(s.rotation)
    T[:3, 3] = np.asarray(s.translation).ravel()
    return T


def fpfh_correspondences(src_pc, tgt_pc, voxel):
    """RECIPROCAL (mutual) FPFH correspondences (review fix: previous code
    documented mutual but returned forward 1-NN with dead reverse query):
    keep src->tgt matches whose target's nearest source is the same point;
    then cap to 8000 by seeded subsample (indices applied to BOTH arrays,
    preserving pairing)."""
    sd = src_pc.voxel_down_sample(voxel)
    td = tgt_pc.voxel_down_sample(voxel)
    for p in (sd, td):
        p.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    fs = o3d.pipelines.registration.compute_fpfh_feature(sd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
    ft = o3d.pipelines.registration.compute_fpfh_feature(td, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
    from scipy.spatial import cKDTree
    S = np.asarray(fs.data).T; Td = np.asarray(ft.data).T
    ts = cKDTree(Td); tt = cKDTree(S)
    _, i1 = ts.query(S, k=1)           # src -> tgt
    _, i2 = tt.query(Td, k=1)          # tgt -> src
    rec = i2[i1] == np.arange(len(i1))  # reciprocal filter
    src_idx = np.where(rec)[0]
    if len(src_idx) > 8000:
        src_idx = src_idx[np.random.default_rng(0).choice(len(src_idx), 8000, replace=False)]
    src_c = np.asarray(sd.points)[src_idx]
    tgt_c = np.asarray(td.points)[i1[src_idx]]
    return src_c, tgt_c


def s4pcs_arm(src_pts, tgt_pts, workdir, label):
    sp = os.path.join(workdir, f"{label}_src.ply")
    tp = os.path.join(workdir, f"{label}_tgt.ply")
    write_ply_f32(sp, src_pts[::max(1, len(src_pts) // 40000)])
    write_ply_f32(tp, tgt_pts)
    mat = os.path.join(workdir, f"{label}_mat.txt")
    cmd = [S4PCS_BIN, "-i", tp, sp, "-d", str(S4_DELTA), "-m", mat, "-t", "60"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if not os.path.exists(mat):
        raise RuntimeError(f"Super4PCS no matrix: {r.stdout[-200:]} {r.stderr[-200:]}")
    M = np.loadtxt(mat, skiprows=2)
    return M


def rpmnet_arm(pairs_dir, out_json):
    """Unique output name per invocation (review fix: the adapter's
    never-overwrite suffix made the old code silently read a STALE base file);
    check subprocess return code; return full adapter record (ckpt/seed/device)."""
    import uuid
    unique = out_json[:-5] + f".{uuid.uuid4().hex[:8]}.json"
    cmd = [GUJI_PY, RPMNET_ADAPTER, "--pairs", pairs_dir, "--out", unique]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=CRANFIELD_ROOT, timeout=3600)
    if r.returncode != 0 or not os.path.exists(unique):
        raise RuntimeError(f"RPMNet adapter failed rc={r.returncode}: "
                           f"{r.stdout[-300:]} {r.stderr[-300:]}")
    return json.load(open(unique))


def main(skip_rpmnet=False):
    mesh, tgt_pts, tgt_nrm, feats = e0.load_gear_cad()
    p2m = e0.make_p2m_evaluator(mesh)
    tgt_pc = o3d.geometry.PointCloud()
    tgt_pc.points = o3d.utility.Vector3dVector(tgt_pts)
    tgt_pc.normals = o3d.utility.Vector3dVector(tgt_nrm)

    rng0 = np.random.default_rng(e0.MASTER_SEED)
    guide = [f.generate_points(120, noise_std=0.0, extent=6.0, rng=rng0) for f in feats]
    sel_idx = greedy_select(feats, guide, **e0.CONFIGS["geop2e_select"], rng=rng0)
    sub = [feats[i] for i in sel_idx]
    from e3_hv_ablation import auto_params, make_reg_fn
    REG = auto_params(tgt_pts)
    reg_fn = make_reg_fn(REG)

    # GeoP2E production on real scans: trigger will not fire (E3 finding) ->
    # single full-CAD solve from the SHARED FGR start (documented).
    pairs_dir = os.path.join(HERE, "artifacts", "rpmnet_pairs_e2")
    os.makedirs(pairs_dir, exist_ok=True)
    rpm_results, rpm_meta = {}, {}
    if not skip_rpmnet:
        rpm_out = os.path.join(HERE, "results", "e2_rpmnet.json")
        for sf, label in e0.SCANS:
            pc = e0.load_scan(os.path.join(e0.GEAR, sf))
            np.savez(os.path.join(pairs_dir, f"{label}.npz"),
                     src=np.asarray(pc.points), tgt=tgt_pts)
        rec = rpmnet_arm(pairs_dir, rpm_out)
        rpm_results = rec["results"]
        rpm_meta = {k: rec.get(k) for k in ("ckpt", "n_pts", "seed", "device", "note")}

    rows = []
    for sf, label in e0.SCANS:
        pc = e0.load_scan(os.path.join(e0.GEAR, sf))
        src = np.asarray(pc.points)
        src_pc = o3d.geometry.PointCloud(); src_pc.points = o3d.utility.Vector3dVector(src)
        src_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))
        sn = estimate_normals(src, radius=2.0, max_nn=30)
        fgr_path = os.path.join(HERE, "artifacts",
                                f"canonical_fgr_v{e0.CONFIGS['fgr_voxel']}_{label}.npy")
        # canonical-start closure (verification #2 prescription): ONE persisted FGR
        # matrix per scan, computed once EVER, then reused verbatim by every run —
        # FGR is nondeterministic across invocations, so recompute-and-warn cannot
        # support a 'frozen start' claim. The matrix + its sha are recorded below.
        import hashlib as _hl
        if os.path.exists(fgr_path):
            T_fgr = np.load(fgr_path)
            t_fgr = None   # canonical matrix loaded; original cost not re-measurable
        else:
            t0 = time.perf_counter()
            T_fgr = e0.fgr_coarse(pc, tgt_pts, tgt_nrm)
            t_fgr = time.perf_counter() - t0
            np.save(fgr_path, T_fgr)
        fgr_sha = _hl.sha256(T_fgr.tobytes()).hexdigest()[:16]

        res = {}
        init_cost = t_fgr if t_fgr is not None else 0.0
        def put(name, T, t, init_cost=0.0):
            res[name] = dict(**e0.eval_pose(src, T, p2m), runtime_s=t + init_cost)

        put("FGR", T_fgr, 0.0, init_cost=init_cost)
        t0 = time.perf_counter(); T = icp_refine(src_pc, tgt_pc, T_fgr); put("FGR+ICP_p2plane", T, time.perf_counter() - t0, init_cost)
        est_p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        crit = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=e0.CONFIGS["icp_max_iter"])
        t0 = time.perf_counter()
        T = o3d.pipelines.registration.registration_icp(src_pc, tgt_pc, e0.CONFIGS["icp_threshold"], T_fgr, est_p2p, crit).transformation
        put("FGR+ICP_p2p", T, time.perf_counter() - t0, init_cost)
        t0 = time.perf_counter()
        T = o3d.pipelines.registration.registration_generalized_icp(
            src_pc, tgt_pc, e0.CONFIGS["icp_threshold"], init=T_fgr,
            estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            criteria=crit).transformation
        put("FGR+GICP", T, time.perf_counter() - t0, init_cost)
        t0 = time.perf_counter()
        T = trimmed_icp(src, tgt_pts, init=T_fgr, **e0.CONFIGS["trimmed"])[0]
        put("FGR+TrimmedICP", T, time.perf_counter() - t0, init_cost)
        t0 = time.perf_counter()
        T = sparse_icp(src, tgt_pts, init=T_fgr, **e0.CONFIGS["sparseicp"])
        put("FGR+SparseICP", T, time.perf_counter() - t0, init_cost)

        with tempfile.TemporaryDirectory() as td:
            t0 = time.perf_counter()
            try:
                cs, ct = fpfh_correspondences(src_pc, tgt_pc, e0.CONFIGS["fgr_voxel"])
                T = teaser_arm(src, tgt_pts, tgt_nrm, cs, ct)
                t_glob = time.perf_counter() - t0
                put("TEASER++", T, t_glob)
                t0 = time.perf_counter(); T2 = icp_refine(src_pc, tgt_pc, T)
                put("TEASER+++ICP_p2plane", T2, t_glob + time.perf_counter() - t0)
            except Exception as ex:
                res["TEASER++"] = {"error": repr(ex)[:200]}
                res["TEASER+++ICP_p2plane"] = {"error": "skipped (global failed)"}
            t0 = time.perf_counter()
            try:
                T = s4pcs_arm(src, tgt_pts, td, label)
                t_glob = time.perf_counter() - t0
                put("Super4PCS", T, t_glob)
                t0 = time.perf_counter(); T2 = icp_refine(src_pc, tgt_pc, T)
                put("Super4PCS+ICP_p2plane", T2, t_glob + time.perf_counter() - t0)
            except Exception as ex:
                res["Super4PCS"] = {"error": repr(ex)[:200]}
                res["Super4PCS+ICP_p2plane"] = {"error": "skipped (global failed)"}

        if not skip_rpmnet:
            rr = rpm_results.get(label, {})
            if "T" in rr:
                # zero-shot: inference time only (preprocessing/model-load excluded,
                # disclosed); runtime NOT comparable to CPU end-to-end arms
                res["RPMNet_zero_shot"] = dict(
                    **e0.eval_pose(src, np.array(rr["T"]), p2m),
                    runtime_s_infer_only=rr.get("time_s", 0.0), device=rpm_meta.get("device"))
            else:
                res["RPMNet_zero_shot"] = {"error": rr.get("error", "missing")[:200]}

        t0 = time.perf_counter()
        try:
            T, info = reg_fn(src, sn, sub, T_fgr)
            put("GeoP2E(FGR-init,production)", T, time.perf_counter() - t0, init_cost)
            res["GeoP2E(FGR-init,production)"]["iters"] = info.get("iters")
        except Exception as ex:
            res["GeoP2E(FGR-init,production)"] = {"error": repr(ex)[:200]}

        rows.append(dict(scan=sf, label=label, n_pts=len(src), results=res,
                         canonical_fgr_sha16=fgr_sha,
                         canonical_fgr_persisted=(t_fgr is None),
                         canonical_fgr_T=T_fgr.tolist()))

        print(f"\n== {label} ({len(src)} pts) ==", flush=True)
        for k, v in res.items():
            if "p2m_median_mm" in v:
                t = v.get("runtime_s", v.get("runtime_s_infer_only"))
                tdesc = "s" if "runtime_s" in v else "s(infer-only)"
                print(f"  {k:26s} p2m_med={v['p2m_median_mm']:.3f} p95={v['p2m_p95_mm']:.3f} "
                      f"inlier={v['inlier_pct']:.1f}% t={t:.2f}{tdesc}")
            else:
                print(f"  {k:26s} ERROR {v.get('error','')[:80]}")

    import hashlib, subprocess as sp
    def _sha(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16] if os.path.exists(p) else None
    try:
        commit = sp.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PARENT,
                                 stderr=sp.DEVNULL).decode().strip()
    except Exception:
        commit = None
    env_rec = {"open3d": o3d.__version__, "numpy": np.__version__,
               "scipy": __import__("scipy").__version__, "git_commit(parent, geop2e untracked)": commit,
               "source_sha256_e2": _sha(__file__), "source_sha256_e1c": _sha(RPMNET_ADAPTER),
               "target_npz_sha256": _sha(os.path.join(HERE, "artifacts", "gear_target_20260827.npz")),
               "opengr_binary_sha256": _sha(S4PCS_BIN),
               "rpmnet_ckpt": rpm_meta, "selected_feature_indices": list(map(int, sel_idx)),
               "configs": e0.CONFIGS, "geop2e_solver_params": REG,
               "timing_scope": ("FGR cost recomputed per run and added to every arm that "
                                "consumes the shared FGR start (incl. GeoP2E); TEASER/S4PCS "
                                "refined rows = global + refine; RPMNet = model inference "
                                "only, preprocessing/model-load excluded, separate device")}

    out = {"protocol": {
            "inputs": "identical scan files, persisted 20k-pt CAD target, same evaluation mesh, mm units",
            "arms_common_FGR_init": ["FGR", "FGR+ICP_p2p", "FGR+ICP_p2plane", "FGR+GICP",
                                     "FGR+TrimmedICP", "FGR+SparseICP", "GeoP2E(FGR-init,production)"],
            "arms_own_global_start": ["TEASER++(+ICP_p2plane)", "Super4PCS(+ICP_p2plane)", "RPMNet_zero_shot"],
            "modality_disclosure": ("GeoP2E consumes 8 analytic B-Rep primitives (FIM-selected); "
                                    "point-cloud methods consume the 20k sampled CAD target; "
                                    "TEASER++ uses reciprocal FPFH correspondences (voxel 1.0, cap 8000); "
                                    "Super4PCS consumes a deterministic raw-scan decimation (~44k pts) "
                                    "vs 20k target; RPMNet uses deterministic 1024-pt subsets with "
                                    "own-centroid + target-extent normalisation (disclosed domain mismatch). "
                                    "Evaluation for every arm: scan->CAD-mesh distance (geometric fit, "
                                    "NOT independent pose accuracy)."),
            "teaser": {"noise_bound_mm": TEASER_NOISE_BOUND, "estimate_scaling": False,
                       "other": "library defaults"},
            "s4pcs": {"delta_mm": S4_DELTA, "time_cap_s": 60},
            "geop2e": "FIM-selected features (E-opt, 8), production solver params, shared FGR start; "
                      "volume trigger does not activate on these scans (E3 finding) so single full-CAD solve",
            "note": "per-scan descriptive reporting only; no pooled superiority ratio is computed (1.6-6x removed)"},
           "env": env_rec,
           "scans": rows}
    p = os.path.join(HERE, "results", "e2_controlled_table.json")
    if os.path.exists(p):
        import uuid
        p += f".{uuid.uuid4().hex[:8]}.json"
    e0.save_json(out, p)


if __name__ == "__main__":
    main(skip_rpmnet="--skip-rpmnet" in sys.argv)
