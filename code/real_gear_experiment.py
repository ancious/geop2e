"""Real-gear defect experiment (R3-5 + R3-6 + metrology) — runs in the `geo` env
(py3.11 + open3d + trimesh). Uses REAL gear CAD + features, simulates a known SE(3)
transform + a large defect (region removal), and compares GeoP2E (with/without
hypothesis verification) against open3d ICP / GICP / FGR and Trimmed ICP.

Independent GT = the known applied transform (no CMM). Units = mm (real gear).
"""
import sys, json, time, os
import numpy as np
import trimesh, open3d as o3d
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import load_features_from_json
from solver import estimate_normals, register, evaluate_alignment, transform_points
from selection import greedy_select, primitive_info, selected_info, spectral
from baselines import icp_point_to_point, icp_point_to_plane, trimmed_icp
import hypothesis as hyp

GEAR = "data/gear_step_cad"
RNG_SEED = 7
N_TARGET = 8000


def load_target_and_features(rng):
    mesh = o3d.io.read_triangle_mesh(f"{GEAR}/gear/model.obj")
    mesh.compute_vertex_normals()
    tgt_pcd = mesh.sample_points_uniformly(number_of_points=N_TARGET)
    tgt_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))
    feats = load_features_from_json(f"{GEAR}/gear/extracted_features.json")
    return np.asarray(tgt_pcd.points), np.asarray(tgt_pcd.normals), feats


def build_guide_sets(feats, rng, n=120, extent=6.0):
    return [f.generate_points(n, noise_std=0.0, extent=extent, rng=rng) for f in feats]


def auto_params(target_pts):
    ext = float(np.linalg.norm(np.ptp(target_pts, axis=0)))
    return dict(init_radius=ext * 0.18, anneal=0.9, max_iter=60, lam=1e-3,
                k_huber=ext * 0.012, angle_thresh_deg=30.0)


def make_register(REG):
    # FIX 2026-08-27: hypothesis.verify() calls register_fn(pts, normals, feats, init_T)
    # (4 args); the previous 3-arg closure raised TypeError inside the experiment's
    # try/except, which is why the n=3 HV ablation silently returned identical
    # 999-error rows for all three GeoP2E arms (root cause of the unusable run).
    def reg_fn(points, normals, feats, init_T=None):
        return register(points, normals, feats, init_T=init_T, **REG)
    return reg_fn


def simulate_defect(target_pts, rng, keep_frac=0.6, rot_deg=10.0, trans_frac=0.25):
    # remove a contiguous slab (random axis) to simulate a large defect
    pts = target_pts.copy()
    dim = rng.integers(0, 3)
    lo, hi = pts[:, dim].min(), pts[:, dim].max()
    cut = lo + (1 - keep_frac) * (hi - lo) * rng.uniform(0.2, 0.8)
    # remove points on one side of the cut (a "broken-off" region)
    keep = pts[:, dim] > cut if rng.random() < 0.5 else pts[:, dim] < (hi - (cut - lo))
    src = pts[keep]
    ext = np.linalg.norm(np.ptp(target_pts, axis=0))
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    T = np.eye(4)
    T[:3, :3] = Rot.from_rotvec(axis * np.deg2rad(rng.uniform(-rot_deg, rot_deg))).as_matrix()
    T[:3, 3] = rng.uniform(-1, 1, 3) * ext * trans_frac
    return transform_points(src, T), T, float(keep.mean())


def run_open3d_icp(src, tgt, tn, method, threshold):
    sp = o3d.geometry.PointCloud(); sp.points = o3d.utility.Vector3dVector(src)
    tp = o3d.geometry.PointCloud(); tp.points = o3d.utility.Vector3dVector(tgt); tp.normals = o3d.utility.Vector3dVector(tn)
    crit = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=100)
    if method == "p2p":
        est = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        r = o3d.pipelines.registration.registration_icp(sp, tp, threshold, np.eye(4), est, crit)
    elif method == "p2plane":
        est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        r = o3d.pipelines.registration.registration_icp(sp, tp, threshold, np.eye(4), est, crit)
    elif method == "gicp":
        # FIX 2026-08-27 (gpt-review): the old positional call passed (threshold, est, I, crit)
        # — est landed in the `init` slot and I in `estimation`, raising TypeError
        # (swallowed by except -> the historical GICP=999 rows). Keyword form:
        est = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
        r = o3d.pipelines.registration.registration_generalized_icp(
            sp, tp, threshold, init=np.eye(4), estimation_method=est, criteria=crit)
    elif method == "fgr":
        # FIX 2026-08-27: registration_fgr_based_registration does not exist in
        # open3d (the old call always raised). Use FPFH + feature-matching FGR.
        fsd = sp.voxel_down_sample(1.0); ftd = tp.voxel_down_sample(1.0)
        fsd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))
        ftd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))
        fs = o3d.pipelines.registration.compute_fpfh_feature(fsd, o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=100))
        ft = o3d.pipelines.registration.compute_fpfh_feature(ftd, o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=100))
        opt = o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=threshold, use_absolute_scale=True)
        r = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(fsd, ftd, fs, ft, opt)
    return r.transformation


def main(n_trials=6, keep_frac=0.6):
    rng = np.random.default_rng(RNG_SEED)
    tgt_pts, tgt_normals, feats = load_target_and_features(rng)
    print(f"target {tgt_pts.shape}, features {len(feats)} ({[f.kind for f in feats[:3]]}...)")
    guide = build_guide_sets(feats, rng)
    sel_idx = greedy_select(feats, guide, criterion="e", target_size=8, min_features=5, rng=rng)
    sub = [feats[i] for i in sel_idx]
    sp_sel = spectral(selected_info(feats, guide, sel_idx))
    print(f"FIM selected {len(sel_idx)} feats: lambda_min={sp_sel['lambda_min']:.2f} kappa={sp_sel['kappa']:.2f}")
    REG = auto_params(tgt_pts)
    reg_fn = make_register(REG)
    threshold = REG["init_radius"]

    methods = ["GeoP2E_no_hyp", "GeoP2E_full_hyp", "GeoP2E_cut_off", "ICP_p2p", "ICP_p2plane", "GICP", "FGR", "Trimmed"]
    results = {m: [] for m in methods}
    for s in range(n_trials):
        r = np.random.default_rng(RNG_SEED + 50 + s)
        src, T_gt, kept = simulate_defect(tgt_pts, r, keep_frac=keep_frac)
        sn = estimate_normals(src, radius=REG["init_radius"] * 0.4, max_nn=30)
        for m in methods:
            try:
                if m == "GeoP2E_no_hyp":
                    T, _ = reg_fn(src, sn, sub)
                elif m == "GeoP2E_full_hyp":
                    T, _, _ = hyp.verify(sub, [guide[i] for i in sel_idx], tgt_pts, src, sn, reg_fn, 1 - keep_frac,
                                         volume_trigger=True, cutting=True, penetration=True)
                elif m == "GeoP2E_cut_off":
                    T, _, _ = hyp.verify(sub, [guide[i] for i in sel_idx], tgt_pts, src, sn, reg_fn, 1 - keep_frac,
                                         volume_trigger=True, cutting=False, penetration=True)
                elif m == "Trimmed":
                    T, _ = trimmed_icp(src, tgt_pts, threshold=threshold, trim_frac=1 - keep_frac + 0.1, max_iter=50)
                else:
                    T = run_open3d_icp(src, tgt_pts, tgt_normals, m.split("_", 1)[1], threshold)
                re, te = evaluate_alignment(T, T_gt)
            except Exception as e:
                re, te = 999.0, 999.0
                print(f"  [{m}] ERROR: {e}")
            results[m].append((re, te))
        print(f"trial {s}: kept={kept:.2f} " + " ".join(f"{m}={results[m][-1][0]:.2f}°/{results[m][-1][1]:.2f}mm" for m in methods))

    summary = {}
    for m, res in results.items():
        res = np.array(res)
        succ = (res[:, 0] < 1.0) & (res[:, 1] < 1.0)   # 1° & 1mm (mm units)
        summary[m] = {"rot_median": float(np.median(res[:, 0])), "rot_p95": float(np.percentile(res[:, 0], 95)),
                      "trans_median": float(np.median(res[:, 1])), "success": f"{int(succ.sum())}/{n_trials}"}
    out = {"config": {"keep_frac": keep_frac, "n_trials": n_trials, "threshold_mm": threshold,
                      "defect": f"~{(1-keep_frac)*100:.0f}% region removed", "units": "mm (real gear)",
                      "FIM_select": {"n": len(sel_idx), "lambda_min": sp_sel["lambda_min"], "kappa": sp_sel["kappa"]}},
           "results": summary}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/real_gear.json", "w"), indent=2)
    print("\n=== Real-gear defect experiment (known-transform GT, mm) ===")
    print(f"{'method':22s} {'rot_med°':>9s} {'rot_p95°':>9s} {'trans_med_mm':>13s} {'success':>8s}")
    for m in methods:
        s = summary[m]
        print(f"{m:22s} {s['rot_median']:9.3f} {s['rot_p95']:9.3f} {s['trans_median']:13.3f} {s['success']:>8s}")


if __name__ == "__main__":
    main()
