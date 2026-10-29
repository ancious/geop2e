"""e6_gear_retro — RETROSPECTIVE repair cases (S1-S3) with GeoP2E-auto.

Truth discipline (GPT ruling): S1-S3 influenced mechanism invention -> these are
retrospective demonstrations, NOT confirmatory tests. Production numbers stay in
the record; auto numbers are reported alongside as failure-informed repair.

Builds a parsed-style part dict for the GEAR from its CAD-parsed features
(extracted_features.json, 207 primitives) with supports assigned by nearest-
primitive residual on the persisted 20k CAD sample (unassigned/tooth points get
large trimmed distances -> mixture outliers, by design).
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import e0_common as e0
import e6_auto as A
from solver import estimate_normals

OUT_PREFIX = "e6_gear_retro"


def gear_part(feats, tgt_pts, assign_gate_frac=0.02):
    """Merge coincident parsed features; assign support clouds from the CAD sample."""
    D = float(np.linalg.norm(np.ptp(tgt_pts, axis=0)))
    tol = 1e-3 * D
    merged = []
    for f in feats:
        tgt = None
        for m in merged:
            if A._same_prim(m["feats"][0], f, tol):
                tgt = m; break
        if tgt is None:
            merged.append(dict(feats=[f]))
        else:
            tgt["feats"].append(f)
    # assign each CAD sample point to the primitive with the smallest |residual|
    R = np.stack([np.abs(f.compute_residual(tgt_pts)) for f in
                  [m["feats"][0] for m in merged]], axis=1)
    lab = R.argmin(axis=1)
    gate = assign_gate_frac * D
    ok = R.min(axis=1) < gate
    for j, m in enumerate(merged):
        m["pts"] = tgt_pts[ok & (lab == j)]
    merged = [m for m in merged if len(m["pts"]) >= 8]
    for m in merged:
        m["feat"] = m["feats"][0]; m["area"] = float(len(m["pts"]))
    all_area = len(tgt_pts)
    return dict(name="gear", D=D, supports=merged,
                analytic_coverage=float(ok.mean()), n_faces_total=len(feats))


def main():
    mesh, tgt_pts, tgt_nrm, feats = e0.load_gear_cad()
    part = gear_part(feats, tgt_pts)
    kinds = {}
    for m in part["supports"]:
        kinds[m["feat"].kind] = kinds.get(m["feat"].kind, 0) + 1
    print(f"gear part: {len(part['supports'])} merged supports {kinds} "
          f"analytic-assigned coverage={part['analytic_coverage']:.3f}")
    tcad = A.TrimmedCAD(part)
    p2m = e0.make_p2m_evaluator(mesh)
    k_H = 0.012 * part["D"]

    prod = {"S1_defect_30pct": 2.340, "S2_defective": 0.425, "S3_fragment": 1.800}
    icp = {"S1_defect_30pct": 0.377, "S2_defective": 0.377, "S3_fragment": 1.310}
    out_rows = []
    for sf, label in e0.SCANS:
        pc = e0.load_scan(os.path.join(e0.GEAR, sf))
        src = np.asarray(pc.points)
        sn = estimate_normals(src, radius=2.0, max_nn=30)
        fgr_path = os.path.join(HERE, "artifacts",
                                f"canonical_fgr_v{e0.CONFIGS['fgr_voxel']}_{label}.npy")
        T_fgr = np.load(fgr_path)
        row = dict(scan=label, n_pts=len(src))
        # v src -> 20k-subsample for solver speed (auto is heavier)
        if len(src) > 40000:
            idx = np.random.default_rng(0).choice(len(src), 40000, replace=False)
            src_s, sn_s = src[idx], sn[idx]
        else:
            src_s, sn_s = src, sn
        t0 = time.time()
        T_a, info_a = A.register_auto(tcad, src_s, sn_s, k_H, T_fgr.copy())
        t_a = time.perf_counter() - t0
        row["auto_A1"] = {**e0.eval_pose(src, T_a, p2m), "runtime_s": t_a,
                          "accepted": info_a["accepted"]}
        t0 = time.perf_counter()
        T_full, info_m = A.multistart_auto(tcad, src_s, sn_s, k_H, T_fgr.copy(), n_starts=5)
        T_hv, info_h = A.staged_hv_auto(tcad, src_s, sn_s, k_H, T_full)
        t_full = time.perf_counter() - t0
        row["auto_full(A4+HV)"] = {**e0.eval_pose(src, T_hv, p2m), "runtime_s": t_full,
                                   "multistart": info_m["chosen"], "hv": info_h["winner"]}
        out_rows.append(row)
        print(f"\n== {label} (production={prod[label]}, best-ICP={icp[label]}) ==")
        for k in ("auto_A1", "auto_full(A4+HV)"):
            v = row[k]
            print(f"  {k:18s} p2m_med={v['p2m_median_mm']:.3f} p95={v['p2m_p95_mm']:.3f} "
                  f"inlier={v['inlier_pct']:.1f}% t={v['runtime_s']:.1f}s")

    p = os.path.join(HERE, "results", f"{OUT_PREFIX}.json")
    if os.path.exists(p):
        p += f".{int(time.time())}.json"
    e0.save_json(dict(note="RETROSPECTIVE repair cases (mechanisms invented after seeing "
                           "these failures) - not confirmatory",
                      rows=out_rows), p)


if __name__ == "__main__":
    main()
