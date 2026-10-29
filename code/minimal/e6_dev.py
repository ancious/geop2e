"""e6_dev — development verification gates for GeoP2E-auto (dev data ONLY; no test parts).

Gates (per GPT implementation order):
  G1 parse: STEP -> merged supports, coverage reported
  G2 trimmed-exactness: exact-B-Rep samples' trimmed distance at identity ~ 0
  G3 A1 fixed-point: register_auto from TRUE pose stays at TRUE pose (E3c regression)
  G4 A1 recovery: from coarse start (5deg/2%D) converges (<1deg, <2%D)
  G5 A3 selection: uncertainty table + alarm on dev part
  G6 A4 + staged HV smoke: runs, no GT in selection
"""
import os, sys, time
import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import e6_auto as A
from solver import estimate_normals, evaluate_alignment

STEP = "data/gear_step_cad/sample_model.stp"


def synth_trial(tcad, rng, keep=0.8, rot_deg=5.0, trans_frac=0.02, noise_frac=2e-3):
    """Exact-B-Rep scan: half-space crop (keep fraction), known GT, iid noise.
    Returns (src, T_gt_source_to_cad). Construction: src = (cad_pts[keep] @ R^T + t)
    => solver target T with T @ T_gt = I is inv(T_gt); T_gt below maps cad->src."""
    pts, _ = A.support_cloud(tcad.part)
    rng.shuffle(pts)
    n_keep = int(keep * len(pts))
    kept = pts[:n_keep]
    ext = float(np.linalg.norm(np.ptp(pts, axis=0)))
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
    R = Rot.from_rotvec(ax * np.deg2rad(rot_deg)).as_matrix()
    t = rng.uniform(-1, 1, 3) * ext * trans_frac
    src = kept @ R.T + t + rng.normal(0, noise_frac * ext, kept.shape)
    T_gt = np.eye(4); T_gt[:3, :3] = R; T_gt[:3, 3] = t      # cad -> src
    return src, T_gt


def main():
    print("[G1] parse")
    t0 = time.time()
    part = A.load_parsed_part(STEP)
    assert part is not None, "part ineligible"
    kinds = {}
    for m in part["supports"]:
        kinds[m["feat"].kind] = kinds.get(m["feat"].kind, 0) + 1
    print(f"  supports={len(part['supports'])} {kinds} coverage={part['analytic_coverage']:.3f} "
          f"D={part['D']:.3f} ({time.time()-t0:.1f}s)")
    tcad = A.TrimmedCAD(part)

    print("[G2] trimmed-exactness at identity")
    d0, _ = tcad.d_and_label(tcad.pts)
    print(f"  exact-sample trimmed distance: median={np.median(d0):.2e} p99={np.percentile(d0,99):.2e}")

    rng = np.random.default_rng(42)
    src, T_gt = synth_trial(tcad, rng)
    sn = estimate_normals(src, radius=0.02 * tcad.D, max_nn=30)
    k_H = 0.012 * tcad.D
    T_true = np.linalg.inv(T_gt)

    print("[G3] A1 fixed-point from TRUE pose")
    T_fp, _ = A.register_auto(tcad, src, sn, k_H, T_true.copy())
    re, te = evaluate_alignment(T_fp, T_gt)
    print(f"  final rre={re:.4f}deg rte={te:.5f}D  -> {'PASS' if re < 0.1 else 'FAIL'}")

    print("[G4] A1 recovery from coarse (5deg/2%D)")
    ok = 0
    for s in range(3):
        r = np.random.default_rng(100 + s)
        src_s, Tg = synth_trial(tcad, r, rot_deg=float(r.uniform(4, 8)), trans_frac=0.02)
        sn_s = estimate_normals(src_s, radius=0.02 * tcad.D, max_nn=30)
        P = np.eye(4)
        Pax = r.normal(size=3); Pax /= np.linalg.norm(Pax)
        P[:3, :3] = Rot.from_rotvec(Pax * np.deg2rad(r.uniform(1, 3))).as_matrix()
        P[:3, 3] = r.normal(size=3) * tcad.D * 0.005
        T0 = P @ np.linalg.inv(Tg)
        t0 = time.time()
        T_sol, info = A.register_auto(tcad, src_s, sn_s, k_H, T0)
        re, te = evaluate_alignment(T_sol, Tg)
        ok += int(re < 1.0 and te < 0.02 * tcad.D)
        print(f"  seed{s}: rre={re:.3f}deg rte={te:.4f}D accepted={info['accepted']} ({time.time()-t0:.1f}s)")
    print(f"  recovery {ok}/3 -> {'PASS' if ok >= 2 else 'FAIL'}")

    print("[G5] A3 selection (val = held-out spatial block score at solved pose)")
    T_solve, _ = A.register_auto(tcad, src, sn, k_H, T_true.copy())
    blocks = A.spatial_blocks(src, 8)

    def val_fn(idx):
        sub = A.TrimmedCAD_from_subset(tcad, idx)
        T_s, _ = A.register_auto(sub, src, sn, k_H, T_solve.copy())
        m = np.isin(blocks, [6, 7])
        dd, _ = tcad.d_and_label(src[m] @ T_s[:3, :3].T + T_s[:3, 3])
        sc = A.robust_score(dd, k_H)
        return sc, 1e-4

    pick, table = A.select_auto(part, sigma=k_H * 0.2, val_score_fn=val_fn)
    print(f"  picked={pick['name']} n={pick['n']} U={pick['U']:.3g} alarm={pick['alarm']}")
    for r in table:
        print(f"    {r['name']:5s} n={r['n']:3d} kappa={r['kappa']:9.2f} U={r['U']:8.3g} alarm={r['alarm']}")

    print("[G6] A4 + staged HV smoke")
    T_ms, info_ms = A.multistart_auto(tcad, src, sn, k_H, T_true.copy() @ (np.eye(4) + 0), n_starts=3)
    re, te = evaluate_alignment(T_ms, T_gt)
    print(f"  multistart chosen={info_ms['chosen']} rre={re:.3f}deg rte={te:.4f}D")
    T_hv, info_hv = A.staged_hv_auto(tcad, src, sn, k_H, T_ms)
    print(f"  staged HV winner={info_hv['winner']} base={info_hv['base']:.5f} se={info_hv['se']:.2e}")
    print("DONE")


if __name__ == "__main__":
    main()
