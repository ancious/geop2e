"""E3b — forced-trigger HV on the three REAL defect scans (GPT-sol-max prescription).

Purpose: the production AABB volume-ratio trigger activated on 0/3 real scans
(ratios 1.858/2.263/0.981 vs tau 0.85; a 1-99% bbox sensitivity flips S3 to
0.817, so "trigger dead" is NOT yet establishable). Before any keep/remove
conclusion about HV on real data, run HV with the trigger FORCED ON:
  3 scans x 20 trials, reusing the EXACT seeds, frozen FGR starts, selected
  features, and slab candidates of the recorded run; bypass arm = the recorded
  no_hv rows from results/e3_hv_ablation_full.json (no rerun).
Report per scan (never pooled): paired p2m median diff (forced-HV minus bypass)
with bootstrap CI, p2m median/p95, inlier, winner, scores, activation/hypothesis
counts, failures, runtime ratio. Interpretation per judge: benefit when forced
diagnoses a failed trigger, not production efficacy; absence of benefit/harm
supports removal.

Usage: python e3b_forced_trigger.py   (geo env; ~60 registrations)
"""
import os, sys, json, time
import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import e0_common as e0
from solver import estimate_normals
from selection import greedy_select, spectral, selected_info
from hypothesis import bbox_volume
from e3_hv_ablation import (trial_seed, auto_params, make_reg_fn,
                            verify_shared_init, REAL_FAIL_P2M_MM)

SRC = os.path.join(HERE, "results", "e3_hv_ablation_full.json")
N = 20


def boot_ci(v, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    bs = [float(np.median(v[rng.integers(0, len(v), len(v))])) for _ in range(B)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    d = json.load(open(SRC))
    mesh, tgt_pts, tgt_nrm, feats = e0.load_gear_cad()
    rng0 = np.random.default_rng(e0.MASTER_SEED)
    guide = [f.generate_points(120, noise_std=0.0, extent=6.0, rng=rng0) for f in feats]
    sel_idx = greedy_select(feats, guide, **e0.CONFIGS["geop2e_select"], rng=rng0)
    sub = [feats[i] for i in sel_idx]
    sub_guide = [guide[i] for i in sel_idx]
    REG = auto_params(tgt_pts)
    reg_fn = make_reg_fn(REG)
    p2m = e0.make_p2m_evaluator(mesh)
    ext = float(np.linalg.norm(np.ptp(tgt_pts, axis=0)))

    out = {"protocol": {"design": "forced-trigger HV vs recorded bypass (no rerun of bypass)",
                        "n": N, "seeds_init": "identical to e3_hv_ablation_full",
                        "interpretation": "forced benefit diagnoses a failed trigger, not production efficacy"}}
    rows = []
    for sf, label in e0.SCANS:
        # reuse the recorded frozen FGR start (cache file written by the full run)
        fgr_path = os.path.join(HERE, "artifacts",
                                f"fgr_v{e0.CONFIGS['fgr_voxel']}_d{e0.CONFIGS['fgr_max_corr_dist']}_{label}.npy")
        T_fgr = np.load(fgr_path)
        pc = e0.load_scan(os.path.join(e0.GEAR, sf))
        src = np.asarray(pc.points)
        sn = estimate_normals(src, radius=2.0, max_nn=30)
        obs_ratio = float(np.clip(bbox_volume(src) / bbox_volume(tgt_pts), 0.1, 0.9))
        for t in range(N):
            seed = trial_seed(e0.MASTER_SEED, "real", label, t)   # SAME seed as the full run
            rng = np.random.default_rng(seed)
            axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
            dT = np.eye(4)
            dT[:3, :3] = Rot.from_rotvec(axis * np.deg2rad(rng.uniform(0, 3))).as_matrix()
            dT[:3, 3] = rng.normal(size=3)
            dT[:3, 3] = dT[:3, 3] / np.linalg.norm(dT[:3, 3]) * ext * 0.005 * rng.uniform(0, 1)
            T_pert = dT @ T_fgr
            t0 = time.perf_counter()
            try:
                # forced trigger: volume_trigger=False forces triggered=True in verify()
                T, info, hv_log = verify_shared_init(
                    sub, sub_guide, tgt_pts, src, sn, reg_fn, init_T=T_pert,
                    volume_trigger=False, cutting=True, penetration=True)
                m = e0.eval_pose(src, T, p2m)
                row = dict(cell=label, trial=t, seed=seed, ok=True, **m,
                           iters=info.get("iters"), update_norm=info.get("update_norm"),
                           runtime_s=time.perf_counter() - t0, **hv_log)
            except Exception as ex:
                row = dict(cell=label, trial=t, seed=seed, ok=False, error=repr(ex)[:200],
                           runtime_s=time.perf_counter() - t0)
            rows.append(row)
        print(f"[forced {label}] done {N} trials", flush=True)

    # pair against recorded bypass rows
    byp = {(r["cell"], r["trial"]): r for r in d["rows"]
           if r["arm"] == "no_hv" and r["cell"].startswith("real_")}
    summary = {}
    for _, label in e0.SCANS:
        frows = [r for r in rows if r["cell"] == label and r["ok"]]
        v = np.array([r["p2m_median_mm"] for r in frows])
        va = np.array([byp[(f"real_{label}", r["trial"])]["p2m_median_mm"]
                       for r in frows if (f"real_{label}", r["trial"]) in byp
                       and byp[(f"real_{label}", r["trial"])]["ok"]])
        common_t = [r["trial"] for r in frows
                    if (f"real_{label}", r["trial"]) in byp and byp[(f"real_{label}", r["trial"])]["ok"]]
        vb = np.array([r["p2m_median_mm"] for r in frows if r["trial"] in common_t])
        s = dict(n_ok=len(frows), n_err=sum(1 for r in rows if r["cell"] == label and not r["ok"]),
                 forced=dict(p2m_median=float(np.median(v)),
                             p2m_p95_median=float(np.median([r["p2m_p95_mm"] for r in frows])),
                             inlier_median=float(np.median([r["inlier_pct"] for r in frows])),
                             fail_pct=float(100 * np.mean(v > REAL_FAIL_P2M_MM)),
                             p2m_median_boot95=boot_ci(v),
                             mean_n_hypotheses=float(np.mean([r["n_hypotheses"] for r in frows])),
                             winners={w: sum(1 for r in frows if r.get("winner") == w)
                                      for w in {r.get("winner") for r in frows}},
                             runtime_s_median=float(np.median([r["runtime_s"] for r in frows]))))
        if len(vb) >= 5:
            diff = vb - va
            s["paired_vs_recorded_bypass"] = dict(
                n_paired=len(vb),
                p2m_median_diff_mm=float(np.median(diff)),
                diff_boot95=boot_ci(diff))
        summary[label] = s

    out["summary"] = summary
    out["rows"] = rows
    p = os.path.join(HERE, "results", "e3b_forced_trigger.json")
    if os.path.exists(p):
        p += f".{int(time.time())}.json"
    e0.save_json(out, p)
    print(json.dumps(summary, indent=1, default=float)[:1800])


if __name__ == "__main__":
    main()
