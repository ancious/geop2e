"""E3 — HV with/without ablation (R2's exact demand: "a with/without comparison").

Minimal two-arm design per RINENG_EXPERIMENT_PLAN_MINIMAL.md (GPT-5.6-sol-max):
- arms: production full-HV vs identical pipeline with HV bypassed;
- paired trials: same input, same noise seed, same shared initial pose for BOTH
  arms (deviation from hypothesis.verify()'s per-candidate centroid init —
  logged as a protocol note; the shared start isolates the hypothesis-set effect);
- synthetic cells on real gear geometry (known-transform GT): loss 0% / 30% / 60%
  x cut aligned / oblique (0% has no cut orientation) -> 5 cells x n=20;
- real cells: the three defect scans x n=20 perturbed initialisations
  (rotation <=3 deg, translation <=0.5% CAD-diagonal ball around the frozen FGR
  coarse start);
- total 5*20*2 + 3*20*2 = 320 registrations;
- minimal logging per trial: trigger decision + V_S/V_M ratio, candidates
  generated/evaluated, total hypotheses (H_max = 1+12 = 13, frozen), winner,
  score components (rmse, penetration, score), iterations, update norm, runtime;
- metrics: symmetry-aware RRE/RTE (declared gear symmetry group) + raw; success
  = RRE_sym<1deg & RTE_sym<1mm (synthetic); real scans report p2m/inlier only.

E5 is computed from the real-scan full-HV arm (per-scan median/SD/bootstrap-CI
of p2m median, failure fraction) — no extra runs.

Usage:  python e3_hv_ablation.py [--n 20] [--cells all|synthetic|real|pilot] [--out results/e3_hv_ablation.json]
Runs in the `geo` env; pure CPU. NEVER overwrites: output path carries a timestamp suffix if it exists.
"""
from __future__ import annotations
import os, sys, json, time, argparse, hashlib
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def trial_seed(master: int, *tags) -> int:
    """Deterministic per-trial seed (gpt-review fix: Python hash() is
    process-randomized via PYTHONHASHSEED; md5 is stable across processes)."""
    h = hashlib.md5(("|".join(map(str, (master,) + tags))).encode()).hexdigest()
    return int(h[:8], 16)

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)
sys.path.insert(0, HERE)

from features import load_features_from_json                      # noqa: E402
from solver import estimate_normals, register, transform_points   # noqa: E402
from selection import greedy_select, spectral, selected_info      # noqa: E402
from hypothesis import penetration_fraction, bbox_volume  # noqa: E402
import e0_common as e0                                             # noqa: E402

TAU_VOL = 0.85          # production volume-ratio trigger (paper Sec 3.4)
H_MAX = 13              # 1 full-CAD + <=12 cut candidates (frozen before runs)
PEN_WEIGHT = 0.5        # production score weight (rmse + 0.5 * penetration)
N_TRIALS_DEFAULT = 20
SUCCESS_RRE_DEG = 1.0
SUCCESS_RTE_MM = 1.0
REAL_FAIL_P2M_MM = 1.0  # declared failure threshold for real-scan runs


# --------------------------------------------------------------------------
def auto_params(target_pts):
    ext = float(np.linalg.norm(np.ptp(target_pts, axis=0)))
    return dict(init_radius=ext * 0.18, anneal=0.9, max_iter=60, lam=1e-3,
                k_huber=ext * 0.012, angle_thresh_deg=30.0)


def make_reg_fn(REG):
    def reg_fn(points, normals, feats, init_T=None):
        return register(points, normals, feats, init_T=init_T, **REG)
    return reg_fn


def simulate_defect_ext(target_pts, rng, keep_frac, oblique):
    """Extended defect simulator (gpt-review fix): keep_frac=1 -> pure pose;
    otherwise remove a (1-keep_frac) QUANTILE slab so the realized loss equals
    the nominal loss regardless of the projection distribution, with the kept
    SIDE randomised for both aligned and oblique cuts (was: always the positive
    side). Aligned cut normal = random principal axis; oblique = random unit normal."""
    pts = target_pts.copy()
    if keep_frac >= 1.0:
        src = pts
    else:
        if oblique:
            n = rng.normal(size=3); n /= np.linalg.norm(n)
        else:
            n = np.zeros(3); n[int(rng.integers(0, 3))] = 1.0
        proj = pts @ n
        if rng.random() < 0.5:
            keep = proj > np.quantile(proj, 1 - keep_frac)   # remove LOWER slab (keep upper)
        else:
            keep = proj < np.quantile(proj, keep_frac)        # remove UPPER slab (keep lower)
        src = pts[keep]
    ext = float(np.linalg.norm(np.ptp(target_pts, axis=0)))
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    T = np.eye(4)
    T[:3, :3] = Rot.from_rotvec(axis * np.deg2rad(rng.uniform(-10, 10))).as_matrix()
    T[:3, 3] = rng.uniform(-1, 1, 3) * ext * 0.05
    return transform_points(src, T), T


def make_candidates_slab(features, guide_sets, max_cands=12):
    """Paper-conformant virtual-cut candidate generator (pilot fix).

    hypothesis.make_candidates (box windows centred inside the target bbox)
    returns ZERO candidates on the gear: parsed feature anchor points sit at
    the two end faces (z ~ 0 and ~21 mm), so no interior-centred window keeps
    >=4 features — the cut machinery never actually ran in the rebuttal code
    (the other half of the silent n=3 failure). The paper describes cutting as
    "slicing along principal axes with grid-sampled positions": here, for each
    axis and cut position in {0.25, 0.5, 0.75} of the guide-midpoint extent,
    keep the features whose anchor lies on one side of the cut plane (both
    sides tried). Dedup, keep the largest subsets first, cap at max_cands.
    """
    mids = np.array([g.mean(0) for g in guide_sets])
    lo, hi = mids.min(0), mids.max(0)
    cands = []
    for ax in range(3):
        for frac in (0.25, 0.5, 0.75):
            cut = lo[ax] + frac * (hi[ax] - lo[ax])
            for side in (0, 1):
                keep = [i for i in range(len(features))
                        if (mids[i, ax] > cut if side else mids[i, ax] < cut)]
                if len(keep) >= 4:
                    cands.append(tuple(sorted(keep)))
    uniq = []
    for c in sorted(set(cands), key=len, reverse=True):
        uniq.append(list(c))
        if len(uniq) >= max_cands:
            break
    return uniq


def verify_shared_init(features, guide_sets, target_pts, source_pts, source_normals,
                       reg_fn, init_T, *,
                       volume_trigger=True, cutting=True, penetration=True,
                       tau_vol=TAU_VOL):
    """Paired-protocol variant of hypothesis.verify: every hypothesis (full CAD +
    cut candidates) starts from the SAME init_T, so the arms differ only in the
    hypothesis set. Full minimal logging. Returns (best_T, best_info, log)."""
    vs, vm = bbox_volume(source_pts), bbox_volume(target_pts)
    ratio = vs / vm
    triggered = (ratio < tau_vol) if volume_trigger else True
    log = {"V_S/V_M": ratio, "triggered": bool(triggered), "tau_vol": tau_vol,
           "n_cand_generated": 0, "n_cand_evaluated": 0, "n_hypotheses": 1,
           "winner": "full", "score_rmse": None, "score_pen": None, "score": None}
    pen_threshold = float(np.linalg.norm(np.ptp(target_pts, axis=0))) * 0.02

    if not triggered or not cutting:
        T, info = reg_fn(source_pts, source_normals, features, init_T)
        log["n_hypotheses"] = 1
        return T, info, log

    cands = make_candidates_slab(features, guide_sets, max_cands=H_MAX - 1)
    log["n_cand_generated"] = len(cands)
    best_T, best_score, best_info, best_tag = None, np.inf, None, "full"
    trial_sets = [("full", list(range(len(features))))] + [(f"cand{i}", c) for i, c in enumerate(cands)]
    for tag, idx_set in trial_sets:
        sub = [features[i] for i in idx_set]
        if len(sub) < 4:
            continue
        log["n_cand_evaluated"] += 1 if tag != "full" else 0
        T, info = reg_fn(source_pts, source_normals, sub, init_T)
        rmse = info.get("history", [{}])[-1].get("rmse", 0.0) if info.get("history") else 0.0
        pen = penetration_fraction(transform_points(source_pts, T), target_pts, pen_threshold) if penetration else 0.0
        score = rmse + PEN_WEIGHT * pen
        if score < best_score:
            best_score, best_T, best_info, best_tag = score, T, info, tag
            log["score_rmse"], log["score_pen"], log["score"], log["winner"] = rmse, pen, score, tag
    log["n_hypotheses"] = log["n_cand_evaluated"] + 1
    return best_T, best_info, log


# --------------------------------------------------------------------------
def run_synthetic(cells, n, tgt_pts, feats, guide, sel_idx, REG, reg_fn, sym_Rs, master, p2m):
    sub = [feats[i] for i in sel_idx]
    sub_guide = [guide[i] for i in sel_idx]
    rows = []
    for cell in cells:                      # e.g. "loss30_aligned"
        keep_frac = {"loss0": 1.0, "loss30": 0.7, "loss60": 0.4}[cell.split("_")[0]]
        oblique = cell.endswith("oblique")
        for t in range(n):
            seed = trial_seed(master, "synth", cell, t)
            rng = np.random.default_rng(seed)
            src, T_gt = simulate_defect_ext(tgt_pts, rng, keep_frac, oblique)
            sn = estimate_normals(src, radius=REG["init_radius"] * 0.4, max_nn=30)
            # observed completeness estimate (NO ground-truth input; gpt-review
            # leakage fix: the old code passed 1-keep_frac = oracle loss)
            obs_ratio = float(np.clip(bbox_volume(src) / bbox_volume(tgt_pts), 0.1, 0.9))
            # SHARED COARSE START (round-2 CRITICAL fix): simulate_defect_ext
            # returns src = T_gt @ CAD-points, so the solver's source->CAD target
            # pose is inverse(T_gt) (solver defines correctness as
            # T_pred @ T_gt = I). The coarse start is therefore P @ inv(T_gt)
            # with P a small perturbation — the OLD P @ T_gt composed the
            # perturbation in the wrong frame, making the true start error
            # ~P @ T_gt^2 (up to ~20 deg). Pose-recovery-from-coarse protocol
            # (published Table A4); BOTH arms receive the IDENTICAL start.
            ext = float(np.linalg.norm(np.ptp(tgt_pts, axis=0)))
            pax = rng.normal(size=3); pax /= np.linalg.norm(pax)
            P = np.eye(4)
            P[:3, :3] = Rot.from_rotvec(pax * np.deg2rad(rng.uniform(0, 2))).as_matrix()
            P[:3, 3] = rng.normal(size=3) * ext * 0.0025   # sigma = 0.25% L
            T_coarse = P @ np.linalg.inv(T_gt)
            for arm in ("no_hv", "full_hv"):
                t0 = time.perf_counter()
                try:
                    if arm == "no_hv":
                        T, info = reg_fn(src, sn, sub, T_coarse)
                        hv_log = {"V_S/V_M": obs_ratio,
                                  "triggered": False, "n_hypotheses": 1, "winner": "full(bypass)",
                                  "score_rmse": None, "score_pen": None, "score": None,
                                  "n_cand_generated": 0, "n_cand_evaluated": 0}
                    else:
                        T, info, hv_log = verify_shared_init(
                            sub, sub_guide, tgt_pts, src, sn, reg_fn, init_T=T_coarse)
                    rre_s, rte_s, rre_r, rte_r = e0.rre_rte_sym(T, T_gt, sym_Rs)
                    m = e0.eval_pose(src, T, p2m)   # teeth-aware residual decouples pose vs feature-rmse
                    row = dict(cell=cell, arm=arm, trial=t, seed=seed, ok=True,
                               rre_sym=rre_s, rte_sym=rte_s, rre_raw=rre_r, rte_raw=rte_r,
                               success=bool(rre_s < SUCCESS_RRE_DEG and rte_s < SUCCESS_RTE_MM),
                               **m,
                               iters=info.get("iters"), update_norm=info.get("update_norm"),
                               runtime_s=time.perf_counter() - t0, **hv_log)
                except Exception as ex:
                    row = dict(cell=cell, arm=arm, trial=t, seed=seed, ok=False,
                               error=repr(ex)[:200], runtime_s=time.perf_counter() - t0)
                rows.append(row)
            print(f"[synth {cell} t{t}] " + " ".join(
                f"{r['arm']}={'ERR' if not r['ok'] else ('ok' if r['success'] else 'miss')}({r['rre_sym']:.2f}deg)"
                if r["ok"] else f"{r['arm']}=ERR" for r in rows[-2:]), flush=True)
    return rows


def run_real(n, tgt_pts, tgt_nrm, feats, guide, sel_idx, REG, reg_fn, p2m, master, cache_dir):
    """Real-scan arms: shared perturbed FGR start; metrics p2m/inlier (no pose GT)."""
    sub = [feats[i] for i in sel_idx]
    sub_guide = [guide[i] for i in sel_idx]
    ext = float(np.linalg.norm(np.ptp(tgt_pts, axis=0)))
    rows = []
    for sf, label in e0.SCANS:
        pc = e0.load_scan(os.path.join(e0.GEAR, sf))
        src = np.asarray(pc.points)
        sn = estimate_normals(src, radius=2.0, max_nn=30)  # fixed 2.0 mm (logged in protocol)
        obs_ratio = float(np.clip(bbox_volume(src) / bbox_volume(tgt_pts), 0.1, 0.9))
        # frozen FGR coarse start per scan (cache key includes the frozen config)
        os.makedirs(cache_dir, exist_ok=True)
        fgr_path = os.path.join(cache_dir, f"fgr_v{e0.CONFIGS['fgr_voxel']}_d{e0.CONFIGS['fgr_max_corr_dist']}_{label}.npy")
        if os.path.exists(fgr_path):
            T_fgr = np.load(fgr_path)
        else:
            T_fgr = e0.fgr_coarse(pc, tgt_pts, tgt_nrm)
            np.save(fgr_path, T_fgr)
        for t in range(n):
            seed = trial_seed(master, "real", label, t)
            rng = np.random.default_rng(seed)
            axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
            dT = np.eye(4)
            dT[:3, :3] = Rot.from_rotvec(axis * np.deg2rad(rng.uniform(0, 3))).as_matrix()
            dT[:3, 3] = rng.normal(size=3)
            dT[:3, 3] = dT[:3, 3] / np.linalg.norm(dT[:3, 3]) * ext * 0.005 * rng.uniform(0, 1)
            T_pert = dT @ T_fgr
            for arm in ("no_hv", "full_hv"):
                t0 = time.perf_counter()
                try:
                    if arm == "no_hv":
                        T, info = reg_fn(src, sn, sub, T_pert)
                        hv_log = {"V_S/V_M": obs_ratio, "triggered": False, "n_hypotheses": 1,
                                  "winner": "full(bypass)", "score_rmse": None,
                                  "score_pen": None, "score": None,
                                  "n_cand_generated": 0, "n_cand_evaluated": 0}
                    else:
                        T, info, hv_log = verify_shared_init(
                            sub, sub_guide, tgt_pts, src, sn, reg_fn, init_T=T_pert)
                    m = e0.eval_pose(src, T, p2m)
                    row = dict(cell=f"real_{label}", arm=arm, trial=t, seed=seed, ok=True, **m,
                               iters=info.get("iters"), update_norm=info.get("update_norm"),
                               runtime_s=time.perf_counter() - t0, **hv_log)
                except Exception as ex:
                    row = dict(cell=f"real_{label}", arm=arm, trial=t, seed=seed, ok=False,
                               error=repr(ex)[:200], runtime_s=time.perf_counter() - t0)
                rows.append(row)
            print(f"[real {label} t{t}] " + " ".join(
                f"{r['arm']}={r.get('p2m_median_mm', float('nan')):.3f}mm" if r["ok"] else f"{r['arm']}=ERR"
                for r in rows[-2:]), flush=True)
    return rows


# --------------------------------------------------------------------------
def summarise(rows):
    """Per-cell/arm summaries + paired stats (gpt-review fixes: paired dict is
    created before first write; n_paired recorded; binomtest is already two-
    sided (no x2); Holm correction across all cell-level McNemar tests; per-arm
    bootstrap CI for real cells)."""
    import scipy.stats as st
    out = {}
    cells = sorted({r["cell"] for r in rows})
    mcnemar_cells = []          # (cell, p) collected for Holm
    for c in cells:
        cr = [r for r in rows if r["cell"] == c]
        for arm in ("no_hv", "full_hv"):
            ar = [r for r in cr if r["arm"] == arm and r["ok"]]
            s = {"n_ok": len(ar), "n_err": sum(1 for r in cr if r["arm"] == arm and not r["ok"])}
            if ar:
                if c.startswith("real_"):
                    v = np.array([r["p2m_median_mm"] for r in ar])
                    rng = np.random.default_rng(1)
                    boots = [float(np.median(v[rng.integers(0, len(v), len(v))])) for _ in range(2000)]
                    s.update(p2m_median_of_trial_medians=float(np.median(v)),
                             p2m_sd=float(np.std(v)),
                             p2m_median_boot95=[float(np.percentile(boots, 2.5)),
                                                float(np.percentile(boots, 97.5))],
                             p2m_min=float(v.min()), p2m_max=float(v.max()),
                             fail_pct=float(100 * np.mean(v > REAL_FAIL_P2M_MM)))
                else:
                    rr = np.array([r["rre_sym"] for r in ar]); rt = np.array([r["rte_sym"] for r in ar])
                    succ = np.array([r["success"] for r in ar])
                    k, ntot = int(succ.sum()), len(succ)
                    lo, hi = st.binomtest(k, ntot, 0.5).proportion_ci(0.95)
                    s.update(success=f"{k}/{ntot}",
                             success_exact95=[float(lo), float(hi)],
                             rre_sym_median=float(np.median(rr)), rre_sym_p75=float(np.percentile(rr, 75)),
                             rte_sym_median=float(np.median(rt)))
            out[f"{c}|{arm}"] = s
        # paired stats (round-2 fix: errored trials count as FAILURES for the
        # success pairing rather than being dropped — differential arm failures
        # must not bias the comparison; continuous metrics use both-ok pairs)
        out[f"{c}|paired"] = {"n_paired": 0}
        a = {r["trial"]: r for r in cr if r["arm"] == "no_hv"}
        b = {r["trial"]: r for r in cr if r["arm"] == "full_hv"}
        common = sorted(set(a) & set(b))
        out[f"{c}|paired"]["n_paired"] = len(common)
        out[f"{c}|paired"]["n_err_treated_as_failure"] = int(sum(1 for i in common
                                                                 if not (a[i]["ok"] and b[i]["ok"])))
        if len(common) >= 5:
            both_ok = [i for i in common if a[i]["ok"] and b[i]["ok"]]
            if c.startswith("real_"):
                va = np.array([a[i]["p2m_median_mm"] for i in both_ok])
                vb = np.array([b[i]["p2m_median_mm"] for i in both_ok])
                key = "p2m_median_diff(full-no)_mm"
            else:
                va = np.array([a[i]["rre_sym"] for i in both_ok])
                vb = np.array([b[i]["rre_sym"] for i in both_ok])
                key = "rre_sym_diff(full-no)_deg"
                sa = np.array([a[i]["success"] if a[i]["ok"] else False for i in common]).astype(int)
                sb = np.array([b[i]["success"] if b[i]["ok"] else False for i in common]).astype(int)
                n01 = int(np.sum((sa == 0) & (sb == 1))); n10 = int(np.sum((sa == 1) & (sb == 0)))
                out[f"{c}|paired"]["mcnemar_no_vs_yes"] = f"{n10}vs{n01}"
                if n01 + n10 > 0:
                    p = float(st.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue)  # two-sided already
                else:
                    p = 1.0   # zero discordance contributes p=1 to the prespecified family
                out[f"{c}|paired"]["mcnemar_exact_p"] = p
                mcnemar_cells.append((c, p))          # ALL cells in the Holm family
                out[f"{c}|paired"]["success_diff(full-no)_pp"] = float(100 * (sb.mean() - sa.mean()))
                # paired success-difference bootstrap CI (uncertainty fix)
                rngs = np.random.default_rng(2)
                diffs = []
                for _ in range(2000):
                    idx = rngs.integers(0, len(common), len(common))
                    diffs.append(100 * (sb[idx].mean() - sa[idx].mean()))
                out[f"{c}|paired"]["success_diff_boot95"] = [float(np.percentile(diffs, 2.5)),
                                                             float(np.percentile(diffs, 97.5))]
            if len(both_ok) >= 5:
                d = vb - va
                out[f"{c}|paired"][key] = float(np.median(d))
                rng = np.random.default_rng(0)
                boots = [float(np.median(d[rng.integers(0, len(d), len(d))])) for _ in range(2000)]
                out[f"{c}|paired"][key + "_boot95"] = [float(np.percentile(boots, 2.5)),
                                                        float(np.percentile(boots, 97.5))]
                if c.startswith("real_"):
                    rel = 100.0 * d / np.maximum(np.abs(va), 1e-9)
                    out[f"{c}|paired"]["p2m_rel_diff(full-no)_pct"] = float(np.median(rel))
                    rb = [float(np.median(rel[rng.integers(0, len(rel), len(rel))])) for _ in range(2000)]
                    out[f"{c}|paired"]["p2m_rel_diff_boot95"] = [float(np.percentile(rb, 2.5)),
                                                                 float(np.percentile(rb, 97.5))]
    # Holm across all cell-level McNemar tests (gpt-review: multiplicity fix)
    if mcnemar_cells:
        order = sorted(range(len(mcnemar_cells)), key=lambda i: mcnemar_cells[i][1])
        m = len(order)
        prev = 0.0
        for rank, i in enumerate(order):
            c, p = mcnemar_cells[i]
            adj = min(1.0, (m - rank) * p)
            adj = max(adj, prev)          # Holm step-down monotonicity
            prev = adj
            out[f"{c}|paired"]["mcnemar_holm_p"] = float(adj)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_TRIALS_DEFAULT)
    ap.add_argument("--cells", default="all", choices=["all", "synthetic", "real", "pilot"])
    ap.add_argument("--out", default=os.path.join(HERE, "results", "e3_hv_ablation.json"))
    args = ap.parse_args()

    master = e0.MASTER_SEED
    mesh, tgt_pts, tgt_nrm, feats = e0.load_gear_cad()
    rng = np.random.default_rng(master)
    guide = [f.generate_points(120, noise_std=0.0, extent=6.0, rng=rng) for f in feats]
    sel_idx = greedy_select(feats, guide, **e0.CONFIGS["geop2e_select"], rng=rng)
    sp = spectral(selected_info(feats, guide, sel_idx))
    REG = auto_params(tgt_pts)
    reg_fn = make_reg_fn(REG)
    axis, N_sym, sym_match = e0.gear_axis_and_symmetry(mesh, feats)
    sym_Rs = e0.symmetry_rotations(axis, N_sym)
    print(f"target {tgt_pts.shape} feats {len(feats)} selected {len(sel_idx)} "
          f"lambda_min={sp['lambda_min']:.2f} kappa={sp['kappa']:.2f} "
          f"N_sym={N_sym} (match={sym_match:.3f})", flush=True)

    n = args.n
    pilot = args.cells == "pilot"
    n_use = 2 if pilot else n
    rows = []
    if args.cells in ("all", "synthetic", "pilot"):
        cells = (["loss0", "loss30_aligned", "loss60_aligned"] if pilot else
                 ["loss0", "loss30_aligned", "loss30_oblique", "loss60_aligned", "loss60_oblique"])
        p2m_eval = e0.make_p2m_evaluator(mesh)
        rows += run_synthetic(cells, n_use, tgt_pts, feats, guide, sel_idx, REG, reg_fn,
                              sym_Rs, master, p2m_eval)
    if args.cells in ("all", "real", "pilot"):
        if args.cells == "real":
            p2m_eval = e0.make_p2m_evaluator(mesh)
        rows += run_real(n_use, tgt_pts, tgt_nrm, feats, guide,
                         sel_idx, REG, reg_fn, p2m_eval, master,
                         cache_dir=os.path.join(HERE, "artifacts"))

    summary = summarise(rows)
    out_path = args.out
    if os.path.exists(out_path):      # never overwrite: extend suffix until unused
        import uuid
        while True:
            cand = out_path + f".{uuid.uuid4().hex}.json"
            if not os.path.exists(cand):
                out_path = cand
                break
    # environment / provenance records (round-2 reproducibility fix)
    import hashlib, subprocess
    def _sha(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16] if os.path.exists(p) else None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         cwd=PARENT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = None
    tgt_npz = os.path.join(HERE, "artifacts", "gear_target_20260827.npz")
    env_rec = {"open3d": __import__("open3d").__version__, "numpy": np.__version__,
               "scipy": __import__("scipy").__version__, "git_commit": commit,
               "source_sha256": _sha(__file__), "target_npz_sha256": _sha(tgt_npz),
               "note": ("geop2e/ is untracked in the parent repo, so source_sha256 of "
                        "this file is the authoritative code identity"),
               "solver_params_REG": REG, "selected_feature_indices": list(map(int, sel_idx)),
               "selected_kinds": [feats[i].kind for i in sel_idx]}
    e0.save_json({"protocol": {
                     "arms": ["no_hv", "full_hv(shared_init)"],
                     "n_per_cell_requested": n, "n_per_cell_used": n_use, "pilot": pilot,
                     "H_max": H_MAX, "tau_vol": TAU_VOL, "pen_weight": PEN_WEIGHT,
                     "success_rule": f"RRE_sym<{SUCCESS_RRE_DEG}deg & RTE_sym<{SUCCESS_RTE_MM}mm",
                     "real_fail_rule": f"p2m_median>{REAL_FAIL_P2M_MM}mm",
                     "N_sym": N_sym, "N_sym_match_frac": sym_match, "master_seed": master,
                     "target_ratio_source": "observed bbox-volume ratio (no GT input)",
                     "real_normals_radius_mm": 2.0,
                     "config_provenance": ("configs inherited from the published pipeline "
                                           "(real_gear_experiment.auto_params; FGR voxel 1.0); "
                                           "no parameter was tuned on the three reported scans "
                                           "in this run; solver params are CAD-scale-derived"),
                     "interpretation_rule": ("material HV benefit per cell = success diff >= +20pp "
                                             "or paired median diff <= -10% (p2m) with bootstrap "
                                             "95% CI excluding 0; Holm-adjusted McNemar reported "
                                             "across cells. The synthetic paired RRE effect is an "
                                             "ESTIMATION-ONLY endpoint (pilot shows a yaw-"
                                             "observability success floor: 2 planes + 6 cylinders "
                                             "cannot constrain rotation about the gear axis; teeth "
                                             "are not primitives), so no RRE materiality threshold "
                                             "is prespecified — RRE effects are reported with CIs "
                                             "and interpreted descriptively"),
                     "shared_init_note": ("both arms start from the SAME init — synthetic: "
                                          "P @ inv(T_gt), P = U(0,2deg) rotation + N(0, 0.25%L) "
                                          "translation (pose-recovery-from-coarse, Table-A4 "
                                          "protocol); real: frozen FGR @ same per-trial "
                                          "perturbation. Differs from production verify() "
                                          "centroid init to isolate the hypothesis-set effect"),
                     "candidate_generator": ("axis-plane slab cuts at 25/50/75% of the anchor "
                                             "extent, both sides (paper Sec 3.4 semantics); the "
                                             "shipped hypothesis.make_candidates box-window "
                                             "generator returns 0 candidates on this gear "
                                             "(anchors sit on the end faces) — logged as a "
                                             "finding")},
                  "fim": sp, "env": env_rec, "rows": rows, "summary": summary}, out_path)
    print(json.dumps(summary, indent=2, default=float)[:3000])


if __name__ == "__main__":
    main()
