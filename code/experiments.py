"""GeoP2E small-data experiments for the Measurement major-revision rebuttal.

Synthetic primitives with known GT; small point counts + multiple seeds.
Outputs results/results.json + markdown tables.

  E1  Analytical vs numerical Jacobian agreement (validates UNIFIED_MATH.md;
      fixes the cylinder-was-numerical honesty gap).
  E2  Feature-selection criteria on a REDUNDANT scene (8 parallel planes + cyl
      + sphere): greedy E/D/A/cond keep H well-conditioned, random is frequently
      rank-deficient. Reports spectral (lambda_min, kappa) + random-degeneracy
      rate + small-perturbation registration.
  E3  Module ablation, split:
        (a) solver efficiency: numerical vs analytical Jacobian (equal accuracy,
            analytical faster); FIM-selected vs all-features (equal accuracy,
            fewer features -> faster).
        (b) robust kernel: Huber vs L2 under heavy Gaussian noise (sigma=0.1).
  E4  Metrology: known-SE(3) pose recovery over N trials (independent GT).
"""
from __future__ import annotations
import json, time, os
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from features import GeometricFeature, PlaneFeature, CylinderFeature, SphereFeature
from solver import estimate_normals, register, evaluate_alignment, transform_points
from selection import greedy_select, selected_info, spectral
from baselines import icp_point_to_point, icp_point_to_plane

RNG_SEED = 1234
N_PER = 600
EXTENT = 8.0
REG_KW = dict(init_radius=4.0, anneal=0.9, max_iter=80, lam=1e-3,
              k_huber=0.5, angle_thresh_deg=30.0)


def make_scene() -> list[GeometricFeature]:
    """Balanced 10-feature scene (for ablation/metrology)."""
    return [PlaneFeature([0, 0, 1], 0.0),  PlaneFeature([1, 0, 0], 4.0),
            PlaneFeature([1, 0, 0], -4.0), PlaneFeature([0, 1, 0], 4.0),
            PlaneFeature([0, 1, 0], -4.0), PlaneFeature([0, 0, 1], 6.0),
            CylinderFeature([0, 0, 0], [0, 0, 1], 1.0),
            CylinderFeature([3, 3, 0], [0, 1, 0], 0.8),
            SphereFeature([-3, -3, 1.0], 1.0),
            SphereFeature([3, -3, 1.0], 1.2)]


def make_redundant_scene() -> list[GeometricFeature]:
    """8 parallel horizontal planes + 2 cylinders + 1 sphere = 11 candidates.
    Any 6-subset dominated by parallel planes is rank-deficient (in-plane
    translation & about-normal rotation unconstrained)."""
    feats = [PlaneFeature([0, 0, 1], z) for z in (0, 1, 2, 3, 4, 5, 6, 7)]
    feats += [CylinderFeature([0, 0, 4], [0, 1, 0], 0.8),
              CylinderFeature([5, 5, 4], [0, 0, 1], 0.8),
              SphereFeature([-5, 5, 4], 1.0)]
    return feats


def sample_cloud(feats, rng, n_per=N_PER, noise=0.02, extent=EXTENT):
    return np.vstack([f.generate_points(n_per, noise_std=noise, extent=extent, rng=rng) for f in feats])


def random_pose(rng, rot_deg=10.0, trans=0.5):
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    ang = np.deg2rad(rng.uniform(-rot_deg, rot_deg))
    T = np.eye(4); T[:3, :3] = Rot.from_rotvec(axis * ang).as_matrix()
    T[:3, 3] = rng.uniform(-trans, trans, 3)
    return T


def _trial(feats_used, feats_sample, rng, rot_deg, trans, noise=0.02, **reg_kw):
    T_gt = random_pose(rng, rot_deg, trans)
    src = transform_points(sample_cloud(feats_sample, rng, noise=noise), T_gt)
    normals = estimate_normals(src, radius=0.7, max_nn=30)
    kw = {**REG_KW, **reg_kw}
    t0 = time.time()
    T_est, _ = register(src, normals, feats_used, **kw)
    dt = time.time() - t0
    re, te = evaluate_alignment(T_est, T_gt)
    return re, te, dt


# ---------------------------------------------------------------------------
def E1_jacobian(out):
    feats = [PlaneFeature([0, 0, 1], 0.0), CylinderFeature([0, 0, 0], [0, 1, 0], 1.5),
             SphereFeature([4.0, 0.0, 2.0], 2.0)]
    T = np.eye(4); T[:3, :3] = Rot.from_euler('yz', [-15, 5], degrees=True).as_matrix(); T[:3, 3] = [0.8, -1.5, 0.5]
    rng = np.random.default_rng(0)
    rows = []
    for f in feats:
        x = transform_points(f.generate_points(2000, extent=8.0, rng=rng), T)
        Ja, Jn = f.compute_jacobian(x), f.compute_numerical_jacobian(x)
        diff = Ja - Jn
        rows.append({"primitive": f.kind,
                     "max_abs_err": float(np.max(np.abs(diff))),
                     "rel_L2": float(np.linalg.norm(diff) / (np.linalg.norm(Ja) + 1e-12))})
    out["E1_jacobian"] = rows


def E2_selection(out):
    """Spectral comparison. For greedy criteria: lambda_min/kappa of the chosen
    subset. For 'random': EXACT enumeration of all C(11,6)=462 six-feature
    subsets (no sampling noise) -> full lambda_min distribution + degeneracy
    rate. Degeneracy threshold lambda_min < 1 (scale-dependent; reported
    alongside the median so the reader can calibrate)."""
    import itertools
    from selection import primitive_info
    feats = make_redundant_scene()
    nF = len(feats)
    rng = np.random.default_rng(RNG_SEED)
    guide = [f.generate_points(400, extent=EXTENT, rng=rng) for f in feats]
    Hk = [primitive_info(f, g) for f, g in zip(feats, guide)]
    rows = []
    for crit in ["e", "d", "a", "trace", "area", "cond", "all"]:
        sel = greedy_select(feats, guide, criterion=crit, target_size=6, min_features=4, rng=rng)
        sp = spectral(selected_info(feats, guide, sel))
        rows.append({"criterion": crit, "n_selected": len(sel),
                     "lambda_min": sp["lambda_min"], "kappa": sp["kappa"]})
    # exact enumeration over all 6-subsets
    lmins = []
    for combo in itertools.combinations(range(nF), 6):
        H = sum(Hk[i] for i in combo)
        lmins.append(np.linalg.eigvalsh(H)[0])
    lmins = np.array(lmins)
    rows.append({"criterion": "random(exact)", "n_selected": 6,
                 "n_subsets": len(lmins),
                 "lambda_min_median": float(np.median(lmins)),
                 "lambda_min_min": float(lmins.min()),
                 "lambda_min_max": float(lmins.max()),
                 "pct_degenerate_lt1": float(np.mean(lmins < 1.0) * 100),
                 "pct_degenerate_lt10": float(np.mean(lmins < 10.0) * 100)})
    out["E2_selection"] = rows


def E3_ablation(out, n_trials=20):
    feats = make_scene()
    rng = np.random.default_rng(RNG_SEED)
    guide = [f.generate_points(400, extent=EXTENT, rng=rng) for f in feats]
    sel_idx = greedy_select(feats, guide, criterion="e", target_size=6, min_features=4, rng=rng)
    sub = [feats[i] for i in sel_idx]

    def _row(name, res, tes, times):
        res, tes, times = map(np.array, (res, tes, times))
        succ = (res < 1.0) & (tes < 0.05)
        return {"config": name, "n": n_trials,
                "rot_median": float(np.median(res)), "rot_p95": float(np.percentile(res, 95)),
                "trans_median": float(np.median(tes)),
                "success": f"{int(succ.sum())}/{n_trials}",
                "time_ms_median": float(np.median(times) * 1000),
                "time_ms_std": float(np.std(times) * 1000)}

    rows = []
    for name, jac, fs in [("numerical_Jac + all", "numerical", feats),
                          ("analytical_Jac + all", "analytic", feats),
                          ("analytical_Jac + FIM-Eopt", "analytic", sub)]:
        res, tes, times = [], [], []
        for s in range(n_trials):
            r = np.random.default_rng(RNG_SEED + 200 + s)
            re, te, dt = _trial(fs, feats, r, rot_deg=10.0, trans=0.4, noise=0.02,
                                jacobian_type=jac, kernel="huber")
            res.append(re); tes.append(te); times.append(dt)
        rows.append(_row(name, res, tes, times))
    for name, ker in [("analytical+L2 (sigma=0.1)", "l2"),
                      ("analytical+Huber (sigma=0.1)", "huber")]:
        res, tes, times = [], [], []
        for s in range(n_trials):
            r = np.random.default_rng(RNG_SEED + 250 + s)
            re, te, dt = _trial(sub, feats, r, rot_deg=10.0, trans=0.4, noise=0.10,
                                jacobian_type="analytic", kernel=ker)
            res.append(re); tes.append(te); times.append(dt)
        rows.append(_row(name, res, tes, times))
    out["E3_ablation"] = rows


def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for a binomial proportion (success rate)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n; den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    spread = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(center - spread), float(center + spread)


def E4_metrology(out, n_trials=30):
    feats = make_scene()
    rng = np.random.default_rng(RNG_SEED)
    guide = [f.generate_points(400, extent=EXTENT, rng=rng) for f in feats]
    sub = [feats[i] for i in greedy_select(feats, guide, criterion="e", target_size=6, min_features=4, rng=rng)]
    rots, trans = [], []
    for s in range(n_trials):
        r = np.random.default_rng(RNG_SEED + 300 + s)
        re, te, _ = _trial(sub, feats, r, rot_deg=12.0, trans=0.5, noise=0.02)
        rots.append(re); trans.append(te)
    rots, trans = np.array(rots), np.array(trans)
    succ = (rots < 1.0) & (trans < 0.05)

    def stats(x):
        x = np.asarray(x)
        return {"n": len(x), "mean": float(np.mean(x)), "median": float(np.median(x)),
                "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
                "p95": float(np.percentile(x, 95)),
                "mean_ci95_half": float(1.96 * np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0}

    k = int(succ.sum())
    clo, chi = wilson_ci(k, n_trials)
    out["E4_metrology"] = {
        "n_trials": n_trials,
        "rot_deg_ALL": stats(rots), "trans_ALL": stats(trans),
        "rot_deg_success_only": stats(rots[succ]) if k > 1 else stats(rots),
        "trans_success_only": stats(trans[succ]) if k > 1 else stats(trans),
        "success_count": f"{k}/{n_trials}",
        "success_rate": float(k / n_trials),
        "success_wilson95": [clo, chi],
        "failed_rot_deg": [float(v) for v in rots[~succ]],
        "failed_trans": [float(v) for v in trans[~succ]],
        "note": "Known synthetic SE(3) transforms = independent pose GT (no CMM). "
                "ALL-trial stats are primary (include failures); success-only is secondary.",
    }


def E2b_selection_accuracy(out, n_trials=10):
    """Per-criterion registration accuracy & time under SMALL perturbation (rot<=3deg,
    trans<=0.2) to decouple solver-basin effects from selection quality (R3-3).
    On the redundant scene, well-conditioned subsets (E/D/A/cond) should register
    accurately; degenerate random subsets should slide (high error)."""
    feats = make_redundant_scene()
    rng = np.random.default_rng(RNG_SEED)
    guide = [f.generate_points(400, extent=EXTENT, rng=rng) for f in feats]
    rows = []
    for crit in ["e", "d", "a", "trace", "area", "cond", "all"]:
        sel = greedy_select(feats, guide, criterion=crit, target_size=6, min_features=4, rng=rng)
        sub = [feats[i] for i in sel]
        res, tes, times = [], [], []
        for s in range(n_trials):
            r = np.random.default_rng(RNG_SEED + 800 + s)
            re, te, dt = _trial(sub, feats, r, rot_deg=3.0, trans=0.2)   # small perturbation
            res.append(re); tes.append(te); times.append(dt)
        rows.append({"criterion": crit,
                     "rot_median": float(np.median(res)), "rot_p95": float(np.percentile(res, 95)),
                     "trans_median": float(np.median(tes)),
                     "success": f"{int(((np.array(res)<1.0)&(np.array(tes)<0.05)).sum())}/{n_trials}",
                     "time_ms": float(np.median(times) * 1000)})
    # random: median over n_trials random draws (each a different 6-subset)
    rres, rtes = [], []
    for s in range(n_trials):
        r = np.random.default_rng(RNG_SEED + 880 + s)
        sel = greedy_select(feats, guide, criterion="random", target_size=6, rng=r)
        sub = [feats[i] for i in sel]
        re, te, _ = _trial(sub, feats, r, rot_deg=3.0, trans=0.2)
        rres.append(re); rtes.append(te)
    rows.append({"criterion": "random",
                 "rot_median": float(np.median(rres)), "rot_p95": float(np.percentile(rres, 95)),
                 "trans_median": float(np.median(rtes)),
                 "success": f"{int(((np.array(rres)<1.0)&(np.array(rtes)<0.05)).sum())}/{n_trials}"})
    out["E2b_selection_accuracy"] = rows


def E6_convergence(out, n_trials=5):
    """Residual-vs-iteration convergence: analytical vs numerical Jacobian (R1-5).
    Reports iters-to-threshold and a representative residual curve (median trial)."""
    feats = make_scene()
    rng = np.random.default_rng(RNG_SEED)
    guide = [f.generate_points(400, extent=EXTENT, rng=rng) for f in feats]
    sub = [feats[i] for i in greedy_select(feats, guide, criterion="e", target_size=6, min_features=4, rng=rng)]
    rows = []
    for jac in ["analytic", "numerical"]:
        iters_list, final_rmse, curves = [], [], []
        for s in range(n_trials):
            r = np.random.default_rng(RNG_SEED + 600 + s)
            T_gt = random_pose(r, rot_deg=10.0, trans=0.4)
            src = transform_points(sample_cloud(feats, r), T_gt)
            normals = estimate_normals(src, radius=0.7, max_nn=30)
            _, info = register(src, normals, sub, jacobian_type=jac, kernel="huber", **REG_KW)
            h = info["history"]
            curves.append([x["rmse"] for x in h])
            iters_list.append(info["iters"])
            final_rmse.append(h[-1]["rmse"] if h else 0.0)
        # representative curve = median-length trial
        rep = min(curves, key=len)
        rows.append({"jacobian": jac, "iters_median": float(np.median(iters_list)),
                     "final_rmse_median": float(np.median(final_rmse)),
                     "curve_first": float(rep[0]) if rep else 0.0,
                     "curve_last": float(rep[-1]) if rep else 0.0,
                     "curve_n": len(rep)})
    out["E6_convergence"] = rows
    # dump a representative residual curve (analytic) for plotting
    r = np.random.default_rng(RNG_SEED + 600)
    T_gt = random_pose(r, rot_deg=10.0, trans=0.4)
    src = transform_points(sample_cloud(feats, r), T_gt)
    normals = estimate_normals(src, radius=0.7, max_nn=30)
    _, info = register(src, normals, sub, jacobian_type="analytic", kernel="huber", **REG_KW)
    out["E6_curve_analytic"] = [(x["iter"], x["rmse"], x["radius"]) for x in info["history"]]


def md_table(rows, cols):
    o = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append(f"{v:.4g}" if isinstance(v, float) else (str(v) if v is not None else ""))
        o.append("| " + " | ".join(cells) + " |")
    return "\n".join(o)


def main():
    import sys, platform
    out = {}; t0 = time.time()
    print("E1 ..."); E1_jacobian(out)
    print("E2 ..."); E2_selection(out)
    print("E2b ..."); E2b_selection_accuracy(out)
    print("E3 ..."); E3_ablation(out)
    print("E4 ..."); E4_metrology(out)
    print("E6 ..."); E6_convergence(out)
    out["config"] = {
        "REG_KW": {k: (list(v) if isinstance(v, tuple) else v) for k, v in REG_KW.items()},
        "n_per_feature": N_PER, "extent": EXTENT, "rng_seed": RNG_SEED,
        "E1_n_per_primitive": 2000,
        "E2_scene": "8 parallel planes (z=0..7) + 2 cylinders + 1 sphere (11 candidates)",
        "E3a_scene": "balanced 10-feature scene (6 planes incl. parallel pairs + 2 cyl + 2 sphere)",
        "noise_E3a": 0.02, "noise_E3b": 0.10, "noise_E4": 0.02,
        "perturb_E3": "rot<=10deg trans<=0.4", "perturb_E4": "rot<=12deg trans<=0.5",
        "units": "synthetic, dimensionless (scene extent ~8); translation errors reported in these units, NOT mm",
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "scipy": __import__("scipy").__version__,
    }
    out["meta"] = {"total_sec": time.time() - t0}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/results.json", "w"), indent=2)

    print("\n=== E1 analytical-vs-numerical Jacobian (N=2000 pts/primitive) ===")
    print(md_table(out["E1_jacobian"], ["primitive", "max_abs_err", "rel_L2"]))
    print("\n=== E2 selection criteria (redundant scene, spectral) ===")
    print(md_table([r for r in out["E2_selection"] if not r["criterion"].startswith("random")],
                   ["criterion", "n_selected", "lambda_min", "kappa"]))
    rnd = [r for r in out["E2_selection"] if r["criterion"].startswith("random")][0]
    print(f"  [random = EXACT enumeration of all {rnd['n_subsets']} 6-subsets]: "
          f"lambda_min median={rnd['lambda_min_median']:.1f} min={rnd['lambda_min_min']:.2f} "
          f"max={rnd['lambda_min_max']:.0f} | %deg(Lmin<1)={rnd['pct_degenerate_lt1']:.1f}% "
          f"%deg(Lmin<10)={rnd['pct_degenerate_lt10']:.1f}%")
    print("\n=== E2b per-criterion registration accuracy (small perturbation rot<=3deg) ===")
    print(md_table(out["E2b_selection_accuracy"], ["criterion", "rot_median", "rot_p95", "trans_median", "success", "time_ms"]))
    print("\n=== E3 module ablation ===")
    print(md_table(out["E3_ablation"], ["config", "n", "rot_median", "rot_p95", "trans_median", "success", "time_ms_median", "time_ms_std"]))
    m = out["E4_metrology"]
    print("\n=== E4 metrology (known-SE(3), independent GT) ===")
    a = m["rot_deg_ALL"]; ta = m["trans_ALL"]
    print(f"  rot(deg) ALL  : median={a['median']:.4f} mean={a['mean']:.4f} p95={a['p95']:.4f} meanCI95+-={a['mean_ci95_half']:.4f}  (n={a['n']})")
    print(f"  trans    ALL  : median={ta['median']:.5f} mean={ta['mean']:.5f} p95={ta['p95']:.5f}")
    s = m["rot_deg_success_only"]
    print(f"  rot(deg) SUCC : median={s['median']:.4f} mean={s['mean']:.4f} (secondary, n={s['n']})")
    clo, chi = m["success_wilson95"]
    print(f"  success {m['success_count']} = {m['success_rate']*100:.1f}%  Wilson95=[{clo*100:.1f}%, {chi*100:.1f}%]")
    if m["failed_rot_deg"]:
        print(f"  failed cases rot(deg)={[round(x,2) for x in m['failed_rot_deg']]} "
              f"trans={[round(x,3) for x in m['failed_trans']]} (basin failures @ large perturbation)")
    print("\n=== E6 convergence (residual-vs-iteration, analytical vs numerical) ===")
    print(md_table(out["E6_convergence"], ["jacobian", "iters_median", "final_rmse_median", "curve_first", "curve_last", "curve_n"]))
    print(f"  analytic curve (iter,rmse,radius) head: {out['E6_curve_analytic'][:3]} ... tail: {out['E6_curve_analytic'][-2:]}")
    print(f"\ntotal {out['meta']['total_sec']:.1f}s")


if __name__ == "__main__":
    main()
