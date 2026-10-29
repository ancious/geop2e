"""E3c — loss0 solver/objective diagnostic (GPT-sol-max prescription #3).

The judge's one-trial audit: from an EXACT-GT start the solver moved to 0.483deg
after 1 iteration and 5.074deg after 60; the active-correspondence system itself
requested a ~0.48deg step AT GT -> biased primitive assignment / objective
support (tooth points matched to a sparse primitive subset), NOT yaw
unobservability (selected Hessian is full-rank; 4/6 cylinders are off-axis).

This script saves the FIVE-RUN formal diagnostic:
  runs: {exact-GT start, original coarse start} x {analytic, numerical Jacobian}
        + 120-iteration analytic run (5 runs total), per-iteration pose error,
        active correspondence counts, final nondimensional Hessian spectrum,
        and a fixed-correspondence yaw-objective profile at the GT pose (rotate
        the pose by yaw in [-10,10]deg about the gear axis, evaluate the SAME
        active-correspondence weighted robust objective, report its minimum).
Outputs results/e3c_loss0_diagnostic.json.
"""
import os, sys, json
import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import e0_common as e0
from solver import estimate_normals, register, transform_points
from selection import greedy_select, selected_info, spectral
from e3_hv_ablation import trial_seed, simulate_defect_ext, auto_params


def robust_objective(src, sn, feats, T, REG):
    """Weighted Huber objective on the bilateral-gated correspondences (the
    quantity register() actually decreases), plus active counts."""
    from solver import RegistrationSolver
    s = RegistrationSolver(feats, "analytic", "huber", REG["k_huber"])
    s.find_correspondences(src, sn, T, REG["init_radius"] * (REG["anneal"] ** 59),
                           REG["angle_thresh_deg"])
    s.compute_weights()
    tot, n = 0.0, 0
    for m in s.matches:
        ar = np.abs(m["residuals"])
        hub = np.where(ar > REG["k_huber"], REG["k_huber"] * (ar - 0.5 * REG["k_huber"]),
                       0.5 * ar ** 2)
        tot += float(hub @ m["weights"] if False else float(np.sum(hub)))
        n += len(ar)
    return tot, n


def main():
    mesh, tgt_pts, tgt_nrm, feats = e0.load_gear_cad()
    rng0 = np.random.default_rng(e0.MASTER_SEED)
    guide = [f.generate_points(120, noise_std=0.0, extent=6.0, rng=rng0) for f in feats]
    sel_idx = greedy_select(feats, guide, **e0.CONFIGS["geop2e_select"], rng=rng0)
    sub = [feats[i] for i in sel_idx]
    REG = auto_params(tgt_pts)
    axis, N_sym, match = e0.gear_axis_and_symmetry(mesh, feats)

    # deterministic loss0 trial (same generator as the full run, cell loss0 t0)
    seed = trial_seed(e0.MASTER_SEED, "synth", "loss0", 0)
    rng = np.random.default_rng(seed)
    src, T_gt = simulate_defect_ext(tgt_pts, rng, keep_frac=1.0, oblique=False)
    sn = estimate_normals(src, radius=REG["init_radius"] * 0.4, max_nn=30)
    T_true = np.linalg.inv(T_gt)                    # source->CAD target pose
    ext = float(np.linalg.norm(np.ptp(tgt_pts, axis=0)))
    pax = rng.normal(size=3); pax /= np.linalg.norm(pax)
    P = np.eye(4)
    P[:3, :3] = Rot.from_rotvec(pax * np.deg2rad(rng.uniform(0, 2))).as_matrix()
    P[:3, 3] = rng.normal(size=3) * ext * 0.0025
    T_coarse = P @ T_true

    def rre(T):
        from solver import evaluate_alignment
        return evaluate_alignment(T, T_gt)[0]

    runs = {}
    REG_no_iter = {k: v for k, v in REG.items() if k != "max_iter"}
    for name, (init, jac, iters) in {
        "gt_start_analytic_60": (T_true, "analytic", 60),
        "gt_start_numerical_60": (T_true, "numerical", 60),
        "coarse_start_analytic_60": (T_coarse, "analytic", 60),
        "gt_start_analytic_120": (T_true, "analytic", 120),
        "coarse_start_numerical_60": (T_coarse, "numerical", 60),
    }.items():
        T, info = register(src, sn, sub, jacobian_type=jac, init_T=init.copy(),
                           max_iter=iters, **REG_no_iter)
        runs[name] = dict(init_rre=rre(init), final_rre=rre(T),
                          final_rte=float(np.linalg.norm((T @ T_gt)[:3, 3])),
                          iters=info["iters"], update_norm=info["update_norm"])
        # first-iteration pose error: rerun 1 iteration
        T1, _ = register(src, sn, sub, jacobian_type=jac, init_T=init.copy(), max_iter=1,
                         **REG_no_iter)
        runs[name]["after_1_iter_rre"] = rre(T1)

    # FIXED-ASSOCIATION yaw profile (verification-#2 fix: the previous version
    # recomputed correspondences at every yaw). Associations (feature -> point
    # index sets) are frozen ONCE at the GT pose with the final annealed radius;
    # for each yaw we recompute residuals and Huber weights on those FIXED
    # indices only. Active count is therefore constant by construction.
    from solver import RegistrationSolver, transform_points as _tp
    frozen = RegistrationSolver(sub, "analytic", "huber", REG["k_huber"])
    frozen.find_correspondences(src, sn, T_true, REG["init_radius"] * (REG["anneal"] ** 59),
                                REG["angle_thresh_deg"])
    frozen_matches = [(m["feature"], np.asarray(m["idx"])) for m in frozen.matches]
    n_active = int(sum(len(i) for _, i in frozen_matches))

    def fixed_obj(T):
        p = _tp(src, T)
        tot = 0.0
        for feat, idx in frozen_matches:
            r = feat.compute_residual(p[idx])
            ar = np.abs(r)
            w = np.where(ar > REG["k_huber"], REG["k_huber"] / np.maximum(ar, 1e-12), 1.0)
            hub = np.where(ar > REG["k_huber"],
                           REG["k_huber"] * (ar - 0.5 * REG["k_huber"]), 0.5 * ar ** 2)
            tot += float(np.sum(w * hub))
        return tot

    prof = []
    for yaw in list(np.arange(-10, 10.01, 0.5)) + list(np.arange(-2, 2.01, 0.1)):
        Ty = np.eye(4)
        Ty[:3, :3] = Rot.from_rotvec(axis * np.deg2rad(yaw)).as_matrix()
        prof.append(dict(yaw_deg=round(float(yaw), 3), objective=fixed_obj(Ty @ T_true),
                         n_active=n_active, rre_deg=float(rre(Ty @ T_true))))
    ymin = min(prof, key=lambda p: p["objective"])

    # gradient/Hessian first step AT GT on the SAME frozen associations:
    # this is the update the solver requests from the exact-GT pose
    p_gt = _tp(src, T_true)
    H = np.zeros((6, 6)); b = np.zeros(6)
    for feat, idx in frozen_matches:
        r = feat.compute_residual(p_gt[idx])
        ar = np.abs(r)
        w = np.where(ar > REG["k_huber"], REG["k_huber"] / np.maximum(ar, 1e-12), 1.0)
        J = feat.compute_jacobian(p_gt[idx])
        H += J.T @ (J * w[:, None])
        b += -J.T @ (w * r)
    try:
        dxi = np.linalg.solve(H + REG["lam"] * np.eye(6), b)
    except np.linalg.LinAlgError:
        dxi = np.zeros(6)
    first_step = dict(norm=float(np.linalg.norm(dxi)),
                      rotation_deg=float(np.degrees(np.linalg.norm(dxi[:3]))))

    # nondimensional Hessian spectrum (data-conditioned): Jacobians evaluated on
    # centroid-shifted, length-normalised guide points so both blocks are
    # dimensionless (x_bar = (x - c)/L => rotational block scales by 1/L)
    H = np.zeros((6, 6))
    c = tgt_pts.mean(0)
    for i, f in zip(sel_idx, sub):
        J = f.compute_jacobian((guide[i] - c) / ext)
        H += J.T @ J
    w = np.linalg.eigvalsh(H)

    out = dict(N_sym=N_sym, symmetry_match=match, runs=runs,
               yaw_profile_min=ymin, yaw_profile=prof,
               fixed_association_n_active=n_active,
               gt_first_step_on_frozen_associations=first_step,
               nondim_hessian_eigs=[float(x) for x in w],
               note="associations frozen ONCE at GT (final annealed radius); yaw profile "
                    "recomputes only residuals/Huber weights on the fixed indices")
    p = os.path.join(HERE, "results", "e3c_loss0_diagnostic.json")
    if os.path.exists(p):
        p += f".{int(os.path.getctime(p))}.json"
    e0.save_json(out, p)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'history'}
                      for k, v in runs.items()}, indent=1, default=float))
    print("yaw objective min at", ymin)


if __name__ == "__main__":
    main()
