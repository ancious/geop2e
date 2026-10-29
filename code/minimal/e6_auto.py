"""e6_auto — GeoP2E-auto: failure-informed automated mechanisms (GPT-co-designed 2026-08-27).

Fixes map to diagnosed failures:
  A1 adaptive association (E3c coarse-radius bias): per-point inlier posterior from a
     two-component mixture on TRIMMED nearest-support distances; softmax-normalised
     per-point support weights (sum_j w_ij <= 1); radius schedule bounded (<=30%
     shrink/iter, floor 2*k_Huber); dimensionally-correct trust region.
  A2 always-on candidates (dead AABB trigger): no trigger; full-CAD + slab cuts all
     evaluated; winner by HELD-OUT spatial-block score; cut wins only if its
     improvement exceeds one spatial-block bootstrap SE.
  A3 uncertainty-aware selection (kappa=2203 non-alarm): normalised FIM
     (J_bar = J*diag(L*I3,I)/sigma_hat); candidates E-opt k in {6,8,12,16,24,32}+all;
     sandwich covariance -> U = max(T95/T_req, R95/R_req); smallest candidate with
     U<=1 and validation score within 1 SE of best; else best + ALARM.
  A4 weak-mode multi-start (basin dependence): 5 starts = canonical FGR +- a1 v1
     +- a2 v2 (v = two weakest normalised-FIM rotation modes, magnitude from the
     data scale, caps 15deg / 0.1D); fit on 80% spatial blocks, select on 20%,
     prefer canonical within 1 SE; polish on all.

Truth discipline: nothing here sees GT. S1-S3 are RETROSPECTIVE repair cases only.
"""
from __future__ import annotations
import os, sys, hashlib
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, PARENT)
from features import PlaneFeature, CylinderFeature, SphereFeature  # noqa: E402
from selection import greedy_select  # noqa: E402

T3D = os.path.expanduser("~/projects/new/3dphy")
RREQ_DEG = 5.0          # declared accuracy requirement: rotation (from success rule)
TREQ_FRAC = 0.02        # translation requirement as fraction of diagonal D


# ================================================================ parsed parts
def load_parsed_part(step_path, merge_tol_frac=1e-3):
    """STEP -> merged analytic supports. Returns dict or None if ineligible.
    Faces restricted to plane/cylinder/sphere; coincident supports merged at
    STEP modeling tolerance (merge_tol_frac * diagonal)."""
    sys.path.insert(0, os.path.join(T3D, "src"))
    if "torch" not in sys.modules:                       # realpart_brep has an UNUSED torch import
        try:
            import torch  # noqa
        except ImportError:
            import types
            _t = types.ModuleType("torch")
            _t.__getattr__ = lambda name: None
            sys.modules["torch"] = _t
    from pathlib import Path as _P
    from tdphy.io.realpart_brep import load_brep_part, sample_exact_surface
    part = load_brep_part(_P(step_path))
    # typ_name is abbreviated in realpart_brep ('Pln'/'Plane', 'Cyl'/'Cylinder', 'Sph'/'Sphere')
    OK = ("Plane", "Pln", "Cylinder", "Cyl", "Sphere", "Sph")
    keep = [f for f in part.faces if f.typ_name in OK]
    if len(keep) < 3:
        return None
    D = part.L
    tol = merge_tol_frac * D
    feats, samples = [], []
    for f in keep:
        p = f.params
        t = f.typ_name.lower()
        if t.startswith("pl"):
            n = p["axis"] / np.linalg.norm(p["axis"])
            feats.append(PlaneFeature(n, -float(n @ p["origin"])))
        elif t.startswith("cy"):
            feats.append(CylinderFeature(p["origin"], p["axis"], float(p["radius"])))
        elif t.startswith("sp"):
            feats.append(SphereFeature(p["origin"], float(p["radius"])))
        else:
            feats.append(None)
        samples.append((f, None))
    pairs = [(f, ft) for f, ft in zip(keep, feats) if ft is not None]
    keep = [x[0] for x in pairs]; feats = [x[1] for x in pairs]
    # exact samples for the WHOLE supported set, then split by prim_id
    # (prim_id indexes part.faces — map GLOBAL face idx -> keep-local idx)
    P, N, prim_id, _ = sample_exact_surface(part, 20000, seed=0)
    g2l = {}
    for li, f in enumerate(part.faces):
        if f in keep:
            g2l[li] = keep.index(f)
    sup_pts = {i: [] for i in range(len(keep))}
    for i, fi in enumerate(prim_id):
        if fi in g2l:
            sup_pts[g2l[fi]].append(P[i])
    sup = []
    for i, f in enumerate(keep):
        pts = np.asarray(sup_pts[i]) if sup_pts[i] else np.zeros((0, 3))
        if len(pts) >= 8:
            sup.append(dict(face=f, feat=feats[i], pts=pts, area=f.area))
    if len(sup) < 3:
        return None
    # merge coincident supports (same primitive within tol)
    merged = []
    for s in sup:
        target = None
        for m in merged:
            if _same_prim(m["feats"][0], s["feat"], tol):
                target = m; break
        if target is None:
            merged.append(dict(feats=[s["feat"]], pts=[s["pts"]], area=s["area"]))
        else:
            target["pts"].append(s["pts"]); target["area"] += s["area"]; target["feats"].append(s["feat"])
    for m in merged:
        m["pts"] = np.vstack(m["pts"])
        m["feat"] = m["feats"][0]
        m["tree"] = cKDTree(m["pts"])
    total_area = sum(m["area"] for m in merged)
    # coverage: merged analytic area / ALL face area (incl. unsupported types)
    all_area = sum(f.area for f in part.faces)
    return dict(name=os.path.basename(str(step_path)), D=D, supports=merged,
                analytic_coverage=total_area / all_area if all_area else 0.0,
                n_faces_total=len(part.faces))


def _softmax_rows(cand, Ra, s_t):
    """Softmax over gated supports; rows with NO candidate get all-zero weights."""
    lg = np.where(cand, -Ra ** 2 / (2 * s_t ** 2), -np.inf)
    mx = lg.max(axis=1, keepdims=True)
    bad = ~np.isfinite(mx[:, 0])
    lg = np.where(bad[:, None], 0.0, lg - np.where(bad[:, None], 0.0, mx))
    w = np.exp(lg)
    w[bad] = 0.0
    s = w.sum(axis=1, keepdims=True)
    w = np.where(s > 1e-12, w / np.maximum(s, 1e-12), 0.0)
    return w


def _same_prim(a, b, tol):
    if a.kind != b.kind:
        return False
    if a.kind == "plane":
        return abs(abs(float(a.normal @ b.normal)) - 1) < 1e-6 and abs(a.d - b.d) < tol
    if a.kind == "cylinder":
        return (abs(abs(float(a.axis @ b.axis)) - 1) < 1e-6
                and np.linalg.norm((a.center - b.center) - (a.center - b.center) @ a.axis[:, None] @ a.axis[None, :]) < tol
                and abs(a.radius - b.radius) < tol)
    return np.linalg.norm(a.center - b.center) < tol and abs(a.radius - b.radius) < tol


def support_cloud(part):
    pts = np.vstack([m["pts"] for m in part["supports"]])
    lab = np.concatenate([np.full(len(m["pts"]), j) for j, m in enumerate(part["supports"])])
    return pts, lab


# ================================================================ trimmed geometry
class TrimmedCAD:
    """Nearest-support-sample association = trimmed distance + face label."""

    def __init__(self, part):
        self.part = part
        self.pts, self.lab = support_cloud(part)
        self.tree = cKDTree(self.pts)
        self.feats = [m["feat"] for m in part["supports"]]
        self.D = part["D"]

    def d_and_label(self, src_T):
        d, idx = self.tree.query(src_T, k=1)
        return d, self.lab[idx]

    def analytic_residuals(self, src_T):
        """R matrix (N x S) of analytic |r_ij| (S = supports, small)."""
        R = np.empty((len(src_T), len(self.feats)))
        for j, f in enumerate(self.feats):
            R[:, j] = np.abs(f.compute_residual(src_T))
        return R


# ================================================================ A1 solver
def em_inlier_posterior(d, D, iters=25):
    """pi*HalfNormal(sigma) + (1-pi)*U(0, 0.25D) on trimmed distances -> gamma_i."""
    d = np.maximum(d, 1e-12)
    umax = 0.25 * D
    d = np.minimum(d, umax)
    pi, sig = 0.6, max(np.median(d), 1e-9)
    for _ in range(iters):
        # E: responsibilities
        ll_in = np.log(pi + 1e-12) - np.log(max(sig, 1e-12)) - 0.5 * (d / max(sig, 1e-12)) ** 2 - 0.5 * np.log(2 * np.pi)
        ll_out = np.log(1 - pi + 1e-12) - np.log(umax)
        g = 1.0 / (1.0 + np.exp(np.clip(ll_out - ll_in, -500, 500)))
        n = g.sum()
        if n < 1e-9:
            break
        pi = min(max(n / len(d), 1e-3), 1 - 1e-3)
        sig = max(np.sqrt((g * d ** 2).sum() / n), 1e-12)
    return g, sig, pi


def robust_score(d, k_H):
    """S = mean rho_Huber(d), unmatched handled by trimmed distance already."""
    a = np.abs(d)
    return float(np.mean(np.where(a > k_H, k_H * (a - 0.5 * k_H), 0.5 * a ** 2)))


def register_auto(tcad, src, sn, k_H, init_T, max_iter=60, verbose=False):
    """A1: responsibility-weighted point-to-equation LM with bounded radius schedule
    and L-scaled trust region. Returns (T, info). No GT anywhere."""
    D = tcad.D
    T = init_T.copy()
    R_t = None
    Delta = None
    accepted = 0
    hist = []
    L = float(np.linalg.norm(np.ptp(src, axis=0)))  # initial lever scale
    for it in range(max_iter):
        p = src @ T[:3, :3].T + T[:3, 3]
        d, _ = tcad.d_and_label(p)
        if R_t is None:
            R_t = min(0.25 * D, max(2 * k_H, float(np.quantile(d, 0.95))))
            Delta = R_t / 2
        g, sig, pi = em_inlier_posterior(d, D)
        if g.sum() / len(d) < 0.30:
            hist.append(dict(it=it, event="hold_low_inlier")); break
        Ra = tcad.analytic_residuals(p)
        s_t = R_t / 2.58
        cand = Ra < R_t
        # softmax weights over gated supports
        w = _softmax_rows(cand, Ra, s_t)
        w *= g[:, None]                     # sum_j w_ij <= gamma_i <= 1
        # weighted normal equations (rows only where cand)
        H = np.zeros((6, 6)); b = np.zeros(6)
        wsum2 = 0.0; lever_num = 0.0; w_tot = 0.0
        c = p.mean(0)
        for j, f in enumerate(tcad.feats):
            idx = np.where(cand[:, j] & (w[:, j] > 1e-9))[0]
            if len(idx) < 4:
                continue
            wj = w[idx, j]
            rj = f.compute_residual(p[idx])
            Jj = f.compute_jacobian(p[idx])
            H += (Jj * wj[:, None]).T @ Jj
            b += -(Jj * wj[:, None]).T @ rj
            wsum2 += float((wj ** 2).sum())
            lever_num += float((wj * ((p[idx] - c) ** 2).sum(1)).sum())
            w_tot += float(wj.sum())
        if np.linalg.matrix_rank(H) < 6:
            hist.append(dict(it=it, event="hold_rank")); break
        L = float(np.sqrt(lever_num / max(w_tot, 1e-12)))
        # L-scaled solve with trust region ||dxi||_L <= Delta
        S = np.diag([L, L, L, 1.0, 1.0, 1.0])
        Si = np.diag([1 / L, 1 / L, 1 / L, 1.0, 1.0, 1.0])
        Hs = Si @ H @ Si
        try:
            dxi_s = np.linalg.solve(Hs + 1e-9 * np.trace(Hs) / 6 * np.eye(6), Si @ b)
        except np.linalg.LinAlgError:
            break
        nrm = float(np.linalg.norm(S @ dxi_s))
        if nrm > Delta:
            dxi_s *= Delta / nrm
        dxi = Si @ dxi_s
        T_new = _exp_se3(dxi) @ T
        # gain ratio on the trimmed robust score
        S_old = robust_score(d, k_H)
        p_new = src @ T_new[:3, :3].T + T_new[:3, 3]
        d_new, _ = tcad.d_and_label(p_new)
        S_new = robust_score(d_new, k_H)
        pred = float(0.5 * dxi @ (H @ dxi))          # predicted decrease (>=0 at LM step)
        actual = S_old - S_new
        rho = actual / pred if pred > 1e-15 else (1.0 if actual >= 0 else -1.0)
        if rho < 0.25:
            Delta *= 0.5
            if rho < 0:
                hist.append(dict(it=it, event="reject", rho=rho))
                continue
        elif rho > 0.75:
            Delta = min(2 * Delta, 0.25 * D)
        T = T_new
        accepted += 1
        R_next = min(R_t, max(0.7 * R_t, 2.58 * sig, 2 * k_H))
        hist.append(dict(it=it, R=R_t, rho=rho, inlier=float(g.mean()), sigma=float(sig)))
        R_t = R_next
        if float(np.linalg.norm(S @ dxi)) < 1e-6:
            break
    # final polish at 2*k_H with same machinery (fixed radius)
    for it in range(15):
        p = src @ T[:3, :3].T + T[:3, 3]
        d, _ = tcad.d_and_label(p)
        g, sig, pi = em_inlier_posterior(d, D)
        Ra = tcad.analytic_residuals(p)
        R_t2 = 2 * k_H
        cand = Ra < R_t2
        w = _softmax_rows(cand, Ra, R_t2 / 2.58); w *= g[:, None]
        H = np.zeros((6, 6)); b = np.zeros(6)
        for j, f in enumerate(tcad.feats):
            idx = np.where(cand[:, j] & (w[:, j] > 1e-9))[0]
            if len(idx) < 4:
                continue
            wj = w[idx, j]
            H += (f.compute_jacobian(p[idx]) * wj[:, None]).T @ f.compute_jacobian(p[idx])
            b += -(f.compute_jacobian(p[idx]) * wj[:, None]).T @ f.compute_residual(p[idx])
        if np.linalg.matrix_rank(H) < 6:
            break
        try:
            dxi = np.linalg.solve(H + 1e-9 * np.trace(H) / 6 * np.eye(6), b)
        except np.linalg.LinAlgError:
            break
        if np.linalg.norm(dxi) < 1e-7:
            break
        T = _exp_se3(dxi) @ T
    return T, dict(accepted=accepted, hist=hist, final_sigma=sig)


def _exp_se3(xi):
    w, v = xi[:3], xi[3:]
    th = float(np.linalg.norm(w))
    R = Rot.from_rotvec(w).as_matrix()
    if th < 1e-8:
        V = np.eye(3)
    else:
        K = _skew(w)
        V = np.eye(3) + ((1 - np.cos(th)) / th ** 2) * K + ((th - np.sin(th)) / th ** 3) * (K @ K)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = V @ v
    return T


def _skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


# ================================================================ A3 selection
def normalized_fim(feats, guide_pts, sigma):
    """H_bar = sum_k J_bar^T J_bar with J_bar = J*diag(L*I3,I)/sigma (per support)."""
    allp = np.vstack(guide_pts)
    c = allp.mean(0)
    L = float(np.linalg.norm(np.ptp(allp, axis=0)))
    S = np.diag([L, L, L, 1.0, 1.0, 1.0]) / max(sigma, 1e-12)
    H = np.zeros((6, 6))
    for f, g in zip(feats, guide_pts):
        J = f.compute_jacobian(g) @ S
        H += J.T @ J
    return H, L


def select_auto(part, sigma, val_score_fn):
    """Candidates = E-opt greedy k in {6,8,12,16,24,32} + all-merged; uncertainty U from
    sandwich covariance; pick smallest with U<=1 and val score within 1 SE of best."""
    sup = part["supports"]
    rng = np.random.default_rng(0)
    guides = [m["pts"][rng.choice(len(m["pts"]), min(80, len(m["pts"])), replace=False)]
              for m in sup]
    cand_sets = []
    n = len(sup)
    for k in (6, 8, 12, 16, 24, 32):
        if k < n:
            sel = greedy_select([m["feat"] for m in sup], guides, criterion="e",
                                target_size=k, min_features=min(5, n), rng=rng)
            if len(sel) >= 4:
                cand_sets.append((f"E{k}", sorted(sel)))
    cand_sets.append(("all", list(range(n))))
    infos = []
    for name, idx in cand_sets:
        feats = [sup[i]["feat"] for i in idx]
        sub_guides = [guides[i] for i in idx]
        H, L = normalized_fim(feats, sub_guides, sigma)
        w = np.linalg.eigvalsh(H)
        kappa = float(w[-1] / max(w[0], 1e-12))
        # sandwich covariance (Huber): H^-1 B H^-1 with B = sum J_bar J_bar^T
        B = np.zeros((6, 6))
        for f, g in zip(feats, sub_guides):
            S = np.diag([L, L, L, 1.0, 1.0, 1.0]) / max(sigma, 1e-12)
            J = f.compute_jacobian(g) @ S
            B += J.T @ J / max(len(g), 1)
        try:
            C = np.linalg.pinv(H) @ B @ np.linalg.pinv(H) / max(sum(len(g) for g in sub_guides), 1)
            var = np.diag(C)
            R95 = float(np.degrees(1.96 * np.sqrt(max(var[:3].sum(), 0))))
            T95 = float(1.96 * np.sqrt(max(var[3:].sum(), 0)))
        except Exception:
            R95, T95 = np.inf, np.inf
        U = max(R95 / RREQ_DEG, T95 / TREQ_FRAC)
        alarm = (np.linalg.matrix_rank(H) < 6) or (U > 1.0)
        infos.append(dict(name=name, n=len(idx), kappa=kappa, lam_min_norm=float(w[0]),
                          R95_deg=R95, T95_frac=T95, U=float(U), alarm=bool(alarm), idx=idx))
    # validation scores (one SE rule) — score fn returns (score, se)
    scored = []
    for inf in infos:
        s, se = val_score_fn(inf["idx"])
        inf["val_score"], inf["val_se"] = s, se
        scored.append(inf)
    best = min(scored, key=lambda x: x["val_score"])
    thr = best["val_score"] + (best["val_se"] or 1e-9)
    pick = None
    for inf in scored:            # smallest set (largest first by size ascending)
        if inf["U"] <= 1.0 and inf["val_score"] <= thr:
            pick = inf; break
    if pick is None:
        pick = best
        pick["alarm"] = True
    return pick, scored


# ================================================================ A4 multi-start
def spatial_blocks(src, n_blocks=8, seed=0):
    """Fixed spatial partition (KMeans-lite: random anchor assignment by proximity)."""
    rng = np.random.default_rng(seed)
    anchors = src[rng.choice(len(src), n_blocks, replace=False)]
    d2 = ((src[:, None, :] - anchors[None, :, :]) ** 2).sum(-1)
    return np.argmin(d2, axis=1)


def block_score(tcad, src, T, k_H, blocks, held):
    m = np.isin(blocks, held)
    p = src[m] @ T[:3, :3].T + T[:3, 3]
    d, _ = tcad.d_and_label(p)
    return robust_score(d, k_H)


def multistart_auto(tcad, src, sn, k_H, T_canon, sel_feats_idx=None, n_starts=5):
    """5 starts: canonical +- a1 v1 +- a2 v2 along the two weakest normalised-FIM
    rotation modes. Fit on 80% spatial blocks, select on held 20%, prefer canonical
    within 1 SE. Returns (T, info)."""
    D = tcad.D
    blocks = spatial_blocks(src, n_blocks=8)
    fitb = list(range(0, 6)); heldb = list(range(6, 8))
    # weak modes from a quick solve at canonical
    T0, info0 = register_auto(tcad, src, sn, k_H, T_canon)
    p = src @ T0[:3, :3].T + T0[:3, 3]
    d, _ = tcad.d_and_label(p)
    g, sig, _ = em_inlier_posterior(d, D)
    Ra = tcad.analytic_residuals(p)
    R_t = 2 * k_H
    cand = Ra < R_t
    w = _softmax_rows(cand, Ra, R_t / 2.58); w *= g[:, None]
    H = np.zeros((6, 6)); lever = 0.0; wt = 0.0
    c = p.mean(0)
    for j, f in enumerate(tcad.feats):
        idx = np.where(cand[:, j] & (w[:, j] > 1e-9))[0]
        if len(idx) < 4:
            continue
        J = f.compute_jacobian(p[idx])
        wj = w[idx, j]
        H += (J * wj[:, None]).T @ J
        lever += float((wj * ((p[idx] - c) ** 2).sum(1)).sum()); wt += float(wj.sum())
    L = float(np.sqrt(lever / max(wt, 1e-12)))
    Hs = np.diag([1 / L, 1 / L, 1 / L, 1, 1, 1]) @ H @ np.diag([1 / L, 1 / L, 1 / L, 1, 1, 1])
    evals, evecs = np.linalg.eigh(Hs)
    v1, v2 = evecs[:, 0], evecs[:, 1]          # two weakest modes (normalised frame)

    def step_along(v):
        # a = R0 / (2 * sqrt(mean((J v)^2))): magnitude placing residual gain ~ R0
        num = 2 * k_H
        den = float(np.sqrt(max(np.mean((np.diag([L, L, L, 1, 1, 1]) @ (H @ v)) ** 2), 1e-18)))
        a = num / max(den, 1e-12)
        a = min(a, np.deg2rad(15) / max(np.linalg.norm(v[:3]), 1e-9),
                0.10 * D / max(np.linalg.norm(v[3:]), 1e-9))
        dxi = np.zeros(6)
        s1 = np.linalg.norm(v[:3]); s2 = np.linalg.norm(v[3:])
        if s1 > 1e-9:
            dxi[:3] = a * v[:3] / s1 * np.linalg.norm(v) * 0.5
        if s2 > 1e-9:
            dxi[3:] = min(0.10 * D, a) * v[3:] / s2 * 0.5
        return _exp_se3(dxi)

    starts = [("canon", T0)]
    for nm, v in (("w1+", v1), ("w1-", -v1), ("w2+", v2), ("w2-", -v2))[:max(0, n_starts - 1)]:
        starts.append((nm, step_along(v) @ T0))
    results = []
    for nm, Ts in starts:
        Tfit, _ = register_auto(tcad, src, sn, k_H, Ts)
        s_fit = block_score(tcad, src, Tfit, k_H, blocks, fitb)
        s_held = block_score(tcad, src, Tfit, k_H, blocks, heldb)
        results.append(dict(name=nm, T=Tfit, s_fit=s_fit, s_held=s_held))
    se = float(np.std([r["s_held"] for r in results]) / max(np.sqrt(len(results)), 1) + 1e-12)
    best = min(results, key=lambda r: r["s_held"])
    canon = results[0]
    chosen = canon if canon["s_held"] <= best["s_held"] + se else best
    # polish chosen on ALL points
    T_final, _ = register_auto(tcad, src, sn, k_H, chosen["T"])
    return T_final, dict(starts=results, chosen=chosen["name"], se=se, info0=info0)


# ================================================================ A2 staged HV
def staged_hv_auto(tcad, src, sn, k_H, T_sel, max_cands=8, seed=0):
    """Always-on candidates: full-CAD + slab cuts of the support set; each polished
    at 2k_H; winner by HELD-OUT spatial-block score; a cut wins only if its
    improvement exceeds one spatial-block bootstrap SE. No AABB trigger."""
    from e3_hv_ablation import make_candidates_slab
    blocks = spatial_blocks(src, n_blocks=8, seed=1)
    held = [6, 7]
    feats = tcad.feats
    guides = [m["pts"] for m in tcad.part["supports"]]
    cands = make_candidates_slab(feats, guides, max_cands=max_cands)
    trial_sets = [("full", list(range(len(feats))))] + [(f"cand{i}", c) for i, c in enumerate(cands)]
    rng = np.random.default_rng(seed)

    def solve_subset(idx, T0):
        sub_tcad = TrimmedCAD_from_subset(tcad, idx)
        return register_auto(sub_tcad, src, sn, k_H, T0)[0]

    base_T = solve_subset(trial_sets[0][1], T_sel)
    base_score = block_score(tcad, src, base_T, k_H, blocks, held)
    rows = [dict(name="full", score=base_score)]
    best_name, best_score, best_T = "full", base_score, base_T
    # spatial-block bootstrap SE of the base score (per-block scores of ALL blocks)
    p_base = src @ base_T[:3, :3].T + base_T[:3, 3]
    d_base, _ = tcad.d_and_label(p_base)
    per_block = [float(robust_score(d_base[blocks == b], k_H)) for b in range(8)]
    se = float(np.std(per_block) / np.sqrt(8) + 1e-12)
    for nm, idx in trial_sets[1:]:
        T_c = solve_subset(idx, T_sel)
        s_c = block_score(tcad, src, T_c, k_H, blocks, held)
        rows.append(dict(name=nm, score=s_c))
        if base_score - s_c > se and s_c < best_score:
            best_name, best_score, best_T = nm, s_c, T_c
    return best_T, dict(winner=best_name, base=base_score, se=se, rows=rows)


def TrimmedCAD_from_subset(tcad, idx):
    """Sub-CAD restricted to support indices (trees rebuilt on subset clouds)."""
    import copy
    sub = copy.copy(tcad)
    sub.feats = [tcad.feats[i] for i in idx]
    pts, lab = [], []
    for j, i in enumerate(idx):
        P = tcad.part["supports"][i]["pts"]
        pts.append(P); lab.append(np.full(len(P), j))
    sub.pts = np.vstack(pts); sub.lab = np.concatenate(lab)
    sub.tree = cKDTree(sub.pts)
    sub.part = dict(tcad.part)
    sub.part["supports"] = [tcad.part["supports"][i] for i in idx]
    return sub
