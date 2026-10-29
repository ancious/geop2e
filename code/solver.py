"""GeoP2E registration solver — point-to-equation LM on se(3), open3d-free.

Ported from `RegistrationSolver` / `register_rihfr` in the project notebooks,
refactored to expose the ablation toggles:
  - jacobian_type : 'analytic' | 'numerical'   (reviewer R1#5 / R3 solver-level)
  - kernel        : 'huber' | 'l2'              (robust kernel ablation)

Pose update is the LEFT perturbation  T <- exp(dxi^) T  (matches UNIFIED_MATH.md
and the original code: solve_one_step returns Delta_T @ T_current).
"""
from __future__ import annotations
from typing import List, Sequence
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import cKDTree

from features import GeometricFeature


# ---------------------------------------------------------------------------
# Normal estimation (replaces open3d estimate_normals) — local PCA + MST-free
# consistent orientation toward the cloud centroid. Sufficient for the clean
# synthetic scans used in the small-data experiments.
# ---------------------------------------------------------------------------
def estimate_normals(points: np.ndarray, radius: float = 0.5, max_nn: int = 30) -> np.ndarray:
    tree = cKDTree(points)
    normals = np.zeros_like(points)
    centroid = points.mean(axis=0)
    for i, p in enumerate(points):
        idx = tree.query_ball_point(p, r=radius)
        if len(idx) < 4:
            idx = tree.query(p, k=min(max_nn, len(points)))[1]
            idx = np.atleast_1d(idx)
        nn = points[idx] - p
        cov = nn.T @ nn
        _, vecs = np.linalg.eigh(cov)
        n = vecs[:, 0]                                   # smallest eigenvec
        if np.dot(n, centroid - p) < 0:                  # orient toward centroid
            n = -n
        normals[i] = n
    return normals


def transform_points(points: np.ndarray, T: np.ndarray):
    return points @ T[:3, :3].T + T[:3, 3]


def _skew(w: np.ndarray) -> np.ndarray:
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Exact SE(3) exponential of a left twist xi=[w,v] (4x4). Uses the SO(3)
    left Jacobian V(w) for the translational part, so the retraction matches the
    paper's T <- exp(xi^) T (not an additive-v approximation)."""
    w, v = xi[:3], xi[3:]
    theta = float(np.linalg.norm(w))
    R = Rot.from_rotvec(w).as_matrix()
    if theta < 1e-8:
        V = np.eye(3)
    else:
        K = _skew(w)
        V = (np.eye(3) + ((1 - np.cos(theta)) / theta**2) * K
             + ((theta - np.sin(theta)) / theta**3) * (K @ K))
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = V @ v
    return T


# ---------------------------------------------------------------------------
class RegistrationSolver:
    def __init__(self, features: Sequence[GeometricFeature],
                 jacobian_type: str = "analytic", kernel: str = "huber",
                 k_huber: float = 0.5):
        self.features = features
        self.jacobian_type = jacobian_type
        self.kernel = kernel
        self.k_huber = k_huber
        self.matches: list = []

    # ---- correspondence (bilateral: residual radius + normal angle) ------
    def find_correspondences(self, points, normals, T, radius_thresh, angle_thresh_deg=30.0,
                             bounded_eps=None):
        """bounded_eps: optional finite-face boundary tolerance (mm). When set
        (and the feature carries a `bounds_tree` of its trimmed sample cloud),
        a match additionally requires the CAD-frame point to lie within
        bounded_eps of that cloud — the bounded-face correspondence test
        (Phase-1 profile switch; see manuscript Sec. 3). None = OFF (frozen)."""
        p = transform_points(points, T)
        n = normals @ T[:3, :3].T
        self.matches = []
        cos_t = np.cos(np.deg2rad(angle_thresh_deg))
        for feat in self.features:
            r = feat.compute_residual(p)
            dist_mask = np.abs(r) < radius_thresh
            if not np.any(dist_mask):
                continue
            idx = np.where(dist_mask)[0]
            fn = feat.compute_feature_normal(p[idx])
            dots = np.einsum('ij,ij->i', n[idx], fn)
            keep = idx[np.abs(dots) > cos_t]   # |cos| gate: robust to normal sign ambiguity
            if bounded_eps is not None and getattr(feat, "bounds_tree", None) is not None:
                # project onto the feature surface, then bound the TANGENTIAL
                # distance to the trimmed cloud (raw 3D distance entangles with
                # the residual capture radius; review round 2026-09-08)
                q = p[keep]
                if feat.kind == "plane":
                    proj = q - np.einsum('i,ij->ij', feat.compute_residual(q),
                                         np.tile(feat.normal, (len(q), 1)))
                elif feat.kind == "cylinder":
                    k = feat.axis / np.linalg.norm(feat.axis)
                    v = q - feat.center
                    along = np.einsum('ij,j->i', v, k)[:, None] * k
                    rad = v - along
                    rn = np.linalg.norm(rad, axis=1, keepdims=True)
                    proj = feat.center + along + rad / np.maximum(rn, 1e-9) * feat.radius
                else:  # sphere
                    v = q - feat.center
                    vn = np.linalg.norm(v, axis=1, keepdims=True)
                    proj = feat.center + v / np.maximum(vn, 1e-9) * feat.radius
                d_b, _ = feat.bounds_tree.query(proj, k=1, workers=1)
                keep = keep[d_b <= bounded_eps]
            if len(keep) > 0:
                self.matches.append({"feature": feat, "idx": keep, "residuals": r[keep]})

    # ---- robust weights ---------------------------------------------------
    def compute_weights(self):
        # Standard robust M-estimator weights (no per-feature count rescaling:
        # the original notebook's `weights /= len(res)` shrank H below the LM
        # damping and stalled convergence; removed as a correctness fix).
        for m in self.matches:
            ar = np.abs(m["residuals"])
            if self.kernel == "l2":
                m["weights"] = np.ones_like(ar)
            else:  # huber
                m["weights"] = np.where(ar > self.k_huber,
                                        self.k_huber / np.maximum(ar, 1e-12), 1.0)

    # ---- one LM step ------------------------------------------------------
    def solve_one_step(self, points, T, lam=1e-2):
        H = np.zeros((6, 6)); b = np.zeros(6)
        p = transform_points(points, T)
        total = 0
        for m in self.matches:
            idx = m["idx"]; w = m["weights"][:, None]; r = m["residuals"]
            J = (m["feature"].compute_jacobian(p[idx]) if self.jacobian_type == "analytic"
                 else m["feature"].compute_numerical_jacobian(p[idx]))
            H += J.T @ (J * w)
            b += -J.T @ (w.flatten() * r)
            total += len(idx)
        if total < 6:
            return T, 0.0
        H += lam * np.eye(6)
        try:
            dxi = np.linalg.solve(H, b)
        except np.linalg.LinAlgError:
            return T, 0.0
        return se3_exp(dxi) @ T, float(np.linalg.norm(dxi))   # exact left SE(3) retraction


def register(points, normals, features, *,
             jacobian_type="analytic", kernel="huber", k_huber=0.5,
             init_T=None, init_radius=2.0, anneal=0.92, max_iter=25,
             angle_thresh_deg=30.0, lam=1e-2, verbose=False,
             bounded_face=False):
    """Run coarse-to-fine point-to-equation LM. Returns (T, info)."""
    T = np.eye(4) if init_T is None else init_T.copy()
    radius = init_radius
    solver = RegistrationSolver(features, jacobian_type, kernel, k_huber)
    last_upd = 0.0
    history = []  # (iter, radius, residual_rmse, update_norm) for convergence curves (R1-5)
    for it in range(max_iter):
        b_eps = None if not bounded_face else max(2.0, 0.25 * radius)   # tangential tolerance
        solver.find_correspondences(points, normals, T, radius, angle_thresh_deg,
                                    bounded_eps=b_eps)
        solver.compute_weights()
        T, upd = solver.solve_one_step(points, T, lam)
        last_upd = upd
        rmse = _rmse(solver)
        history.append({"iter": it, "radius": float(radius), "rmse": float(rmse), "update": float(upd)})
        if verbose:
            print(f"  it{it:02d} r={radius:7.4f} upd={upd:9.2e} rmse={rmse:9.3e}")
        radius *= anneal
        if upd < 1e-6:
            break
    return T, {"update_norm": last_upd, "iters": it + 1, "history": history}


def _rmse(solver: RegistrationSolver) -> float:
    tot_sq, n = 0.0, 0
    for m in solver.matches:
        tot_sq += float(np.sum(m["residuals"] ** 2)); n += len(m["residuals"])
    return float(np.sqrt(tot_sq / n)) if n > 0 else 0.0


# ---------------------------------------------------------------------------
def evaluate_alignment(T_pred: np.ndarray, T_perturb: np.ndarray):
    """T_pred @ T_perturb should be identity => rot geodesic (deg) + trans L2."""
    C = T_pred @ T_perturb
    tr = np.clip((np.trace(C[:3, :3]) - 1) / 2, -1.0, 1.0)
    rot_err = float(np.arccos(tr) * 180.0 / np.pi)
    trans_err = float(np.linalg.norm(C[:3, 3]))
    return rot_err, trans_err
