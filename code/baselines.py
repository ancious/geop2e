"""Classical point-cloud baselines (pure numpy/scipy, open3d-free).

- Point-to-point ICP (Kabsch/SVD) — ported from the notebook's CustomICPSolver.
- Point-to-plane ICP (linearized, target normals) — standard numpy impl.

These operate on DISCRETE point clouds (point<->point), i.e. a different input
modality from GeoP2E (point<->equation). Per REBUTTAL_STRATEGY.md Reframe B they
characterize the *value of exploiting CAD equations*, not a like-for-like contest.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot


def _apply(points, T):
    return points @ T[:3, :3].T + T[:3, 3]


def icp_point_to_point(source, target, init=None, threshold=2.0, max_iter=50, tol=1e-6):
    T = np.eye(4) if init is None else init.copy()
    cur = _apply(source, T)
    tree = cKDTree(target)
    prev = np.inf
    for _ in range(max_iter):
        d, idx = tree.query(cur, k=1)
        mask = d < threshold
        if mask.sum() < 10:
            break
        P, Q = cur[mask], target[idx[mask]]
        cP, cQ = P.mean(0), Q.mean(0)
        U, S, Vt = np.linalg.svd((P - cP).T @ (Q - cQ))
        Rm = Vt.T @ U.T
        if np.linalg.det(Rm) < 0:
            Vt[2] *= -1; Rm = Vt.T @ U.T
        t = cQ - Rm @ cP
        step = np.eye(4); step[:3, :3] = Rm; step[:3, 3] = t
        T = step @ T
        cur = _apply(source, T)
        err = np.mean(d[mask] ** 2)
        if abs(prev - err) < tol:
            break
        prev = err
    return T, prev


def icp_point_to_plane(source, target, target_normals, init=None, threshold=2.0, max_iter=50, tol=1e-6):
    """Linearized point-to-plane ICP: min sum ( n_j . (R p_i + t - q_j) )^2.
    Solves the 6x6 linear system A^T A x = A^T b with the point-to-plane Jacobian."""
    T = np.eye(4) if init is None else init.copy()
    cur = _apply(source, T)
    tree = cKDTree(target)
    prev = np.inf
    for _ in range(max_iter):
        d, idx = tree.query(cur, k=1)
        mask = d < threshold
        if mask.sum() < 10:
            break
        p = cur[mask]; q = target[idx[mask]]; n = target_normals[idx[mask]]
        # cross product matrix per point -> A_i = [ p_i x n_i , n_i ] (1x6)
        cr = np.cross(p, n)
        A = np.hstack([cr, n])                 # (M,6)
        b = -np.einsum('ij,ij->i', n, p - q)   # (M,)
        # optional Cauchy/huber omitted for a plain p2plane baseline
        H = A.T @ A + 1e-9 * np.eye(6)
        g = A.T @ b
        dxi = np.linalg.solve(H, g)
        step = np.eye(4); step[:3, :3] = Rot.from_rotvec(dxi[:3]).as_matrix(); step[:3, 3] = dxi[3:]
        T = step @ T
        cur = _apply(source, T)
        err = np.mean(d[mask] ** 2)
        if abs(prev - err) < tol:
            break
        prev = err
    return T, prev


def trimmed_icp(source, target, init=None, threshold=2.0, trim_frac=0.3, max_iter=50, tol=1e-6):
    """Trimmed ICP (Chetverikov et al. 2002, ref [71]) — point-to-point ICP that
    discards the worst trim_frac of correspondences each iteration. Designed for
    partial overlap / missing data; the canonical robust baseline for the
    defect-robustness claim. Pure numpy."""
    T = np.eye(4) if init is None else init.copy()
    cur = _apply(source, T)
    tree = cKDTree(target)
    prev = np.inf
    for _ in range(max_iter):
        d, idx = tree.query(cur, k=1)
        cand = d < threshold
        if cand.sum() < 10:
            break
        dc, ic, cc = d[cand], idx[cand], cur[cand]
        keep = int(round((1.0 - trim_frac) * len(dc)))
        keep = max(keep, 10)
        order = np.argsort(dc)[:keep]            # trim the worst residuals
        P, Q = cc[order], target[ic[order]]
        cP, cQ = P.mean(0), Q.mean(0)
        U, S, Vt = np.linalg.svd((P - cP).T @ (Q - cQ))
        Rm = Vt.T @ U.T
        if np.linalg.det(Rm) < 0:
            Vt[2] *= -1; Rm = Vt.T @ U.T
        t = cQ - Rm @ cP
        step = np.eye(4); step[:3, :3] = Rm; step[:3, 3] = t
        T = step @ T
        cur = _apply(source, T)
        err = np.mean(dc[order] ** 2)
        if abs(prev - err) < tol:
            break
        prev = err
    return T, prev
