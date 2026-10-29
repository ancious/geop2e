"""Observability-driven feature selection (paper Sec. 3.2, Algorithm 1).

Greedy selection over candidate primitives by a swappable scalar criterion on
the Fisher information matrix  H_S = sum_k J_k^T J_k  (computed from each
primitive's discretized guiding point set). Criteria:
  - 'e'     : E-optimality  -> maximize  lambda_min(H_S)        [paper's choice]
  - 'd'     : D-optimality  -> maximize  log det(H_S + eps I)
  - 'a'     : A-optimality  -> maximize -trace(inv(H_S + eps I))
  - 'trace' : information-amount -> maximize trace(H_S)  (non-worst-case baseline;
             analogous in spirit to surface-area-based selection of prominent features)
  - 'area'  : prominence-based -> pick the target_size features with the largest
             individual spatial extent (bounding-box volume of the guiding set);
             a non-observability "surface-area / geometric prominence" baseline
  - 'cond'  : minimize condition number  kappa = lambda_max/lambda_min
  - 'random': random subset of target size
  - 'all'   : keep every candidate (no selection)

All primitives use the SAME analytical Jacobian J_i = [(x_i x n_i), n_i] from
features.py, so selection and the LM solver operate on the identical Hessian
(mechanism-level novelty hook in REBUTTAL_STRATEGY.md).
"""
from __future__ import annotations
from typing import List, Sequence
import numpy as np

from features import GeometricFeature


def primitive_info(feature: GeometricFeature, guide_pts: np.ndarray) -> np.ndarray:
    """H_k = J^T J for one primitive over its guiding points (6x6)."""
    J = feature.compute_jacobian(guide_pts)          # (N,6) analytical
    return J.T @ J


def _eig_sorted(H):
    w = np.linalg.eigvalsh(H)
    return np.sort(w)                                  # ascending


def _criterion(H, kind, eps=1e-9):
    """Return a scalar to MAXIMIZE (cond/random handled by caller)."""
    if kind == "e":
        return _eig_sorted(H)[0]                       # lambda_min
    if kind == "d":
        sg = np.linalg.svd(H + eps * np.eye(6), compute_uv=False)
        return float(np.sum(np.log(np.maximum(sg, 1e-12))))
    if kind == "a":
        try:
            return -float(np.trace(np.linalg.inv(H + eps * np.eye(6))))
        except np.linalg.LinAlgError:
            return -1e18
    if kind == "trace":
        return float(np.trace(H))                       # maximize total information
    if kind == "cond":
        w = _eig_sorted(H); wmin = max(w[0], 1e-12)
        return -w[-1] / wmin                           # maximize = minimize kappa
    raise ValueError(kind)


def greedy_select(features: Sequence[GeometricFeature],
                  guide_sets: Sequence[np.ndarray],
                  criterion: str = "e",
                  target_size: int | None = None,
                  min_features: int = 3,
                  rng: np.random.Generator | None = None) -> List[int]:
    """Greedy selection. Returns selected indices. Stops at target_size or when
    lambda_min stops improving (E-opt) / when all candidates exhausted."""
    n = len(features)
    if criterion == "all":
        return list(range(n))
    if criterion == "random":
        rng = rng or np.random.default_rng()
        k = target_size or max(min_features, int(np.ceil(np.sqrt(n))))
        return list(rng.choice(n, size=min(k, n), replace=False))
    if criterion == "area":
        # surface-area proxy: 2D convex-hull area of each guiding set projected
        # onto its 2 principal axes (planes -> large area, the intended behaviour
        # for "select large/prominent features"; may pick parallel planes -> degenerate)
        from scipy.spatial import ConvexHull
        def hull2d_area(pts):
            pts = pts - pts.mean(0)
            try:
                _, _, Vt = np.linalg.svd(pts, full_matrices=False)
                pts2d = pts @ Vt[:2].T
                return float(ConvexHull(pts2d).volume)   # .volume == area in 2D
            except Exception:
                return 0.0
        areas = [hull2d_area(g) for g in guide_sets]
        order = sorted(range(n), key=lambda i: areas[i], reverse=True)
        k = target_size or max(min_features, 6)
        return order[:min(k, n)]
    if target_size is None:
        target_size = n

    Hk = [primitive_info(f, g) for f, g in zip(features, guide_sets)]
    chosen: List[int] = []
    H = np.zeros((6, 6))
    remaining = set(range(n))
    prev_score = -np.inf
    while remaining and len(chosen) < target_size:
        best_i, best_score = None, -np.inf
        for i in remaining:
            sc = _criterion(H + Hk[i], criterion)
            if sc > best_score:
                best_score, best_i = sc, i
        # E-opt early stop: no lambda_min improvement
        if criterion == "e" and chosen and best_score <= prev_score + 1e-9 and len(chosen) >= min_features:
            break
        chosen.append(best_i)
        H = H + Hk[best_i]
        prev_score = best_score
        remaining.discard(best_i)
    return chosen


def spectral(H, eps=1e-12):
    w = _eig_sorted(H + eps * np.eye(6))
    return {"lambda_min": float(w[0]), "lambda_max": float(w[-1]),
            "kappa": float(w[-1] / max(w[0], eps))}


def selected_info(features, guide_sets, idx):
    H = np.zeros((6, 6))
    for i in idx:
        H += primitive_info(features[i], guide_sets[i])
    return H
