"""Hypothesis-verification module (paper Sec. 3.4) — numpy port of volume.ipynb.

Given a DEFECTIVE source (large missing region) the local GeoP2E solver drifts
(centroid bias / wrong basin). This module generates candidate sub-CAD hypotheses
(virtual cutting: spatial crops of the feature set), runs the GeoP2E solver on
each, and picks the best by robust residual (+ optional solid-penetration penalty).

Toggles for the R3-6 ablation:
  - volume_trigger : only activate hypothesis when V_S/V_M < tau_vol
  - cutting        : generate candidate sub-CADs (else: single full-CAD hypothesis)
  - penetration    : add solid-penetration penalty to the score
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def _transform(points, T):
    return points @ T[:3, :3].T + T[:3, 3]


def bbox_volume(points):
    return float(np.prod(np.ptp(points, axis=0)))


def make_candidates(features, guide_sets, target_pts, target_ratio, step_frac=0.25, max_cands=12):
    """Sliding-window crop of the CAD (by guiding-point location) -> candidate
    feature subsets. target_ratio ~ expected remaining volume fraction."""
    lo, hi = target_pts.min(0), target_pts.max(0)
    extent = hi - lo
    win = extent * max(0.3, target_ratio) ** (1 / 3)        # cubic-root volume->linear
    step = extent * step_frac
    cands = []
    # also include principal-axis-aligned mid cuts
    centers = [0.5 * (lo + hi)]
    for ax in range(3):
        for frac in (0.25, 0.5, 0.75):
            c = lo.copy(); c[ax] = lo[ax] + frac * extent[ax]; centers.append(c)
    for cx in centers[:max_cands]:
        wlo, whi = cx - win / 2, cx + win / 2
        keep = []
        for i, g in enumerate(guide_sets):
            mid = g.mean(0)
            if np.all(mid > wlo) and np.all(mid < whi):
                keep.append(i)
        if len(keep) >= 4:
            cands.append(keep)
    # dedup + cap
    uniq = []
    for c in cands:
        if c not in uniq:
            uniq.append(c)
    return uniq[:max_cands]


def penetration_fraction(source_pts_T, target_tree_target_pts, threshold):
    """Fraction of source points (after transform) whose nearest target distance
    exceeds threshold (a cheap proxy for solid penetration / bad overlap)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(target_tree_target_pts)
    d, _ = tree.query(source_pts_T, k=1)
    return float(np.mean(d > threshold))


def verify(features, guide_sets, target_pts, source_pts, source_normals,
           register_fn, target_ratio, *,
           volume_trigger=True, cutting=True, penetration=True,
           tau_vol=0.85, pen_threshold=None, verbose=False):
    """Run GeoP2E with hypothesis verification. register_fn(points,normals,features)->(T,info).
    Returns (best_T, info, meta)."""
    vs, vm = bbox_volume(source_pts), bbox_volume(target_pts)
    ratio = vs / vm
    triggered = (ratio < tau_vol) if volume_trigger else True
    meta = {"V_S/V_M": ratio, "triggered": bool(triggered)}
    pen_threshold = pen_threshold if pen_threshold is not None else np.linalg.norm(np.ptp(target_pts, axis=0)) * 0.02

    if not triggered or not cutting:
        # single full-CAD hypothesis (hypothesis OFF path)
        T, info = register_fn(source_pts, source_normals, features, np.eye(4))
        meta["n_hypotheses"] = 1
        return T, info, meta

    cands = make_candidates(features, guide_sets, target_pts, target_ratio)
    meta["n_hypotheses"] = len(cands) + 1
    best_T, best_score, best_info = None, np.inf, None
    src_c = source_pts.mean(0)
    # include full-CAD as one hypothesis too
    trial_sets = [list(range(len(features)))] + cands
    for idx_set in trial_sets:
        sub = [features[i] for i in idx_set]
        if len(sub) < 4:
            continue
        # COARSE INIT: centroid-align the candidate's guiding region to the source
        cand_c = np.vstack([guide_sets[i] for i in idx_set]).mean(0)
        init_T = np.eye(4); init_T[:3, 3] = src_c - cand_c
        T, info = register_fn(source_pts, source_normals, sub, init_T)
        rmse = info.get("history", [{}])[-1].get("rmse", 0.0) if info.get("history") else 0.0
        score = rmse
        if penetration:
            aligned = _transform(source_pts, T)
            pen = penetration_fraction(aligned, target_pts, pen_threshold)
            score = rmse + 0.5 * pen          # lambda_pen balance
        if score < best_score:
            best_score, best_T, best_info = score, T, info
    return best_T, best_info, meta
